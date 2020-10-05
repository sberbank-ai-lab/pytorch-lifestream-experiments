import json
import logging
import sys

import torch
from sklearn.model_selection import StratifiedKFold

import numpy as np
from dltranz.metric_learn.inference_tools import infer_part_of_data, save_scores
from dltranz.seq_encoder import Squeeze, LastStepEncoder
from dltranz.trx_encoder import TrxEncoder
from dltranz.util import init_logger, get_conf
from scenario_rosbank.fit_target import create_ds, run_experiment, read_consumer_data, score_data

logger = logging.getLogger(__name__)


def load_model(conf):
    pretrained_model_path = conf['pretrained_model_path']

    pre_model = torch.load(pretrained_model_path, map_location=torch.device("cpu"))
    if not isinstance(pre_model[0], TrxEncoder):
        pre_model = pre_model[0]
    trx_encoder = pre_model[0]
    rnn_encoder = pre_model[1]
    step_select_encoder = pre_model[2]

    model_type = conf['model_type']
    if model_type == 'rnn':
        input_size = conf['rnn.hidden_size']
    elif model_type == 'transf':
        input_size = conf['transf.input_size']
    else:
        raise NotImplementedError(f'NotImplementedError for model_type="{model_type}"')

    head_output_size = 1

    layers = [
        trx_encoder,
        rnn_encoder,
        step_select_encoder,
    ]
    if conf['use_batch_norm']:
        layers.append(torch.nn.BatchNorm1d(input_size))

    layers.extend([
        torch.nn.Linear(input_size, head_output_size),
        torch.nn.Sigmoid(),
        Squeeze(),
    ])

    model = torch.nn.Sequential(*layers)
    return model


def prepare_parser(parser):
    pass


def main(_):
    init_logger(__name__)
    init_logger('dltranz')
    init_logger('metric_learning')

    conf = get_conf(sys.argv[2:])
    # min_seq_len = conf['dataset'].get('min_seq_len', 1)
    min_seq_len = 1

    model_f = load_model
    train_data = read_consumer_data(conf['dataset.train_path'], conf)
    test_data = read_consumer_data(conf['dataset.test_path'], conf)

    # train
    results = []

    skf = StratifiedKFold(conf['cv_n_split'])
    nrows = conf['params'].get('labeled_amount',-1) # semi-supervised setup. default = supervised

    target_values = [rec['target'] for rec in train_data]
    for i, (i_train, i_valid) in enumerate(skf.split(train_data, target_values)):
        logger.info(f'Train fold: {i}')
        i_train_data = [rec for i, rec in enumerate(train_data)
                        if i in i_train and len(rec['event_time']) >= min_seq_len]
        i_valid_data = [rec for i, rec in enumerate(train_data) if i in i_valid]

        if nrows > 0: i_train_data = i_train_data[:nrows]

        train_ds, valid_ds = create_ds(i_train_data, i_valid_data, conf)
        model, _ = run_experiment(train_ds, valid_ds, conf['params'], model_f)
        result = {
          "fold_id": i,
          "model_name": conf.get('stats.model_name', None),
          "feature_name": conf.get('stats.feature_name', None),
        }

        # inference
        columns = conf['output.columns']

        train_scores = infer_part_of_data(i, i_train_data, columns, model, conf)
        save_scores(train_scores, i, conf['output.valid'])
        result['scores_train'] = score_data(conf, i_train_data, train_scores)

        train_scores = infer_part_of_data(i, i_valid_data, columns, model, conf)
        save_scores(train_scores, i, conf['output.valid'])
        result['scores_valid'] = score_data(conf, i_valid_data, train_scores)

        print(f"Fold {i:2d}: cores train: {result['scores_train']['auroc']:.3f}, valid: {result['scores_valid']['auroc']:.3f}")

        test_scores = infer_part_of_data(i, test_data, columns, model, conf)
        save_scores(test_scores, i, conf['output.test'])
        result['scores_test'] = score_data(conf, test_data, test_scores)

        results.append(result)

    # results
    stats_file = conf.get('stats.path', None)
    if stats_file is not None:
        with open(stats_file, 'w') as f:
            json.dump(results, f)

    valid_scores = np.array([x['scores_valid']['auroc'] for x in results])
    print(f'Valid scores: {valid_scores.mean():.3f} [{valid_scores}]')
