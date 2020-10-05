import logging
import os
import random
from itertools import islice

import numpy as np
import torch
from torch.utils.data import DataLoader

from tqdm.auto import tqdm
from dltranz.data_load import ConvertingTrxDataset, DropoutTrxDataset, read_data_gen, AllTimeShuffleMLDataset
from dltranz.experiment import update_model_stats
from dltranz.metric_learn.dataset import SplittingDataset, split_strategy
from dltranz.metric_learn.dataset import TargetEnumeratorDataset, collate_splitted_rows
from dltranz.metric_learn.losses import get_loss
from dltranz.metric_learn.metric import BatchRecallTop
from dltranz.metric_learn.ml_models import ml_model_by_type
from dltranz.metric_learn.sampling_strategies import get_sampling_strategy
from dltranz.train import get_optimizer, get_lr_scheduler, fit_model, CheckpointHandler
from dltranz.util import init_logger, get_conf, switch_reproducibility_on

logger = logging.getLogger(__name__)

if __name__ == '__main__':
    switch_reproducibility_on()


def prepare_embeddings(seq, conf, is_train):
    min_seq_len = conf['dataset'].get('min_seq_len', 1)
    embeddings = list(conf['params.trx_encoder.embeddings'].keys())

    feature_keys = embeddings + list(conf['params.trx_encoder.numeric_values'].keys())

    for rec in seq:
        seq_len = len(rec['event_time'])
        if is_train and seq_len < min_seq_len:
            continue

        if 'feature_arrays' in rec:
            feature_arrays = rec['feature_arrays']
            feature_arrays = {k: v for k, v in feature_arrays.items() if k in feature_keys}
        else:
            feature_arrays = {k: v for k, v in rec.items() if k in feature_keys}

        # TODO: datetime processing. Take date-time features

        # shift embeddings to 1, 0 is padding value
        feature_arrays = {k: v + (1 if k in embeddings else 0) for k, v in feature_arrays.items()}

        # clip embeddings dictionary by max value
        for e_name, e_params in conf['params.trx_encoder.embeddings'].items():
            feature_arrays[e_name] = feature_arrays[e_name].clip(0, e_params['in'] - 1)

        feature_arrays['event_time'] = rec['event_time']

        rec['feature_arrays'] = feature_arrays
        yield rec


def shuffle_client_list_reproducible(conf, data):
    if conf['dataset.client_list_shuffle_seed'] != 0:
        dataset_col_id = conf['dataset'].get('col_id', 'client_id')
        data = sorted(data, key=lambda x: x.get(dataset_col_id, x.get('customer_id', x.get('installation_id'))))
        random.Random(conf['dataset.client_list_shuffle_seed']).shuffle(data)
    return data


def prepare_data(conf):
    data = read_data_gen(conf['dataset.train_path'])
    data = tqdm(data)
    if 'max_rows' in conf['dataset']:
        data = islice(data, conf['dataset.max_rows'])
    data = prepare_embeddings(data, conf, is_train=True)
    data = shuffle_client_list_reproducible(conf, data)
    data = list(data)
    if 'client_list_keep_count' in conf['dataset']:
        data = data[:conf['dataset.client_list_keep_count']]

    valid_ix = np.arange(len(data))
    valid_ix = np.random.choice(valid_ix, size=int(len(data) * conf['dataset.valid_size']), replace=False)
    valid_ix = set(valid_ix.tolist())

    logger.info(f'Loaded {len(data)} rows. Split in progress...')
    train_data = [rec for i, rec in enumerate(data) if i not in valid_ix]
    valid_data = [rec for i, rec in enumerate(data) if i in valid_ix]

    logger.info(f'Train data len: {len(train_data)}, Valid data len: {len(valid_data)}')

    return train_data, valid_data


def create_data_loaders(conf):
    train_data, valid_data = prepare_data(conf)

    train_dataset = SplittingDataset(
        train_data,
        split_strategy.create(**conf['params.train.split_strategy'])
    )
    train_dataset = TargetEnumeratorDataset(train_dataset)
    train_dataset = ConvertingTrxDataset(train_dataset)
    train_dataset = DropoutTrxDataset(train_dataset, trx_dropout=conf['params.train.trx_dropout'],
                                      seq_len=conf['params.train.max_seq_len'])

    if conf['params.train'].get('all_time_shuffle',False):
        train_dataset = AllTimeShuffleMLDataset(train_dataset)
        logger.info('AllTimeShuffle used')

    train_loader = DataLoader(
        dataset=train_dataset,
        shuffle=True,
        collate_fn=collate_splitted_rows,
        num_workers=conf['params.train'].get('num_workers', 0),
        batch_size=conf['params.train.batch_size'],
    )

    valid_dataset = SplittingDataset(
        valid_data,
        split_strategy.create(**conf['params.valid.split_strategy'])
    )
    valid_dataset = TargetEnumeratorDataset(valid_dataset)
    valid_dataset = ConvertingTrxDataset(valid_dataset)
    valid_dataset = DropoutTrxDataset(valid_dataset, trx_dropout=0.0,
                                      seq_len=conf['params.valid.max_seq_len'])
    valid_loader = DataLoader(
        dataset=valid_dataset,
        shuffle=False,
        collate_fn=collate_splitted_rows,
        num_workers=conf['params.valid'].get('num_workers', 0),
        batch_size=conf['params.valid.batch_size'],
    )

    return train_loader, valid_loader


def run_experiment(model, conf):
    import time
    start = time.time()

    stats_file = conf['stats.path']
    params = conf['params']

    train_loader, valid_loader = create_data_loaders(conf)

    sampling_strategy = get_sampling_strategy(params)
    loss = get_loss(params, sampling_strategy)

    valid_metric = {'BatchRecallTop': BatchRecallTop(k=params['valid.split_strategy.split_count'] - 1)}
    optimizer = get_optimizer(model, params)
    scheduler = get_lr_scheduler(optimizer, params)

    train_handlers = []
    if 'checkpoints' in conf['params.train']:
        checkpoint = CheckpointHandler(
            model=model,
            **conf['params.train.checkpoints']
        )
        train_handlers.append(checkpoint)

    metric_values = fit_model(model, train_loader, valid_loader, loss, optimizer, scheduler, params, valid_metric,
                              train_handlers=train_handlers)

    exec_sec = time.time() - start

    if conf.get('save_model', False):
        save_dir = os.path.dirname(conf['model_path.model'])
        os.makedirs(save_dir, exist_ok=True)

        m_encoder = model[0] if conf['model_path.only_encoder'] else model

        torch.save(m_encoder, conf['model_path.model'])
        logger.info(f'Model saved to "{conf["model_path.model"]}"')

    results = {
        'exec-sec': exec_sec,
        'Recall_top_K': metric_values,
    }

    if conf.get('log_results', True):
        update_model_stats(stats_file, params, results)


def main(args=None):
    conf = get_conf(args)

    model_f = ml_model_by_type(conf['params.model_type'])
    model = model_f(conf['params'])

    return run_experiment(model, conf)


if __name__ == '__main__':
    init_logger(__name__)
    init_logger('dltranz')
    init_logger('dataset_preparation')

    main()
