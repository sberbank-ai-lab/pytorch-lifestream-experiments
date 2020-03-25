import logging
import os
import pickle
import numpy as np
import torch

from dltranz.util import init_logger, get_conf
from metric_learning import prepare_embeddings
from dltranz.metric_learn.inference_tools import load_model, score_part_of_data
from dltranz.metric_learn.ml_models import ml_model_by_type

logger = logging.getLogger(__name__)

if __name__ == '__main__':
    # reproducibility
    np.random.seed(42)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)


def read_dataset(path, conf):
    with open(path, 'rb') as f:
        data = pickle.load(f)

    logger.info(f'loaded {len(data)} records')

    for rec in data:
        rec['target'] = -1

    data = list(prepare_embeddings(data, conf))

    return data


def main(args=None):
    conf = get_conf(args)

    model_path = conf['model_path.model']
    if os.path.splitext(model_path)[1] == '.pth':
        # load state_dict from checkpoint
        model_f = ml_model_by_type(conf['params.model_type'])
        model = model_f(conf['params'])
        model_d = load_model(conf)
        model.load_state_dict(model_d)
        logger.info('Model loaded from checkpoint')

        model = model[0] if conf['model_path.only_encoder'] else model
    else:
        # load full model
        model = load_model(conf)
        logger.info('Model loaded as full model')

    columns = conf['output.columns']

    train_data = read_dataset(conf['dataset.train_path'], conf)
    if conf['dataset'].get('test_path',None) is not None:
        test_data = read_dataset(conf['dataset.test_path'], conf)
    else:
        test_data = []
    score_part_of_data(None, train_data+test_data, columns, model, conf)


if __name__ == '__main__':
    init_logger(__name__)
    init_logger('dltranz')
    init_logger('retail_embeddings_projects.embedding_tools')

    main(None)
