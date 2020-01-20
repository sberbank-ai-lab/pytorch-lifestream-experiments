import datetime
import logging
import math
import pickle
import sys
import random
from copy import deepcopy

import numpy as np
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset, WeightedRandomSampler
from torch.utils.data.dataloader import DataLoader

from dltranz.data_load import TrxDataset, ConvertingTrxDataset, DropoutTrxDataset, padded_collate, \
    create_validation_loader
from dltranz.loss import get_loss
from dltranz.models import model_by_type
from dltranz.train import get_optimizer, get_lr_scheduler, fit_model
from dltranz.util import init_logger, get_conf
from dltranz.experiment import get_epoch_score_metric, update_model_stats
from dltranz.metric_learn.inference_tools import score_part_of_data
from metric_learning import prepare_embeddings

logger = logging.getLogger(__name__)


def read_consumer_data(conf):
    logger.info(f'Data loading...')

    with open(conf['dataset.path'], 'rb') as f:
        data = pickle.load(f)
    logger.info(f'Loaded raw data: {len(data)}')

    data = [rec for rec in data if rec['target'] is not None]
    logger.info(f'Loaded data with target: {len(data)}')

    data = list(prepare_embeddings(data, conf))
    logger.info(f'Fit data to config')

    return data


class ClippingDataset(Dataset):
    def __init__(self, delegate, min_len=250, max_len=350, rate_for_min=0.9):
        super().__init__()

        self.delegate = delegate
        self.min_len = min_len
        self.max_len = max_len
        self.rate_for_min = rate_for_min

    def __len__(self):
        return len(self.delegate)

    def __getitem__(self, item):
        item = self.delegate[item]

        seq_len = len(item['event_time'])
        if seq_len <= 5:
            return item

        new_len = random.randint(self.min_len, self.max_len)
        if new_len > seq_len * self.rate_for_min:
            new_len = math.ceil(seq_len * self.rate_for_min)

        avail_pos = seq_len - new_len
        pos = random.randint(0, avail_pos)

        item = deepcopy(item)
        item['feature_arrays'] = {k: v[pos:pos+new_len] for k, v in item['feature_arrays'].items()}
        item['event_time'] = item['event_time'][pos:pos+new_len]
        return item


class FlipDataset(Dataset):
    def __init__(self, delegate, p=0.25):
        super().__init__()

        self.delegate = delegate
        self.p = p

    def __len__(self):
        return len(self.delegate)

    def __getitem__(self, item):
        item = self.delegate[item]

        if random.random() > self.p:
            return item

        item = deepcopy(item)
        item['feature_arrays'] = {k: deepcopy(v[::-1]) for k, v in item['feature_arrays'].items()}
        item['event_time'] = deepcopy(item['event_time'][::-1])
        return item


class ShiftDataSet(Dataset):
    def __init__(self, delegate, p=0.25):
        super().__init__()

        self.delegate = delegate
        self.p = p

    def __len__(self):
        return len(self.delegate)

    def __getitem__(self, item):
        item = self.delegate[item]

        if random.random() > self.p:
            return item

        seq_len = len(item['event_time'])
        if seq_len <= 5:
            return item

        item = deepcopy(item)
        split_pos = random.randint(int(seq_len * 0.1), int(seq_len * 0.9))

        item['feature_arrays'] = {k: np.concatenate([v[split_pos:], v[:split_pos]])
                                  for k, v in item['feature_arrays'].items()}
        item['event_time'] = np.concatenate([item['event_time'][split_pos:], item['event_time'][:split_pos]])
        return item


def create_ds(train_data, valid_data, conf):
    if 'clip_seq' in conf['params.train']:
        train_data = ClippingDataset(train_data,
                                     min_len=conf['params.train.clip_seq.min_len'],
                                     max_len=conf['params.train.clip_seq.max_len'],
                                     )
    train_data = FlipDataset(train_data, p=0.0)
    train_data = ShiftDataSet(train_data, p=0.0)

    train_ds = ConvertingTrxDataset(TrxDataset(train_data, y_dtype=np.int64))
    valid_ds = ConvertingTrxDataset(TrxDataset(valid_data, y_dtype=np.int64))

    return train_ds, valid_ds


def get_sampler(data):
    norm_lens = [max(min(len(x['event_time']), 2000), 250) for x in data]
    max_mcc = [np.percentile(x['feature_arrays']['mcc_code'], 90) for x in data]

    weights = [(x / 2000) ** 0.5 for x in norm_lens]
    num_samples = int(sum([(x // 250) ** 0.5 for x in norm_lens]))
    return WeightedRandomSampler(weights=weights, num_samples=num_samples, replacement=True)


def run_experiment(train_ds, valid_ds, params, model_f, train_sampler=None):
    model = model_f(params)

    train_ds = DropoutTrxDataset(train_ds, params['train.trx_dropout'], params['train.max_seq_len'])
    train_loader = DataLoader(
        train_ds,
        batch_size=params['train.batch_size'],
        shuffle=True if train_sampler is None else False,
        sampler=train_sampler,
        num_workers=params['train.num_workers'],
        collate_fn=padded_collate)

    valid_loader = create_validation_loader(valid_ds, params['valid'])

    loss = get_loss(params)
    opt = get_optimizer(model, params)
    scheduler = get_lr_scheduler(opt, params)

    metric_name = params['score_metric']
    metrics = {metric_name: get_epoch_score_metric(metric_name)()}
    handlers = []

    scores = fit_model(model, train_loader, valid_loader, loss, opt, scheduler, params, metrics, handlers)

    return model, {
        **scores,
        'finish_time': datetime.datetime.now().isoformat(),
    }


def prepare_parser(parser):
    pass


def main(_):
    init_logger(__name__)
    init_logger('dltranz')
    init_logger('metric_learning')

    conf = get_conf(sys.argv[2:])

    model_f = model_by_type(conf['params.model_type'])
    all_data = read_consumer_data(conf)

    # train
    results = []

    skf = StratifiedKFold(conf['cv_n_split'])
    target_values = [rec['target'] for rec in all_data]
    for i, (i_train, i_valid) in enumerate(skf.split(all_data, target_values)):
        logger.info(f'Train fold: {i}')
        i_train_data = [rec for i, rec in enumerate(all_data) if i in i_train and random.random() <= 1.0]
        i_valid_data = [rec for i, rec in enumerate(all_data) if i in i_valid]

        train_ds, valid_ds = create_ds(i_train_data, i_valid_data, conf)
        model, result = run_experiment(train_ds, valid_ds, conf['params'], model_f,
                                       train_sampler=get_sampler(i_train_data))

        # inference
        columns = conf['output.columns']
        score_part_of_data(i, i_valid_data, columns, model, conf)

        results.append(result)

    # results
    stats_file = conf.get('stats.path', None)
    if stats_file is not None:
        update_model_stats(stats_file, conf, results)
