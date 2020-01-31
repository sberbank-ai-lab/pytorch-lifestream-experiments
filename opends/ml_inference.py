if __name__ == '__main__':
    import sys
    sys.path.append('../')

import logging
import pickle
import random

import numpy as np
import pandas as pd
import torch

from tqdm.auto import tqdm

from dltranz.metric_learn.inference_tools import load_model, score_part_of_data, save_scores
from dltranz.util import init_logger, get_conf
from metric_learning import prepare_embeddings

logger = logging.getLogger(__name__)

if __name__ == '__main__':
    # reproducibility
    np.random.seed(42)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)


def clip_transactions(seq, conf):
    min_len = conf['min_len']
    max_len = conf['max_len']

    for rec in seq:
        seq_len = len(rec['event_time'])
        if seq_len <= 5:
            yield rec
            continue

        new_len = random.randint(min_len, max_len)
        if new_len >= seq_len:
            yield rec
            continue

        avail_pos = seq_len - new_len
        pos = random.randint(0, avail_pos)

        rec['feature_arrays'] = {k: v[pos:pos + new_len] for k, v in rec['feature_arrays'].items()}
        rec['event_time'] = rec['event_time'][pos: pos + new_len]

        yield rec


def read_dataset(path, conf):
    with open(path, 'rb') as f:
        data = pickle.load(f)

    logger.info(f'loaded {len(data)} records')

    for rec in data:
        rec['target'] = -1

    data = prepare_embeddings(data, conf)
    data = clip_transactions(data, conf['dataset.clip_transactions'])

    return list(data)


def make_trx_features(data, conf):
    _num_percentile_list = [0, 10, 25, 50, 75, 90, 100]

    def _num_features(col, val):
        val_orig = np.expm1(abs(val)) * np.sign(val)
        return {
            f'{col}_count': len(val),
            f'{col}_sum': val_orig.sum(),
            f'{col}_std': val_orig.std(),
            f'{col}_mean': val.mean(),
            **{f'{col}_p{p_level}': val
               for val, p_level in zip(np.percentile(val, _num_percentile_list), _num_percentile_list)}
        }

    def _embed_features(col_embed, col_num, val_embed, val_num, size):
        def norm_row(a, agg_func):
            return a / (agg_func(a) + 1e-5)

        val_num_orig = np.expm1(abs(val_num)) * np.sign(val_embed)

        seq_len = len(val_embed)

        m_cnt = np.zeros((seq_len, size), np.float)
        m_sum = np.zeros((seq_len, size), np.float)
        ix = np.arange(seq_len)

        m_cnt[(ix, val_embed)] = 1
        m_sum[(ix, val_embed)] = val_num_orig

        return {
            f'{col_embed}_nunique': np.unique(col_embed).size,
            **{f'{col_embed}_X_{col_num}_{k}_cnt': v for k, v in enumerate(norm_row(m_cnt.sum(axis=0), np.sum))},
            **{f'{col_embed}_X_{col_num}_{k}_sum': v for k, v in enumerate(norm_row(m_sum.sum(axis=0), np.sum))},
            **{f'{col_embed}_X_{col_num}_{k}_std': v for k, v in enumerate(norm_row(m_sum.std(axis=0), np.sum))},
        }

    numeric_values = conf['params.trx_encoder.numeric_values']
    embeddings = conf['params.trx_encoder.embeddings']

    df_data = []
    for rec in tqdm(data, desc='Feature preparation'):
        # labels and target
        features = {k: v for k, v in rec.items() if k not in ('event_time', 'feature_arrays')}
        event_time = rec['event_time']
        feature_arrays = rec['feature_arrays']

        # trans_common_features
        for col_num in numeric_values.keys():
            features.update(_num_features(col_num, feature_arrays[col_num]))

        # embeddings features (like mcc)
        for col_embed, options in embeddings.items():
            for col_num in numeric_values.keys():
                features.update(_embed_features(
                    col_embed, col_num,
                    feature_arrays[col_embed], feature_arrays[col_num],
                    options['in'])
                )

        df_data.append(features)

    return pd.DataFrame(df_data)


def main(args=None):
    conf = get_conf(args)

    model = load_model(conf)
    columns = conf['output.columns']

    train_data = read_dataset(conf['dataset.train_path'], conf)
    if conf['dataset'].get('test_path', None) is not None:
        test_data = read_dataset(conf['dataset.test_path'], conf)
    else:
        test_data = []
    all_data = train_data + test_data

    score_part_of_data(None, all_data, columns, model, conf)

    df_trx_features = make_trx_features(all_data, conf)
    save_scores(df_trx_features, None, conf['trx_features'])


if __name__ == '__main__':
    init_logger(__name__)
    init_logger('dltranz')
    init_logger('retail_embeddings_projects.embedding_tools')

    main(None)
