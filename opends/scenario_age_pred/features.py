import os
import pickle
from glob import glob

import numpy as np
import pandas as pd

from scenario_age_pred.const import DATASET_FILE, COL_ID


def _random(conf):
    all_clients = pd.read_csv(os.path.join(conf['data_path'], DATASET_FILE)).set_index(COL_ID)
    all_clients = all_clients.assign(random=np.random.rand(len(all_clients)))
    return all_clients[['random']]


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


def __client_agg(conf):
    transactions_train = pd.read_csv(os.path.join(conf['data_path'], 'transactions_train.csv'))
    transactions_test = pd.read_csv(os.path.join(conf['data_path'], 'transactions_test.csv'))
    df_transactions = pd.concat([transactions_train, transactions_test])

    agg_features = pd.concat([
        df_transactions.groupby(COL_ID)['amount_rur'].agg(['sum', 'mean', 'std', 'min', 'max']),
        df_transactions.groupby(COL_ID)['small_group'].nunique().rename('small_group_nunique'),
    ], axis=1)

    return agg_features


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


def _client_agg(conf):
    with open(os.path.join(os.path.join(conf['data_path'], 'train_trx.p')), 'rb') as f1, \
            open(os.path.join(os.path.join(conf['data_path'], 'test_trx.p')), 'rb') as f2:
        data = pickle.load(f1) + pickle.load(f2)

    df_data = []
    for rec in data:
        # labels and target
        features = {k: v for k, v in rec.items() if k == COL_ID}
        feature_arrays = rec['feature_arrays']

        # trans_common_features
        features.update(_num_features('amount_rur', feature_arrays['amount_rur']))

        df_data.append(features)

    return pd.DataFrame(df_data).set_index(COL_ID)


def __small_group_stat(conf):
    transactions_train = pd.read_csv(os.path.join(conf['data_path'], 'transactions_train.csv'))
    transactions_test = pd.read_csv(os.path.join(conf['data_path'], 'transactions_test.csv'))
    df_transactions = pd.concat([transactions_train, transactions_test])

    cat_counts_train = pd.concat([
        df_transactions.pivot_table(
            index=COL_ID, columns='small_group', values='amount_rur', aggfunc='count').fillna(0.0),
        df_transactions.pivot_table(
            index=COL_ID, columns='small_group', values='amount_rur', aggfunc='mean').fillna(0.0),
        df_transactions.pivot_table(
            index=COL_ID, columns='small_group', values='amount_rur', aggfunc='std').fillna(0.0),
    ], axis=1, keys=['small_group_count', 'small_group_mean', 'small_group_std'])

    cat_counts_train.columns = ['_'.join(map(str, c)) for c in cat_counts_train.columns.values]
    return cat_counts_train


def _small_group_stat(conf):
    with open(os.path.join(os.path.join(conf['data_path'], 'train_trx.p')), 'rb') as f1, \
            open(os.path.join(os.path.join(conf['data_path'], 'test_trx.p')), 'rb') as f2:
        data = pickle.load(f1) + pickle.load(f2)

    df_data = []
    for rec in data:
        # labels and target
        features = {k: v for k, v in rec.items() if k == COL_ID}
        feature_arrays = rec['feature_arrays']

        # embeddings features (like mcc)
        features.update(_embed_features(
            'small_group', 'amount_rur',
            feature_arrays['small_group'], feature_arrays['amount_rur'],
            250)
        )

        df_data.append(features)

    return pd.DataFrame(df_data).set_index(COL_ID)


def _metric_learning_embeddings(conf, file_name):
    df = pd.read_pickle(os.path.join(conf['data_path'], file_name)).set_index(COL_ID)
    return df


def load_features(
        conf,
        use_random=False,
        use_client_agg=False,
        use_small_group_stat=False,
        metric_learning_embedding_name=None,
        target_scores_name=None,
):
    features = []
    if use_random:
        features.append(_random(conf))

    if use_client_agg:
        features.append(_client_agg(conf))

    if use_small_group_stat:
        features.append(_small_group_stat(conf))

    if metric_learning_embedding_name is not None:
        features.append(_metric_learning_embeddings(conf, metric_learning_embedding_name))

    return features

def load_scores(conf, target_scores_name):
    valid_files = glob(os.path.join(conf['data_path'], target_scores_name, 'valid', '*'))
    valid_scores = [pd.read_pickle(f).set_index(COL_ID) for f in valid_files]

    test_files = glob(os.path.join(conf['data_path'], target_scores_name, 'test', '*'))
    test_scores = [pd.read_pickle(f).set_index(COL_ID) for f in test_files]

    return valid_scores, test_scores