import logging
from random import Random

import torch

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from scenario_tinkoff.data import load_data, get_hist_count, get_encoder
from scenario_tinkoff.feature_preparation import load_user_features, load_item_features, COL_ID
from scenario_tinkoff.history_file import save_result
from scenario_tinkoff.metrics import tinkoff_reward
from scenario_tinkoff.models import StoriesRecModel, model_inspection

logger = logging.getLogger(__name__)


def prepare_parser(sub_parser):
    sub_parser.add_argument('--name', required=True, type=str)
    sub_parser.add_argument('--cv_n_folds', type=int, default=5)
    sub_parser.add_argument('--cv_salt', type=int, default=42)

    sub_parser.add_argument('--model_type', default='nn', choices=['nn'])
    sub_parser.add_argument('--device', type=str, default='cuda')
    sub_parser.add_argument('--max_epoch', type=int, default=5)
    sub_parser.add_argument('--optim_weight_decay', type=float, nargs='+', default=[0.0001])
    sub_parser.add_argument('--optim_lr', type=float, default=0.01)
    sub_parser.add_argument('--lr_step_size', type=int, default=1)
    sub_parser.add_argument('--lr_step_gamma', type=float, default=0.5)
    sub_parser.add_argument('--loss', type=str, default='mse', choices=['mae', 'mse'])
    sub_parser.add_argument('--model_inspection_off', action='store_true')

    sub_parser.add_argument('--user_layers', type=str)
    sub_parser.add_argument('--item_layers', type=str)

    sub_parser.add_argument('--use_user_popular_features', action='store_true')
    sub_parser.add_argument('--use_trans_common_features', action='store_true')
    sub_parser.add_argument('--use_gender', action='store_true')
    sub_parser.add_argument('--use_trans_mcc_features', action='store_true')
    #
    sub_parser.add_argument('--use_embeddings', action='store_true')
    sub_parser.add_argument('--embedding_file_name', default="embeddings.pickle")

    sub_parser.add_argument('--use_item_popular_features', action='store_true')
    #
    sub_parser.add_argument('--train_batch_size', type=int, default=128)
    sub_parser.add_argument('--valid_batch_size', type=int, default=5000)
    sub_parser.add_argument('--train_num_workers', type=int, default=8)
    sub_parser.add_argument('--valid_num_workers', type=int, default=8)


def train_test_model(config, fold_n, df_train, df_valid, df_test):
    logger.info(f'Fold {fold_n}. train: {len(df_train)}, valid: {len(df_valid)}, test: {len(df_test)}')
    device = torch.device(config.device)

    df_train_hist_count = get_hist_count(df_train)
    df_valid = pd.merge(df_valid, df_train_hist_count, on=COL_ID, how='left')
    df_valid['hist_count'] = df_valid['hist_count'].fillna(0)
    df_test = pd.merge(df_test, df_train_hist_count, on=COL_ID, how='left')
    df_test['hist_count'] = df_test['hist_count'].fillna(0)

    df_users = load_user_features(config, df_train)
    df_items = load_item_features(config, df_train)

    if config.model_type == 'nn':
        user_encoder = get_encoder(df_train, COL_ID, min_count=2)
        item_encoder = get_encoder(df_train, 'story_id', min_count=10)

        model = StoriesRecModel(
            user_layers=config.user_layers,
            user_fixed_vector_size=0 if df_users is None else df_users.size,
            user_encoder=user_encoder,
            df_users=df_users,
            item_layers=config.item_layers,
            item_fixed_vector_size=0 if df_items is None else df_items.size,
            item_encoder=item_encoder,
            df_items=df_items,
            config=config,
            device=device,
        )

        def valid_fn():
            if not config.model_inspection_off:
                model_inspection(model)

            return {
                'train_reward': tinkoff_reward(model.model_predict(df_train)),
                'valid_reward': tinkoff_reward(model.model_predict(df_valid)),
            }

        model.add_valid_fn(valid_fn)
    else:
        raise NotImplementedError(f'Not implemented for model_type: {config.model_type}')

    train_metrics = model.model_train(df_log=df_train)

    scores = {
        'oof_reward': tinkoff_reward(model.model_predict(df_valid)),
        'test_reward': tinkoff_reward(model.model_predict(df_test)),
    }

    for k, v in scores.items():
        logger.info(f'{model.__class__.__name__} _valid_predict {k}: {v:.4f}')

    save_result(config, fold_n, scores, train_metrics)

    return model


class GroupHashSplit:
    def __init__(self, n_splits, salt):
        self.n_splits = n_splits
        self.salt = salt

    def get_clients(self, X, groups):
        unique_clients = set(cl_id for cl_id in groups)

        client_list = (cl_id for cl_id in unique_clients)
        client_list = sorted(client_list)
        client_list = np.array([cl_id for cl_id in client_list])
        Random(self.salt).shuffle(client_list)

        n = len(client_list)
        fold_len = (n + self.n_splits - 1) // self.n_splits

        for i in range(self.n_splits):
            yield client_list[i * fold_len: (i + 1) * fold_len]

    def split(self, X, groups):
        ix = np.arange(len(X))
        for client_ids in self.get_clients(X, groups):
            test_ix = groups.isin(client_ids)
            yield ix[~test_ix], ix[test_ix]


def main(config):
    df_log_train, df_log_test = load_data(config)
    cv = GroupHashSplit(n_splits=config.cv_n_folds, salt=config.cv_salt)

    for fold_n, (i_train, i_valid) in enumerate(cv.split(df_log_train, groups=df_log_train[COL_ID])):
        logger.info(f'=== Train fold: {fold_n:3} ===')
        df_train = df_log_train.iloc[i_train].copy()
        df_valid = df_log_train.iloc[i_valid].copy()

        train_test_model(config, fold_n, df_train, df_valid, df_log_test)
