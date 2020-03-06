import logging
from functools import partial, reduce
from operator import iadd

import numpy as np
import pandas as pd
import os
from sklearn.metrics import make_scorer, accuracy_score

import dltranz.scenario_cls_tools as sct
from scenario_age_pred.const import (
    DEFAULT_DATA_PATH, DEFAULT_RESULT_FILE, PRIVATE_FILE, PUBLIC_FILE, DATASET_FILE, COL_ID, COL_TARGET
)
from scenario_age_pred.features import load_features, load_scores

logger = logging.getLogger(__name__)


def prepare_parser(parser):
    sct.prepare_common_parser(parser, data_path=DEFAULT_DATA_PATH, output_file=DEFAULT_RESULT_FILE)


def get_scores(args):
    name, conf, params, public_target_, private_target_ = args

    logger.info(f'[{name}] Scoring started: {params}')

    result, scores = [], []
    valid_scores, test_scores = load_scores(conf, **params)
    for fold_n, both_folds in enumerate(test_scores):
        public_target = public_target_.copy()
        private_target = private_target_.copy()

        both_folds['pred'] = np.argmax(both_folds.values, 1)

        valid_fold = public_target.merge(both_folds, on=COL_ID, how='left')
        test_fold = private_target.merge(both_folds, on=COL_ID, how='left')

        result.append({
            'name': name,
            'fold_n': fold_n,
            'oof_accuracy': (valid_fold['pred'] == valid_fold[COL_TARGET]).mean(),
            'test_accuracy': (test_fold['pred'] == test_fold[COL_TARGET]).mean(),
        })

        preds = pd.DataFrame({
            **{
                f'preds_valid_{i}': valid_fold[f'v00{i}'].values for i in range(4)
            },
            **{
                f'preds_test_{i}': test_fold[f'v00{i}'].values for i in range(4)
            },
            'name': name,
            'y_valid': valid_fold[COL_TARGET].values,
            'y_test': test_fold[COL_TARGET].values,
            'ids': list(range(len(valid_fold)))
        })
        scores.append(preds)

    preds = pd.concat(scores).groupby(['name','ids']) \
            ['preds_valid_0','preds_valid_1','preds_valid_2','preds_valid_3',
            'preds_test_0','preds_test_1','preds_test_2','preds_test_3','y_valid','y_test'].mean().reset_index(drop=False)
    preds['public_accuracy'] = np.argmax(preds[[f'preds_valid_{i}' for i in range(4)]].values, axis=1) == preds['y_valid']
    preds['private_accuracy'] = np.argmax(preds[[f'preds_test_{i}' for i in range(4)]].values, axis=1) == preds['y_test']
    preds.drop(columns = [f'preds_valid_{i}' for i in range(4)]+[f'preds_test_{i}' for i in range(4)], inplace=True)
    preds = preds.groupby(['name'])['public_accuracy','private_accuracy'].mean().reset_index(drop=False)

    return result, preds

def read_train_test(data_path, dataset_file, public_file, private_file, col_id):
    target = pd.concat([
        pd.read_csv(os.path.join(data_path, dataset_file)),
        pd.read_csv(os.path.join(data_path, public_file)),
        pd.read_csv(os.path.join(data_path, private_file))
    ])
    public_ids = set(pd.read_csv(os.path.join(data_path, public_file))[col_id].tolist())
    private_ids = set(pd.read_csv(os.path.join(data_path, private_file))[col_id].tolist())

    ix_public = target[col_id].isin(public_ids)
    ix_private = target[col_id].isin(private_ids)
    ix_test = target[col_id].isin(public_ids.union(private_ids))

    logger.info(f'Train size: {(~ix_test).sum()} clients')
    logger.info(f'Public size: {ix_public.sum()} clients')
    logger.info(f'Private size: {ix_private.sum()} clients')

    return target[~ix_test].set_index(col_id), target[ix_public].set_index(col_id), target[ix_private].set_index(col_id)

def main(conf):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-7s %(funcName)-20s   : %(message)s')

    approaches_to_train = {
        **{
            f"embeds: {file_name}": {'metric_learning_embedding_name': file_name}
            for file_name in conf['embedding_file_names']
        },
        # **{
        #     f"embeds: {file_name}": {'target_scores_name': file_name}
        #     for file_name in conf['score_file_names']
        # },
    }
    if conf['add_baselines']:
        approaches_to_train.update({
            'baseline': {'use_client_agg': True, 'use_small_group_stat': True},
        })

    if conf['add_emb_baselines']:
        approaches_to_train.update({
            f"embeds: {file_name} and baseline": {
                'metric_learning_embedding_name': file_name, 'use_client_agg': True, 'use_small_group_stat': True}
            for file_name in conf['embedding_file_names']
        })
        approaches_to_train.update({
            f"embeds: {file_name} and baseline": {
                'target_scores_name': file_name, 'use_client_agg': True, 'use_small_group_stat': True}
            for file_name in conf['score_file_names']
        })

    approaches_to_score = {
        f"scores: {file_name}": {'target_scores_name': file_name}
        for file_name in conf['score_file_names']
    }

    pool = sct.WPool(processes=conf['n_workers'])
    df_results = None
    df_scores = None

    df_target, public_target, private_target = read_train_test(conf['data_path'], DATASET_FILE, PUBLIC_FILE, PRIVATE_FILE, COL_ID)
    if len(approaches_to_train) > 0:
        folds = sct.get_folds(df_target, COL_TARGET, conf['cv_n_split'], conf['random_state'], conf.get('labeled_amount',-1))

        model_types = {
            'xgb': dict(
                objective='multi:softprob',
                num_class=4,
                n_jobs=4,
                seed=conf['model_seed'],
                n_estimators=300,
            ),
            'linear': dict(),
            'lgb': dict(
                n_estimators=1200,
                boosting_type='gbdt',
                objective='multiclass',
                num_class=4,
                metric='multi_error',
                learning_rate=0.02,
                subsample=0.75,
                subsample_freq=1,
                feature_fraction=0.75,
                max_depth=12,
                lambda_l1=1,
                lambda_l2=1,
                min_data_in_leaf=50,
                num_leaves=50,
                #random_state=conf['model_seed'],
                n_jobs=2,
            ),
        }

        # train and score models
        args_list = [sct.KWParamsTrainAndScore(
            name=name,
            fold_n=fold_n,
            load_features_f=partial(load_features, conf=conf, **params),
            model_type=model_type,
            model_params=model_params,
            scorer_name='accuracy',
            scorer=make_scorer(accuracy_score),
            col_target=COL_TARGET,
            df_train=df_target.sample(frac=1),
            df_valid=public_target,
            df_test=private_target,
        )
            for name, params in approaches_to_train.items()
            for fold_n, (train_target, valid_target) in enumerate(folds)
            for model_type, model_params in model_types.items() if model_type in conf['models']
        ]
        results, preds = [], []
        for i, (r, p) in enumerate(pool.imap_unordered(sct.train_and_score, args_list)):
            results.append(r)
            preds.append(p)
            logger.info(f'Done {i + 1:4d} from {len(args_list)}')
        df_results = pd.DataFrame(results).set_index('name')[['oof_accuracy', 'test_accuracy']]

        # ensemble
        preds = pd.concat(preds).groupby(['name','ids']) \
            ['preds_valid_0','preds_valid_1','preds_valid_2','preds_valid_3',
            'preds_test_0','preds_test_1','preds_test_2','preds_test_3','y_valid','y_test'].mean().reset_index(drop=False)
        preds['public_accuracy'] = np.argmax(preds[[f'preds_valid_{i}' for i in range(4)]].values, axis=1) == preds['y_valid']
        preds['private_accuracy'] = np.argmax(preds[[f'preds_test_{i}' for i in range(4)]].values, axis=1) == preds['y_test']
        preds.drop(columns = [f'preds_valid_{i}' for i in range(4)]+[f'preds_test_{i}' for i in range(4)], inplace=True)
        preds = preds.groupby(['name'])['public_accuracy','private_accuracy'].mean().reset_index(drop=False)
    else:
        preds = None

    if len(approaches_to_score) > 0:
        # score already trained models on valid and test sets
        args_list = [(name, conf, params, public_target, private_target) for name, params in approaches_to_score.items()]
        temp_res = pool.map(get_scores, args_list)
        results = reduce(iadd, [x[0] for x in temp_res])
        preds_scored= pd.concat([x[1] for x in temp_res])
        df_scores = pd.DataFrame(results).set_index('name')[['oof_accuracy', 'test_accuracy']]
    else:
        preds_scored = None

    logger.info(f'{pd.concat([x for x in [preds, preds_scored] if x is not None])}')

    # combine results
    df_results = pd.concat([df for df in [df_results, df_scores] if df is not None])
    df_results = sct.group_stat_results(df_results, 'name', ['oof_accuracy', 'test_accuracy'])

    with pd.option_context(
            'display.float_format', '{:.4f}'.format,
            'display.max_columns', None,
            'display.max_rows', None,
            'display.expand_frame_repr', False,
            'display.max_colwidth', 100,
    ):
        logger.info(f'Results:\n{df_results}')
        with open(conf['output_file'], 'w') as f:
            print(df_results, file=f)
