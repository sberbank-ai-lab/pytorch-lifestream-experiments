import sys
import os
script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(script_dir, '../'))

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql import Window
import numpy as np
from encoders import EncoderCollection, CategoryStringEncoder
import constants as cns
from pyspark.sql import SparkSession
from typing import Optional

from data_load import padded_temporal_collate, RecomendationTemporalClassificationWithFeaturesDataSet
import sparkpickle
import glob
from torch.utils.data import DataLoader
import math


spark_logs = f'/data/molchanov/spark_temp/okko'
spark_ui_port = 5679

os.environ['PYSPARK_SUBMIT_ARGS'] = f"""
  --master local[32]
  --driver-memory 128g
  --conf spark.local.dir={spark_logs}
  --conf spark.driver.maxResultSize=8G
  --conf spark.ui.port={spark_ui_port}
  pyspark-shell
"""
ss = SparkSession.builder.getOrCreate()


# ------------------------------------------------------------------------------------------------------------------------------------
# Some Constants

# override some constants
cns.NUM_WORKERS = 0
cns.EMBEDDING_DIM = 16#16
cns.BATCH_SIZE = 64
cns.RNN_HIDDEN_DIM = 256
cns.FEATURES = {
    #'t_delta_1'          : {'type': 'reg', 'in': 1, 'out': 1},
    #'t_delta_2'          : {'type': 'reg', 'in': 1, 'out': 1},
    #'ts'          : {'type': 'reg', 'in': 1, 'out': 1}
}
cns.N_CEIDS += 1 # because encoders takes 0 and 1 for special needs (all categorical features starts from 2)
cns.SAVE_WEIGHTS_NAME = 'recomendation_model_pickle'
cns.SAVE_MODEL_EACH_EPOCH = True
print(f'\nAttention: ovveride constants: BATCH_SIZE = {cns.BATCH_SIZE}, EMBEDDING_DIM = {cns.EMBEDDING_DIM}, SMALL_DATASET = {cns.B_TEST_DATA}, SAVE_WEIGHTS_NAME = {cns.SAVE_WEIGHTS_NAME}, SAVE_MODEL_EACH_EPOCH = {cns.SAVE_MODEL_EACH_EPOCH}')
print(f'Additional features: {cns.FEATURES}')


# ------------------------------------------------------------------------------------------------------------------------------------
# Convert Data to format suit for temporal models

def encode_cat_features(df):
    dl_encoder = EncoderCollection(
        [CategoryStringEncoder(feature_name, save_stat=True) for feature_name in cns.CATEGORICAL_COLUMNS]
    )
    dl_encoder.fit(df)
    dl_encoder.print_stat()
    dft = dl_encoder.transform(df)

    # save mappings
    features_map = dl_encoder[cns.ELEMENT_ID_COLUMNS].mapping_stat
    features_map[f'orig_{cns.ELEMENT_ID_COLUMNS}'] = features_map.index
    features_map = features_map[[f'orig_{cns.ELEMENT_ID_COLUMNS}', 'code']]
    features_map.loc[features_map.index, 'code'] += 1
    features_map = features_map.rename(columns={'code': cns.ELEMENT_ID_COLUMNS})

    users_map = dl_encoder[cns.USER_ID_COLUMN].mapping_stat
    users_map[f'orig_{cns.USER_ID_COLUMN}'] = users_map.index
    users_map = users_map[[f'orig_{cns.USER_ID_COLUMN}', 'code']]
    users_map.loc[users_map.index, 'code'] += 1
    users_map = users_map.rename(columns={'code': cns.USER_ID_COLUMN})

    features_map.to_csv('./features_map.csv', index=False)
    users_map.to_csv('./users_map.csv', index=False)

    return dft


def aggregate(dft):
    dfg = None
    w = Window.partitionBy(cns.AGGREGATE_COLUMN).orderBy(cns.TIME_COLUMN)

    for col in dft.columns:
        if col == cns.AGGREGATE_COLUMN:
            continue
        tmp = dft.withColumn('tmp', F.collect_list(col).over(w)) \
            .groupBy(cns.AGGREGATE_COLUMN) \
            .agg(F.max('tmp').alias(col))
        if dfg is not None:
            dfg = dfg.join(tmp, on=cns.AGGREGATE_COLUMN)
        else:
            dfg = tmp

    return dfg


def to_final_dataset(dfg):
    def make_prediction_eids(eids):
        pos_eid = np.stack([eids[1:] for i in range(cns.TRAIN_WINDOW)])
        for i in range(1, cns.TRAIN_WINDOW):
            pos_eid[i, :-i] = eids[i + 1:]
        return pos_eid

    def to_features_array(row):
        features_array = {
            'feature_arrays': {k: np.array(v) for k, v in row.asDict().items() if k not in cns.AGGREGATE_COLUMN}
        }
        # prediction eids creation
        features_array['feature_arrays']['pos_ceid'] = make_prediction_eids(
            features_array['feature_arrays'][cns.ELEMENT_ID_COLUMNS])
        features_array[cns.USER_ID_COLUMN] = row[cns.USER_ID_COLUMN]

        # time features
        t = features_array['feature_arrays'][cns.TIME_COLUMN]

        t0 = np.full(t.shape[0], t[0])
        t0[1:] = t[:-1]
        t_delta_1 = t - t0
        t_delta_1 = np.log(1 + t_delta_1)
        features_array['feature_arrays']['t_delta_1'] = t_delta_1

        tend = np.full(t.shape[0], t[-1])
        t_delta_2 = tend - t
        # t_delta_2 = np.log(1 + t_delta_2)
        t_delta_2 = np.sqrt(1 + t_delta_2)
        features_array['feature_arrays']['t_delta_2'] = t_delta_2

        # rename some features
        features_array['feature_arrays']['ceid'] = features_array['feature_arrays'].pop(cns.ELEMENT_ID_COLUMNS)

        return features_array

    data = dfg.rdd.map(lambda x: to_features_array(x))
    data = data.filter(lambda x: next(iter(x['feature_arrays'].values())).size > 3 * cns.TRAIN_WINDOW)

    return data


def convert_data_and_save(df, pth, partitions_count):
    df_cat_encoded = encode_cat_features(df)
    df_aggregated = aggregate(df_cat_encoded)
    dataset = to_final_dataset(df_aggregated)
    dataset.repartition(cns.OUTPUT_PARTITIONS_COUNT).saveAsPickleFile(pth)
    print('DataSet ready\n')


# ------------------------------------------------------------------------------------------------------------------------------------
# Create data loader

class MultipartitionTemporalClassificationWithFeaturesDataLoader:

    def load_part(self, part_idx):

        def load_part_gen(part_path):
            with open(part_path, 'rb') as f:
                for rec in sparkpickle.load_gen(f):
                    yield dict(rec)

        lst_data = list(load_part_gen(self.parts[part_idx]))
        return RecomendationTemporalClassificationWithFeaturesDataSet(lst_data)

    def crete_part_loader(self, part_idx):
        part_data = self.load_part(part_idx)
        return DataLoader(part_data,
                          batch_size=cns.BATCH_SIZE,
                          shuffle=True if self.mode == 'train' else False,
                          num_workers=cns.NUM_WORKERS,
                          collate_fn=padded_temporal_collate)

    # mode: train (suffle data in dataloaders), prediction (not suffle - important for final predictions)
    def __init__(self, partition_pathes, mode, partition_epoch=False):
        self.mode = mode
        self.parts = partition_pathes
        self.curent_part_idx = 0
        self.curent_part_loader = None
        self.partition_epoch = partition_epoch

        # get dataset whole lenght
        self.l = 0
        self.partition_lenghts = []
        for i in range(len(self.parts)):
            tmp = self.load_part(i)
            part_len = int(math.ceil(float(len(tmp)) / cns.BATCH_SIZE))
            self.l += part_len
            self.partition_lenghts.append(part_len)
            del tmp

    def __iter__(self):
        while True:
            if self.curent_part_loader is None:
                self.curent_part_loader = self.crete_part_loader(self.curent_part_idx)
            for val in self.curent_part_loader:
                yield val

            # next part
            self.curent_part_idx += 1
            self.curent_part_loader = None
            if self.partition_epoch:
                break

            # check last part
            if self.curent_part_idx == len(self.parts):
                self.curent_part_idx = 0
                break

    def __len__(self):
        if not self.partition_epoch:
            return self.l
        else:
            return self.partition_lenghts[self.curent_part_idx]


# ------------------------------------------------------------------------------------------------------------------------------------
# Get Data

def get_train_data_loader(log: DataFrame,
                          user_features: Optional[DataFrame],
                          item_features: Optional[DataFrame]):

    if log is not None and user_features is not None and item_features is not None:
        data = log.join(user_features, on=cns.USER_ID_COLUMN)
        data = data.join(item_features, on=cns.ELEMENT_ID_COLUMNS)
        convert_data_and_save(data, cns.DATASET_TRAIN_PATH, cns.OUTPUT_PARTITIONS_COUNT)
    else:
        # preparing test dataset
        #df = ss.read.csv(os.path.join('/data/molchanov/okko', 'transactions.csv'), inferSchema=True, header=True)
        #convert_data_and_save(df, cns.DATASET_TRAIN_PATH, cns.OUTPUT_PARTITIONS_COUNT)
        print('Attention: using existing dataset')

    data_partitions_pathes = glob.glob(f'{cns.DATASET_TRAIN_PATH}/part-*')
    return MultipartitionTemporalClassificationWithFeaturesDataLoader(data_partitions_pathes, mode='train', partition_epoch=False),\
           MultipartitionTemporalClassificationWithFeaturesDataLoader(data_partitions_pathes, mode='train', partition_epoch=False)


def get_validation_data_loader(users: DataFrame,
                               log: DataFrame,
                               user_features: Optional[DataFrame],
                               item_features: Optional[DataFrame]):

    if log is not None and user_features is not None and item_features is not None and users is not None:
        log = users.join(log, on=cns.USER_ID_COLUMN)
        data = log.join(user_features, on=cns.USER_ID_COLUMN)
        data = data.join(item_features, on=cns.ELEMENT_ID_COLUMNS)
        convert_data_and_save(data, cns.DATASET_PREDICTION_PATH, cns.OUTPUT_PREDICTIONS_PARTITIONS_COUNT)
    else:
        print('Attention: using existing dataset')

    data_partitions_pathes = glob.glob(f'{cns.DATASET_PREDICTION_PATH}/part-*')
    return MultipartitionTemporalClassificationWithFeaturesDataLoader(data_partitions_pathes, mode='prediction', partition_epoch=False)