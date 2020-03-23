N_CEIDS, N_UIDS = 8297, 499664
BATCH_SIZE = 2#64
EMBEDDING_DIM = 16#1024
N_EPOCH = 10
B_TEST_DATA = False
MARGIN = 0.1
NUM_WORKERS = 0

TRANSFORMER_HEADS = 2
TRANSFORMER_LAYERS = 1

TRAIN_WINDOW = 4
TEST_WINDOW = 4
NEG_COEF = 3
MAKE_SPARCE_PREDICTIONS = False
SPARCNESS = 0.3

USE_CONSTANT_ORDINAL_TRESHOLDS = True
SAVE_WEIGHTS_NAME = 'model_weights'
SAVE_MODEL_EACH_EPOCH = False
RNN_HIDDEN_DIM = 256
MAX_SEQ_LEN = 100


def extend_add_info(add_info, k, v):
    if add_info is None:
        return {k: v}
    add_info[k] = v
    return add_info


ALL_FEATURES = {

    't_delta_1'          : {'type': 'reg', 'in': 1, 'out': 1},
    't_delta_2'          : {'type': 'reg', 'in': 1, 'out': 1},
    'rating'             : {'type': 'reg', 'in': 1, 'out': 1},
    'watched_rating'     : {'type': 'reg', 'in': 1, 'out': 1},
    'feature_1'          : {'type': 'reg', 'in': 1, 'out': 1},
    'feature_2'          : {'type': 'reg', 'in': 1, 'out': 1},
    'feature_4'          : {'type': 'reg', 'in': 1, 'out': 1},
    'feature_5'          : {'type': 'reg', 'in': 1, 'out': 1},

    'bookmark'           : {'type': 'cat', 'in': 3, 'out': 2},
    'device_type'        : {'type': 'cat', 'in': 8, 'out': 2},
    'device_manufacturer': {'type': 'cat', 'in': 101, 'out': 8},
    'purchase'           : {'type': 'cat', 'in': 3, 'out': 2},
    'rent'               : {'type': 'cat', 'in': 3, 'out': 2},
    'subscription'       : {'type': 'cat', 'in': 3, 'out': 2},
    'feature_3'          : {'type': 'cat', 'in': 52, 'out': 4},
}


FEATURES = ALL_FEATURES
FEATURES = {}


def get_features_embedding_size():
    import numpy as np
    return int(np.array([v['out'] for v in FEATURES.values()]).sum())


# -------------------------------------------------------------------------------------------------------
# recommendation model settings

AGGREGATE_COLUMN = 'user_uid'
USER_ID_COLUMN = 'user_uid'
ELEMENT_ID_COLUMNS = 'element_uid'
TIME_COLUMN = 'ts'
CATEGORICAL_COLUMNS = [
    'consumption_mode',
    'device_type',
    'device_manufacturer',
    ELEMENT_ID_COLUMNS,
    USER_ID_COLUMN
]

OUTPUT_PARTITIONS_COUNT = 3
OUTPUT_PREDICTIONS_PARTITIONS_COUNT = 3
DATASET_TRAIN_PATH = f'./dataset'
DATASET_PREDICTION_PATH = f'./dataset_prediction'
DEPLOY_MODE = False