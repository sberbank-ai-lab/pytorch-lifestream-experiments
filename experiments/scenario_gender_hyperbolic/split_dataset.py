import os
from random import Random

import pandas as pd
import logging

from const import (
    COL_CLIENT_ID, COL_TERM_ID,
    TEST_SIZE, DATA_PATH, TARGET_FILE, SALT,
    OUTPUT_TEST_IDS_PATH,
)


logger = logging.getLogger(__name__)


def load_source_data(data_path, trx_files):
    data = []
    for file in trx_files:
        file_path = os.path.join(data_path, file)
        df = pd.read_csv(file_path)
        data.append(df)
        logger.info(f'Loaded {len(df)} rows from "{file_path}"')

    data = pd.concat(data, axis=0)
    logger.info(f'Loaded {len(data)} rows in total')

    return data


def split_dataset(test_size, data_path, target_files, salt):
    df_target = load_source_data(data_path, target_files)
    s_clients = set(df_target[COL_CLIENT_ID].astype(str).values.tolist())

    # shuffle client list
    s_clients = sorted(s_clients)
    s_clients = [cl_id for cl_id in s_clients]
    Random(salt).shuffle(s_clients)

    # split client list
    Nrows_test = int(len(s_clients) * test_size)
    s_clients_train = s_clients[:-Nrows_test]
    s_clients_test = s_clients[-Nrows_test:]

    return s_clients_train, s_clients_test


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(funcName)-20s   : %(message)s')

    s_clients_train, s_clients_test = split_dataset(TEST_SIZE, DATA_PATH, [TARGET_FILE], SALT)
    logger.info(f'Split: {len(s_clients_train)} - train, {len(s_clients_test)} - test')

    df_test_ids = pd.DataFrame({COL_CLIENT_ID: s_clients_test})
    df_test_ids.to_csv(OUTPUT_TEST_IDS_PATH, index=False)
    logger.info(f'Saved {len(df_test_ids)} clients in "{OUTPUT_TEST_IDS_PATH}"')
