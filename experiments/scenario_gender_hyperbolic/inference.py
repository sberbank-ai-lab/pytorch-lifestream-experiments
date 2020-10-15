import logging

import torch
import os
import pandas as pd

from dltranz.util import get_conf

from const import COL_CLIENT_ID

from train_graph_embeddings import NodeEncoder, NNEmbeddigs

logger = logging.getLogger(__name__)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(funcName)-20s   : %(message)s')
    conf = get_conf()

    if os.path.isfile(conf['output']):
        logger.info(f'File "{conf["output"]}" already exists')
        if not conf.get('overwrite', False):
            exit(0)

    if not os.path.isfile(conf['nn_embeddings']):
        logger.error(f'File "{conf["nn_embeddings"]}" is not exists')
        exit(-1)

    node_incoder = torch.load(conf['node_encoder'], map_location='cpu')
    nn_embeddings = torch.load(conf['nn_embeddings'], map_location='cpu')

    df_target = pd.read_csv(conf['target_path'])
    output = pd.DataFrame({
        COL_CLIENT_ID: df_target[COL_CLIENT_ID],
    })
    indices = node_incoder.encode_client(df_target[COL_CLIENT_ID])
    with torch.no_grad():
        nn_embeddings.eval()
        vectors = nn_embeddings(torch.from_numpy(indices.values)).detach().numpy()
    for i in range(vectors.shape[1]):
        output[f'vec_{i:04d}'] = vectors[:, i]
    output.to_csv(conf['output'], index=False)
    logger.info(f'Dump {output.shape} to "{conf["output"]}"')
