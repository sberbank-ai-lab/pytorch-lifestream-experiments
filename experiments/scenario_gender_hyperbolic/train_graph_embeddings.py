import logging
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from const import (
    DATA_PATH, TRX_FILE_NAME,
    COL_CLIENT_ID, COL_TERM_ID, COL_AMOUNT,
)
from dltranz.util import get_conf
from split_dataset import load_source_data

logger = logging.getLogger(__name__)


COL_COUNT = 'TRX_COUNT'

def find_nodes(df_trx, col_id, small_rate=0.03):
    df = abs(df_trx.groupby(col_id)[COL_AMOUNT].sum()).sort_values(ascending=False)
    return df.index.tolist()[:-int(len(df) * small_rate)]


class NodeEncoder:
    def __init__(self, client_ids, term_ids):
        self._client_to_node = {v: 1 + i for i, v in enumerate(client_ids)}
        start_pos = len(client_ids) + 1
        self._term_to_node = {v: start_pos + 1 + i for i, v in enumerate(term_ids)}

        self._missing_client = 0
        self._missing_term = start_pos

    def encode_client(self, client_id):
        if type(client_id) is pd.Series:
            return client_id.map(self._client_to_node).fillna(self._missing_client).astype(int)
        return self._client_to_node.get(client_id, self._missing_client)

    def encode_term(self, term_id):
        if type(term_id) is pd.Series:
            return term_id.map(self._term_to_node).fillna(self._missing_term).astype(int)
        return self._term_to_node.get(term_id, self._missing_term)

    @property
    def node_count(self):
        return len(self._client_to_node) + len(self._term_to_node) + 2


def prepare_links(all_links):
    client_term = all_links.groupby(COL_CLIENT_ID)[COL_TERM_ID].agg(set).to_dict()
    term_client = all_links.groupby(COL_TERM_ID)[COL_CLIENT_ID].agg(set).to_dict()
    return {**client_term, **term_client}


def achievable_nodes(selected, edges, max_node_count):
    nodes = set(selected.numpy().tolist())
    for i in selected.numpy().tolist():
        nodes.update(set(edges.get(i, [])))
        if len(nodes) >= max_node_count:
            break
    return torch.tensor(list(nodes)[:max_node_count])


def a_indexes(selected, edges):
    def gen():
        for i in s_selected:
            for j in edges.get(i, set()).intersection(s_selected):
                yield i, j

    s_selected = set(selected.numpy().tolist())
    a_ix = [(i, j) for i, j in gen()]
    if len(a_ix) == 0:
        return [], []
    a_ix = torch.tensor(a_ix)
    return a_ix[:, 0], a_ix[:, 1]


def get_distance(conf):
    def l2_distance(d):
        n, s = d.size()
        return ((d.view(n, 1, s) - d.view(1, n, s)).pow(2).sum(dim=2) + 1e-6).pow(0.5)

    def acosh(z, eps=1e-9):
        return torch.log(z + (torch.clamp(z - 1, 0, None) + eps).pow(0.5) * (z + 1).pow(0.5))

    def pairwize_poincare_distance(a, b):
        t = ((a - b) ** 2).sum(dim=1)
        t = t / ((1 - (a ** 2).sum(dim=1)) * (1 - (b ** 2).sum(dim=1)))
        t = 1 + 2 * t
        return acosh(t)

    def poincare_distance(d):
        # map input on hyperbolic space with N+1 dimensions
        # only `z` we need
        z = (d.pow(2).sum(dim=1) + 1).pow(0.5)
        # map points from hyperbola to poincare ball
        d = d.div(z.view(-1, 1) + 1)

        n, s = d.size()
        a = d.view(n, 1, s).repeat(1, n, 1).view(n * n, s)
        b = d.view(1, n, s).repeat(n, 1, 1).view(n * n, s)
        return pairwize_poincare_distance(a, b).view(n, n)

    distance = conf['distance']
    if distance == 'l2':
        return l2_distance
    elif distance == 'poincare':
        return poincare_distance
    else:
        raise AttributeError(f'Unknown distance: {distance}')


class ContrastiveLoss(torch.nn.Module):
    def __init__(self, neg_margin, pos_margin):
        super().__init__()
        self.neg_margin = neg_margin
        self.pos_margin = pos_margin

    def forward(self, d, a):
        not_a = 1 - a - torch.diag(torch.ones(a.size()[0], device=d.device))
        pos_loss = torch.relu(d[a.bool()] - self.pos_margin).pow(2)
        neg_loss = torch.relu(self.neg_margin - d[not_a.bool()]).pow(2)
        loss = torch.cat([pos_loss, neg_loss])
        loss = loss.sum()
        return loss


def get_loss(conf):
    def create(name, **params):
        if name == 'ContrastiveLoss':
            return ContrastiveLoss(**params)
        else:
            raise AttributeError(f'Unknown loss: {name}')

    return create(**conf['loss'])


def get_optimiser(conf, model):
    def create(name, **params):
        if name == 'SGD':
            return torch.optim.SGD(model.parameters(), **params)
        else:
            raise AttributeError(f'Unknown optimizer: {name}')
    return create(**conf['optimizer'])


def train_embeddings(conf, nn_embedding, edges):
    """

    :param conf:
    :param nn_embedding: simple torch embeddings
    :param edges: {node: [list of connected nodes]}, all ids are ints, compatible with `nn_embedding` indexes
    :return:
    """
    def update_stat(k, v, alpha=0.999):
        if k in stat:
            stat[k] = stat[k] * alpha + v * (1 - alpha)
        else:
            stat[k] = v

    node_count = nn_embedding.num_embeddings
    batch_size = conf['batch_size']
    max_node_count = conf['max_node_count']
    tree_batching_level = conf['tree_batching_level']
    device = torch.device(conf['device'])

    f_distance = get_distance(conf)
    f_loss = get_loss(conf)
    optim = get_optimiser(conf, nn_embedding)

    stat = {}

    nn_embedding.to(device)
    rev_node_indexes = torch.zeros(node_count).long()

    nn_embedding.train()

    for epoch in range(1, conf['epoch_n'] + 1):
        node_indexes = torch.multinomial(torch.ones(node_count), node_count, replacement=False).long()

        with tqdm(total=(node_count + batch_size - 1) // batch_size, mininterval=1.0, miniters=10) as p:
            for i in range(0, node_count, batch_size):
                # 6170 it\s
                selected_nodes = node_indexes[i:i + batch_size]
                for _ in range(tree_batching_level):
                    selected_nodes = achievable_nodes(selected_nodes, edges, max_node_count)
                #
                # 1700 it\s
                selected_node_count = len(selected_nodes)
                update_stat('node_count', selected_node_count)
                #
                embeddings = nn_embedding(selected_nodes.to(device))
                #
                # 1400 it\s
                d = f_distance(embeddings)
                #
                # 1100 it\s
                rev_node_indexes[selected_nodes] = torch.arange(selected_node_count)
                # 1000 it\s
                a = torch.zeros(selected_node_count, selected_node_count)
                # 300 it\s
                ix_0, ix_1 = a_indexes(selected_nodes, edges)
                update_stat('pos_count', len(ix_0))
                # 130 it\s
                ix_0 = rev_node_indexes[ix_0]
                ix_1 = rev_node_indexes[ix_1]
                a[ix_0, ix_1] = 1
                a[ix_1, ix_0] = 1
                a = a.to(device)
                #
                # 100 it\s
                loss = f_loss(d, a)
                update_stat('loss', loss.item())
                #
                # 100 it\s
                optim.zero_grad()
                loss.backward()
                optim.step()
                #
                # 100 it\s
                p.update(1)
                stat_str = ', '.join([f'{k}: {v:.3f}' for k, v in stat.items()])
                p.set_description(f'Epoch [{epoch:03}]: {stat_str}')

        torch.save(nn_embedding, f'models/nn_embedding_{epoch:03}.p')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(funcName)-20s   : %(message)s')
    conf = get_conf()

    all_trans = load_source_data(DATA_PATH, [TRX_FILE_NAME])
    all_trans = all_trans[[COL_CLIENT_ID, COL_TERM_ID, COL_AMOUNT]]
    all_trans = all_trans[~all_trans.isna().any(axis=1)]
    logger.info(f'all_trans final shape: {all_trans.shape}')

    all_links = pd.DataFrame({
        COL_CLIENT_ID: all_trans[COL_CLIENT_ID],
        COL_TERM_ID: all_trans[COL_TERM_ID],
        COL_COUNT: 1,
        COL_AMOUNT: all_trans[COL_AMOUNT],
    })
    all_links = all_links.groupby([COL_CLIENT_ID, COL_TERM_ID]).sum().reset_index()
    logger.info(f'all_links shape: {all_links.shape}')

    def print_hist(name, hist, total_count):
        values, bins = hist
        data = pd.Series(data=(values / total_count).round(6),
                         index=[f'{a} - {b}' for a, b in zip(bins[:-1], bins[1:])])
        logger.info(f'{name}:\n{data}')

    _a = all_links[COL_CLIENT_ID].value_counts().values
    _stat = np.histogram(_a, bins=(2**np.arange(0, np.ceil(np.log2(_a.max())) + 1)).astype(int))
    print_hist('Links per client', _stat, len(_a))

    _a = all_links[COL_TERM_ID].value_counts().values
    _stat = np.histogram(_a, bins=(2 ** np.arange(0, np.ceil(np.log2(_a.max())) + 1)).astype(int))
    print_hist('Links per term', _stat, len(_a))

    _a = all_links[COL_COUNT].values
    _stat = np.histogram(_a, bins=(2 ** np.arange(0, np.ceil(np.log2(_a.max())) + 1)).astype(int))
    print_hist('Transactions count per edge', _stat, len(_a))

    _a = np.abs(all_links[COL_AMOUNT].values)
    _stat = np.histogram(_a, bins=[0] + (2 ** np.arange(0, np.ceil(np.log2(_a.max())) + 1)).astype(int).tolist())
    print_hist('Amount sum per edge', _stat, len(_a))

    # drop
    _ix_to_save = all_links[COL_COUNT].gt(1)
    logger.info(f'Drop {(~_ix_to_save).sum()} edges with `transaction count` <= 1')
    all_links = all_links[_ix_to_save]
    #
    _more_than_one_client_per_term = all_links[COL_TERM_ID].value_counts()[lambda x: x > 1]
    _ix_to_save = all_links[COL_TERM_ID].isin(_more_than_one_client_per_term.index)
    logger.info(f'Drop {(~_ix_to_save).sum()} edges with `edge count per terminal` <= 1')
    all_links = all_links[_ix_to_save]
    logger.info(f'all_links final shape: {all_links.shape}')
    logger.info(f'{all_links[COL_CLIENT_ID].nunique()} clients lost, {all_links[COL_TERM_ID].nunique()} terms lost')

    nodes_clients = find_nodes(all_links, COL_CLIENT_ID)
    nodes_terms = find_nodes(all_links, COL_TERM_ID)
    logger.info(f'Found {len(nodes_clients)} client nodes and {len(nodes_terms)} term nodes')
    node_encoder = NodeEncoder(nodes_clients, nodes_terms)

    all_links[COL_CLIENT_ID] = node_encoder.encode_client(all_links[COL_CLIENT_ID])
    all_links[COL_TERM_ID] = node_encoder.encode_term(all_links[COL_TERM_ID])

    nn_embedding = torch.nn.Embedding(node_encoder.node_count, conf['embedding_dim'])
    edges = prepare_links(all_links[[COL_CLIENT_ID, COL_TERM_ID]])

    logger.info('Train start')
    torch.save(node_encoder, 'models/node_encoder.p')
    train_embeddings(conf, nn_embedding, edges)
    torch.save(nn_embedding, 'models/nn_embedding.p')
    logger.info('Train end')
