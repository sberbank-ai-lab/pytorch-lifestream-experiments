import logging
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from const import (
    DATA_PATH, TRX_FILE_NAME,
    COL_CLIENT_ID, COL_TERM_ID, COL_AMOUNT,
)
from data_loader import create_train_dataloader
from dltranz.util import get_conf
from loss import get_loss, get_distance
from split_dataset import load_source_data

logger = logging.getLogger(__name__)


COL_COUNT = 'TRX_COUNT'


def print_hist(name, hist, total_count):
    values, bins = hist
    data = pd.Series(data=(values / total_count).round(6),
                     index=[f'{a} - {b}' for a, b in zip(bins[:-1], bins[1:])])
    logger.info(f'{name}:\n{data}')


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


def get_optimiser(conf, model):
    def create(name, **params):
        if name == 'SGD':
            return torch.optim.SGD(model.parameters(), **params)
        else:
            raise AttributeError(f'Unknown optimizer: {name}')
    return create(**conf['optimizer'])


class NormLayer(torch.nn.Module):
    def forward(self, x):
        x = x - x.mean(dim=0, keepdim=True)
        x = x / (x.pow(2).sum(dim=1, keepdim=True) + 1e-9).pow(0.5)
        return x


def validate(conf, nn_embedding, s_edges):
    def gen():
        for i in s_edges.keys():
            for j in s_edges.get(i, set()):
                if i < j:
                    yield i, j
                if i > j:
                    yield j, i

    a_ix = [(i, j) for i, j in gen()]
    if len(a_ix) == 0:
        return None
    a_ix = torch.tensor(a_ix)

    node_count = nn_embedding.num_embeddings
    device = torch.device(conf['device'])
    valid_batch_size = conf['valid.batch_size']
    valid_neg_top_k = conf['valid.neg_top_k']
    f_distance = get_distance(conf)

    nn_embedding.eval()
    all_pos_distances = []
    all_neg_distances = []
    pos_hit_sum = 0.0
    pos_hit_cnt = 0.0

    with torch.no_grad():
        for i in tqdm(range(0, len(a_ix), valid_batch_size), leave=False):
            ix = a_ix[i:i + valid_batch_size].to(device)
            d = f_distance(nn_embedding(ix[:, 0]), nn_embedding(ix[:, 1]))
            all_pos_distances.append(d.cpu().numpy())

        ix = torch.arange(node_count).to(device)
        all_embeddings = nn_embedding(ix)
        _, H = all_embeddings.size()

        for i in tqdm(range(0, node_count, valid_batch_size), leave=False):
            selected_nodes = ix[i:i + valid_batch_size]
            B = len(selected_nodes)

            d = f_distance(
                all_embeddings[i:i + valid_batch_size].unsqueeze(1).repeat(1, node_count, 1).
                    view(B * node_count, H),
                all_embeddings.unsqueeze(0).repeat(B, 1, 1).view(B * node_count, H)
            ).view(B, node_count)
            d[selected_nodes - i, selected_nodes] = float('nan')  # remove self

            top_k_indices = torch.topk(d, k=valid_neg_top_k, dim=1, largest=False).indices
            z = torch.zeros(B, node_count)

            for node_i in selected_nodes:
                node_j = list(s_edges.get(node_i.item(), set()))
                d[node_i - i, node_j] = float('nan')
                z[node_i - i, node_j] = 1

            all_neg_distances.append(torch.topk(d, k=valid_neg_top_k, dim=1, largest=False).
                                     values.view(-1).cpu().numpy())

            pos_hit_sum += z[
                torch.arange(len(top_k_indices)).view(-1, 1).repeat(1, valid_neg_top_k).view(-1),
                top_k_indices.view(-1),
            ].sum().item()
            pos_hit_cnt += z.sum().item()

    all_pos_distances = np.concatenate(all_pos_distances)
    all_neg_distances = np.concatenate(all_neg_distances)
    logger.info(f'Validation: {len(all_pos_distances)} pos samples {len(all_neg_distances)} neg samples. '
                f'Pos hit: {pos_hit_sum / pos_hit_cnt:.5f}')

    bins = np.linspace(
        min(all_pos_distances.min(), all_neg_distances.min()),
        max(all_pos_distances.max(), all_neg_distances.max()),
        10,
    )
    values_pos, _ = np.histogram(all_pos_distances, bins=bins)
    values_neg, _ = np.histogram(all_neg_distances, bins=bins)

    data = pd.DataFrame(data={
        'pos': values_pos,
        'neg': values_neg,
        'pos_pp': np.round(values_pos / len(all_pos_distances), 2),
        'neg_pp': np.round(values_neg / len(all_neg_distances), 2),
    },
                        index=[f'{a:7.3f} - {b:7.3f}' for a, b in zip(bins[:-1], bins[1:])])
    logger.info(f'Distances:\n{data}')


def train_embeddings(conf, nn_embedding, s_edges):
    """

    :param conf:
    :param nn_embedding: simple torch embeddings
    :param s_edges: {node: [list of connected nodes]}, all ids are ints, compatible with `nn_embedding` indexes
    :return:
    """
    def update_stat(k, v, alpha=0.99):
        if k in stat:
            stat[k] = stat[k] * alpha + v * (1 - alpha)
        else:
            stat[k] = v

    node_count = nn_embedding.num_embeddings
    batch_size = conf['batch_size']
    max_node_count = conf['max_node_count']
    tree_batching_level = conf['tree_batching_level']
    device = torch.device(conf['device'])
    model_prefix = conf['model_prefix']
    num_workers = conf['num_workers']
    valid_epoch_step = conf['valid.epoch_step']

    f_loss = get_loss(conf)
    optim = get_optimiser(conf, nn_embedding)

    stat = {}

    nn_embedding.to(device)

    t_edges = torch.tensor([[k, j] for k, v in s_edges.items() for j in v])
    t_edges = t_edges.to(device)
    logger.info(f'edges shape: {t_edges.shape}')

    data_loader = create_train_dataloader(
        node_count, s_edges, batch_size, tree_batching_level, max_node_count, num_workers)

    validate(conf, nn_embedding, s_edges)
    for epoch in range(1, conf['epoch_n'] + 1):
        nn_embedding.train()
        with tqdm(total=(node_count + batch_size - 1) // batch_size, mininterval=1.0, miniters=10) as p:
            for selected_nodes, pos_ix in data_loader:
                selected_nodes = selected_nodes.to(device)
                selected_node_count = len(selected_nodes)
                update_stat('node_count', selected_node_count)

                if pos_ix is None:
                    p.update(1)
                    continue
                update_stat('pos_count', len(pos_ix[0]))
                pos_ix = tuple(t.to(device) for t in pos_ix)
                #
                neg_ix = f_loss.get_neg_indexes_all_below_margin(nn_embedding, selected_nodes, pos_ix)
                update_stat('neg_count', len(neg_ix[0]))
                #
                loss = f_loss(nn_embedding, pos_ix, neg_ix)
                update_stat('loss', loss.item())
                #
                optim.zero_grad()
                loss.backward()
                optim.step()
                #
                p.update(1)
                stat_str = ', '.join([f'{k}: {v:.3f}' for k, v in stat.items()])
                p.set_description(f'Epoch [{epoch:03}]: {stat_str}', refresh=False)

        if epoch % valid_epoch_step == 0:
            validate(conf, nn_embedding, s_edges)
        torch.save(nn_embedding, model_prefix + f'nn_embedding_{epoch:04}.p')


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

    if conf['print_stat']:
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
    min_trx_per_edge = conf['data_prepare.min_trx_per_edge']
    _ix_to_save = all_links[COL_COUNT].ge(min_trx_per_edge)
    logger.info(f'Drop {(~_ix_to_save).sum()} edges with `transaction count` < {min_trx_per_edge}')
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
    model_prefix = conf['model_prefix']
    torch.save(node_encoder, model_prefix + 'node_encoder.p')
    train_embeddings(conf, nn_embedding, edges)
    torch.save(nn_embedding, model_prefix + 'nn_embedding.p')
    logger.info('Train end')
