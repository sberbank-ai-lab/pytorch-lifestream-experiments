import torch
from pip._internal.utils import logging
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset


logger = logging.getLogger(__name__)


class GraphDataset(Dataset):
    def __init__(self, node_count):
        self.node_count = node_count

    def __len__(self):
        return self.node_count

    def __getitem__(self, item):
        return item


class GraphBatchCollector:
    def __init__(self, s_edges, tree_batching_level, max_node_count):
        self.edges = s_edges
        self.tree_batching_level = tree_batching_level
        self.max_node_count = max_node_count

    def __call__(self, batch):
        selected_nodes = set(batch)  # list[int]
        for _ in range(self.tree_batching_level):
            selected_nodes = self.achievable_nodes(selected_nodes)
        pos_ix = self.get_pos_indexes(selected_nodes)

        selected_nodes = torch.tensor(selected_nodes)
        return selected_nodes, pos_ix

    def achievable_nodes(self, selected):
        if len(selected) >= self.max_node_count:
            return selected

        nodes = set(selected)
        for i in selected:
            nodes.update(set(self.edges.get(i, [])))
            if len(nodes) >= self.max_node_count:
                break
        return list(nodes)[:self.max_node_count]

    def get_pos_indexes(self, selected):
        def gen():
            for i in s_selected:
                for j in self.edges.get(i, set()).intersection(s_selected):
                    if i < j:
                        yield i, j
                    # if i > j:
                    #     yield j, i

        s_selected = set(selected)
        a_ix = [(i, j) for i, j in gen()]
        if len(a_ix) == 0:
            return None
        a_ix = torch.tensor(a_ix)
        return a_ix[:, 0], a_ix[:, 1]


def create_train_dataloader(node_count, s_edges, batch_size, tree_batching_level, max_node_count, num_workers):
    dataset = GraphDataset(node_count)
    collate_fn = GraphBatchCollector(s_edges, tree_batching_level, max_node_count)

    logger.info(f'Created DataLoader with {num_workers} workers')
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )

