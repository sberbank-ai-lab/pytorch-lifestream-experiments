import torch


def get_distance(conf):
    def pairwise_l2_distance(a, b):
        return ((a - b).pow(2).sum(dim=-1) + 1e-6).pow(0.5)

    def acosh(z, eps=1e-9):
        return torch.log(z + (torch.clamp(z - 1, 0, None) + eps).pow(0.5) * (z + 1).pow(0.5))

    def l2_to_poincare(d):
        # raise NotImplementedError('Not compatible with shape with more than 2 dimensions')
        # map input on hyperbolic space with N+1 dimensions
        # only `z` we need
        z = (d.pow(2).sum(dim=1) + 1).pow(0.5)
        # map points from hyperbola to poincare ball
        return d.div(z.view(-1, 1) + 1)

    def pairwise_poincare_distance(a, b):
        a = l2_to_poincare(a)
        b = l2_to_poincare(b)

        t = ((a - b) ** 2).sum(dim=1)
        t = t / ((1 - (a ** 2).sum(dim=1)) * (1 - (b ** 2).sum(dim=1)))
        t = 1 + 2 * t
        return acosh(t)

    distance = conf['distance']
    if distance == 'l2':
        return pairwise_l2_distance
    elif distance == 'poincare':
        return pairwise_poincare_distance
    else:
        raise AttributeError(f'Unknown distance: {distance}')


class ContrastiveLoss(torch.nn.Module):
    def __init__(self, f_distance, neg_margin, pos_margin, total_node_count):
        super().__init__()
        self.f_distance = f_distance
        self.neg_margin = neg_margin
        self.pos_margin = pos_margin

        self.rev_node_indexes = torch.zeros(total_node_count).long()

    def forward(self, embedding_model, pos_ix, neg_ix):
        pos_distance = self.f_distance(embedding_model(pos_ix[0]), embedding_model(pos_ix[1]))
        neg_distance = self.f_distance(embedding_model(neg_ix[0]), embedding_model(neg_ix[1]))
        pos_loss = torch.relu(pos_distance - self.pos_margin).pow(2)
        neg_loss = torch.relu(self.neg_margin - neg_distance).pow(2)
        loss = torch.cat([pos_loss, neg_loss])
        loss = loss.sum()
        return loss

    # @staticmethod
    # def get_pos_indexes_s(selected, edges):
    #     def gen():
    #         for i in s_selected:
    #             for j in edges.get(i, set()).intersection(s_selected):
    #                 if i < j:
    #                     yield i, j
    #                 if i > j:
    #                     yield j, i
    #
    #     # TODO: move it to GPU
    #     s_selected = set(selected.cpu().numpy().tolist())
    #     a_ix = [(i, j) for i, j in gen()]
    #     if len(a_ix) == 0:
    #         return None
    #     a_ix = torch.tensor(a_ix).to(selected.device)
    #     return a_ix[:, 0], a_ix[:, 1]
    #
    # @staticmethod
    # def get_pos_indexes_t(selected, edges):
    #     with torch.no_grad():
    #         n, _ = edges.size()
    #         ix = edges.view(n, 2, 1) == selected.view(1, 1, -1)
    #         ix = ix.any(dim=2).all(dim=1)
    #     return edges[ix, 0], edges[ix, 1]

    def get_neg_indexes_all_below_margin(self, embedding_model, selected, pos_ix):
        with torch.no_grad():
            selected_node_count = len(selected)
            self.rev_node_indexes[selected] = torch.arange(selected_node_count)

            all_embeddings = embedding_model(selected).detach()

            n, s = all_embeddings.size()
            a = all_embeddings.view(n, 1, s).repeat(1, n, 1).view(n * n, s)
            b = all_embeddings.view(1, n, s).repeat(n, 1, 1).view(n * n, s)
            d = self.f_distance(a, b).view(n, n)  # 1:30

            ix_0, ix_1 = pos_ix
            ix_0 = self.rev_node_indexes[ix_0]
            ix_1 = self.rev_node_indexes[ix_1]
            d[ix_0, ix_1] = float('nan')  # removing pos_ix

            neg_margin = self.neg_margin
            ix = (neg_margin - d) > 0

            ix_0, ix_1 = ix.nonzero(as_tuple=True)
            ix = ix_0 < ix_1
            ix_0 = ix_0[ix]  # triu
            ix_1 = ix_1[ix]

        return selected[ix_0], selected[ix_1]

    def get_neg_indexes_top_k(self, embedding_model, selected, pos_ix, k):
        with torch.no_grad():
            selected_node_count = len(selected)
            self.rev_node_indexes[selected] = torch.arange(selected_node_count)

            all_embeddings = embedding_model(selected).detach()

            n, s = all_embeddings.size()
            a = all_embeddings.view(n, 1, s).repeat(1, n, 1).view(n * n, s)
            b = all_embeddings.view(1, n, s).repeat(n, 1, 1).view(n * n, s)
            d = self.f_distance(a, b).view(n, n)

            ix_0, ix_1 = pos_ix
            ix_0 = self.rev_node_indexes[ix_0]
            ix_1 = self.rev_node_indexes[ix_1]

            d[ix_0, ix_1] = float('nan')  # removing pos_ix
            d[ix_1, ix_0] = float('nan')  # removing pos_ix
            ix_diag = torch.arange(n, device=d.device)
            d[ix_diag, ix_diag] = float('nan')  # removing pos_ix

            neg_margin = self.neg_margin
            d_neg = (neg_margin - d)
            values, ix = torch.topk(d_neg, k, dim=1)

            ix = ix[values > 0]

        if len(ix) == 0:
            return [], []
        return selected[ix[0]], selected[ix[1]]


def get_loss(conf):
    def create(name, **params):
        if name == 'ContrastiveLoss':
            return ContrastiveLoss(**params)
        else:
            raise AttributeError(f'Unknown loss: {name}')

    f_distance = get_distance(conf)
    return create(f_distance=f_distance, **conf['loss'])
