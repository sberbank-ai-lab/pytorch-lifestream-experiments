import torch
import torch.nn as nn


class OrdinalLoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.f = nn.ReLU()

    def make_rank_target(self, idx, x, tresholds):
        t = tresholds.view(tresholds.size(0), x.size(0), -1, tresholds.size(-1))
        y = torch.zeros_like(x)
        idx0 = torch.arange(x.size(0)).view(-1, 1).repeat(1, x.size(1)).view(-1)
        idx1 = torch.arange(x.size(1)).repeat(x.size(0))
        for i in range(idx.size(-1)):
            idx2 = idx[:, :, i].view(-1)
            idxt = torch.full_like(idx0, i)
            y[[idx0, idx1, idx2]] = t[[idxt, idx0, idx1, idx2]]
        return y

    def get_masks(self, y, tresholds):
        m = tresholds - y.unsqueeze(0).repeat(y.size(0), 1, 1)
        m[(m <= 0)] = -1
        m[(m > 0)] = 1

        return m.detach()

    def forward(self, predictions, tresholds, target):
        # prepare tresholds
        tresholds = tresholds.view(-1, tresholds.size(-1)).transpose(0, 1).unsqueeze(-1).repeat(1, 1, predictions.size(-1))

        # make ranking target
        y = self.make_rank_target(target, predictions, tresholds)
        predictions = predictions.view(-1, predictions.size(-1))
        y = y.view(-1, n_items)

        # get masks
        m = self.get_masks(y, tresholds)

        # calculate loss
        arg = tresholds - predictions.unsqueeze(0).repeat(tresholds.size(0), 1, 1)
        arg = arg * m
        return self.f(-1 * arg).sum(0).mean()


# params
n_items = 5
window = 2

# input
x = torch.randn(1, 1, n_items)
x = nn.Softmax(dim=-1)(x)
idx = torch.randint(1, n_items, (1, 1, window))

# user/time specific tresholds
tresholds = torch.sort(nn.Softmax(dim=-1)(nn.Linear(n_items, window)(x)), dim=-1, descending=True)[0]


loss = OrdinalLoss()
#error = loss(x, tresholds, idx)
#print(f'ERROR = {error}')

