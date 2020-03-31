import torch
import torch.nn as nn
import torch.nn.functional as F
import constants as cns


class BCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.BCELoss()

    def forward(self, pred, true):
        return self.loss(pred.float(), true.float())


class PairwiseMarginRankingLoss(nn.Module):
    def __init__(self, margin=0.0, size_average=None, reduce=None, reduction='mean'):
        """
        Pairwise Margin Ranking Loss. All setted parameters redirected to nn.MarginRankingLoss.
        All the difference is that pairs automatically generated for margin ranking loss.
        All possible pairs of different class are generated.
        """
        super().__init__()
        self.margin_loss = nn.MarginRankingLoss(margin, size_average, reduce, reduction)

    def forward(self, prediction, label, mask):
        """
        Get pairwise margin ranking loss.
        :param prediction: tensor of shape Bx1 of predicted probabilities
        :param label: tensor of shape Bx1 of true labels for pair generation
        """

        # positive-negative selectors
        mask_1 = label*mask == 1
        mask_0 = label * mask + 1 - mask == 0

        # selected predictions
        pred_0 = torch.masked_select(prediction, mask_0)
        pred_1 = torch.masked_select(prediction, mask_1)
        pred_1_n = pred_1.size()[0]
        pred_0_n = pred_0.size()[0]

        if pred_1_n > 0 and pred_0_n:
            # create pairs
            pred_00 = pred_0.unsqueeze(0).repeat(1, pred_1_n)
            pred_11 = pred_1.unsqueeze(1).repeat(1, pred_0_n).view(pred_00.size())
            out01 = -1 * torch.ones(pred_1_n*pred_0_n).to(prediction.device)

            return self.margin_loss(pred_00.view(-1), pred_11.view(-1), out01)
        else:
            return torch.sum(prediction) * 0.0


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss

    "Signature verification using a siamese time delay neural network", NIPS 1993
    https://papers.nips.cc/paper/769-signature-verification-using-a-siamese-time-delay-neural-network.pdf
    """

    def __init__(self, margin):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def get_pairs(self, distns, b_pos=True):
        mask1 = (torch.triu(torch.ones(distns.size(0), distns.size(0))) == 0).transpose(0, 1).cuda()
        mask2 = (distns > 0) if b_pos else (distns == 0)
        mask = (mask1 * mask2)
        return mask.nonzero(), torch.masked_select(distns, mask)

    def forward(self, embeddings, distances):
        positive_pairs, pos_distances = self.get_pairs(distances, True)
        negative_pairs, _ = self.get_pairs(distances, False)

        #positive_loss = (F.cosine_similarity(embeddings[positive_pairs[:, 0]], embeddings[positive_pairs[:, 1]]) - pos_distances).pow(2)
        #negative_loss = F.relu(self.margin - F.cosine_similarity(embeddings[negative_pairs[:, 0]], embeddings[negative_pairs[:, 1]])).pow(2)

        positive_loss = (F.pairwise_distance(embeddings[positive_pairs[:, 0]], embeddings[positive_pairs[:, 1]]) - pos_distances).pow(2)
        negative_loss = F.relu(self.margin - F.pairwise_distance(embeddings[negative_pairs[:, 0]], embeddings[negative_pairs[:, 1]])).pow(2)
        loss = torch.cat([positive_loss, negative_loss], dim=0)

        return loss.mean()


class KlLoss(nn.Module):
    def __init__(self, symetric=False):
        super().__init__()
        self.loss = nn.KLDivLoss(reduction='none')
        self.eps = 1e-12
        self.symetric = symetric

    def forward(self, pred, true, mask):
        mask = mask.unsqueeze(-1).repeat(1, 1, true.size(-1)).bool()
        pred_ = torch.masked_select(pred, mask).unsqueeze(0)
        true_ = torch.masked_select(true, mask).unsqueeze(0)
        return self.loss(pred_, true_).sum() / (mask.size(0)*mask.size(1))


class OrdinalLoss_(nn.Module):

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
        m = tresholds - y.unsqueeze(0).repeat(tresholds.size(0), 1, 1)
        m[(m <= 0)] = -1
        m[(m > 0)] = 1

        return m.detach()

    def forward(self, predictions, tresholds, target, mask, target_mask=None):
        # prepare tresholds
        tresholds = tresholds.view(-1, tresholds.size(-1)).transpose(0, 1).unsqueeze(-1).repeat(1, 1, predictions.size(-1))

        # make ranking target
        y = self.make_rank_target(target, predictions, tresholds)
        predictions = predictions.view(-1, predictions.size(-1))
        y = y.view(-1, cns.N_CEIDS)
        if target_mask is not None:
            target_mask = target_mask.view(-1, cns.N_CEIDS)

        # get masks
        m = self.get_masks(y, tresholds)

        # calculate loss
        deltas = tresholds - predictions.unsqueeze(0).repeat(tresholds.size(0), 1, 1)
        deltas = deltas * m
        if target_mask is not None:
            deltas = deltas * target_mask.unsqueeze(0).repeat(tresholds.size(0), 1, 1)
        deltas = self.f(-1 * deltas).sum(0).sum(-1) * mask.view(-1)
        return deltas.mean()


class OrdinalLoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.f = nn.ReLU()

    def forward(self, predictions, tresholds, target, mask):

        n_t, b, l = tresholds.size(-1), predictions.size(0), predictions.size(1)
        predictions = predictions.view(-1, predictions.size(-1))
        target = target.view(-1, target.size(-1))

        t0 = torch.arange(target.size(0)).unsqueeze(1)
        t1 = target
        target_mask = torch.zeros_like(predictions)
        target_mask[t0, t1] = tresholds

        phi = tresholds.unsqueeze(-1).unsqueeze(-1).repeat(1, target_mask.size(0), target_mask.size(1))

        t = target_mask.unsqueeze(0).repeat(n_t, 1, 1)

        m = t - phi
        m[m >= 0] = 1
        m[m < 0] = -1
        m = m.detach()

        p = predictions.unsqueeze(0).repeat(n_t, 1, 1)

        res = (p - phi) * m * (-1)
        res = self.f(res).sum(0)
        res = res.view(b, l, -1)

        error = (res.sum(-1) * mask).sum(-1).mean()/n_t
        return error