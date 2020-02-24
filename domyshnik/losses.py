import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveLoss(nn.Module):
    """
    Contrastive loss
    
    "Signature verification using a siamese time delay neural network", NIPS 1993
    https://papers.nips.cc/paper/769-signature-verification-using-a-siamese-time-delay-neural-network.pdf
    """

    def __init__(self, margin, pair_selector):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.pair_selector = pair_selector
        self.kpos = 1
        self.kneg = 1

    def forward(self, embeddings, target):
        
        positive_pairs, negative_pairs = self.pair_selector.get_pairs(embeddings, target)
        positive_loss = F.pairwise_distance(embeddings[positive_pairs[:, 0]], embeddings[positive_pairs[:, 1]]).pow(2)

        positive_loss = (F.relu(positive_loss - self.margin * self.kpos)).sum()
        
        negative_loss = F.relu(
            self.margin * self.kneg - F.pairwise_distance(embeddings[negative_pairs[:, 0]], embeddings[negative_pairs[:, 1]])
        ).pow(2).sum()
        
        return positive_loss, negative_loss

    def step(self, gamma_pos=1, gamma_neg=1):
        self.kpos *= gamma_pos
        self.kneg /=gamma_neg

    def get_margings(self):
        return f'pos {self.margin * self.kpos}, neg {self.margin *self.kneg}'


class LocalConstantLoss(nn.Module):

    def __init__(self, margin, pair_selector):
        super(LocalConstantLoss, self).__init__()
        self.margin = margin
        self.pair_selector = pair_selector
        self.k = 1.1

    def forward(self, embeddings, target):
        positive_pairs, negative_pairs = self.pair_selector.get_pairs(embeddings, target)
        positive_loss = F.pairwise_distance(embeddings[positive_pairs[:, 0]], embeddings[positive_pairs[:, 1]]).pow(2)

        #positive_loss = positive_loss.sum()
        positive_loss = (F.relu(positive_loss - self.margin)).sum()
        
        negative_loss = F.relu(
            self.margin * self.k - F.pairwise_distance(embeddings[negative_pairs[:, 0]], embeddings[negative_pairs[:, 1]])
        ).pow(2).sum()

        return positive_loss, negative_loss