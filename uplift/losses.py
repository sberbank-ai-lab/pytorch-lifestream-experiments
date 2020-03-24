import torch
import torch.nn as nn
import torch.nn.functional as F
from dltranz.metric_learn.sampling_strategies import HardNegativePairSelector
torch.autograd.set_detect_anomaly(True)
from dltranz.metric_learn.ml_models import L2Normalization

def outer_pairwise_distance(A, B=None):
    """
        Compute pairwise_distance of Tensors
            A (size(A) = n x d, where n - rows count, d - vector size) and
            B (size(A) = m x d, where m - rows count, d - vector size)
        return matrix C (size n x m), such as C_ij = distance(i-th row matrix A, j-th row matrix B)

        if only one Tensor was given, computer pairwise distance to itself (B = A)
    """

    if B is None: B = A

    max_size = 2 ** 26
    n = A.size(0)
    m = B.size(0)
    d = A.size(1)

    if n * m * d <= max_size or m == 1:

        return torch.pairwise_distance(
            A[:, None].expand(n, m, d).reshape((-1, d)),
            B.expand(n, m, d).reshape((-1, d))
        ).reshape((n, m))

    else:

        batch_size = max(1, max_size // (n * d))
        batch_results = []
        for i in range((m - 1) // batch_size + 1):
            id_left = i * batch_size
            id_rigth = min((i + 1) * batch_size, m)
            batch_results.append(outer_pairwise_distance(A, B[id_left:id_rigth]))

        return torch.cat(batch_results, dim=1)


def outer_cosine_similarity(A, B=None):
    """
        Compute cosine_similarity of Tensors
            A (size(A) = n x d, where n - rows count, d - vector size) and
            B (size(A) = m x d, where m - rows count, d - vector size)
        return matrix C (size n x m), such as C_ij = cosine_similarity(i-th row matrix A, j-th row matrix B)

        if only one Tensor was given, computer pairwise distance to itself (B = A)
    """

    if B is None: B = A

    max_size = 2 ** 32
    n = A.size(0)
    m = B.size(0)
    d = A.size(1)

    if n * m * d <= max_size or m == 1:

        A_norm = torch.div(A.transpose(0, 1), A.norm(dim=1)).transpose(0, 1)
        B_norm = torch.div(B.transpose(0, 1), B.norm(dim=1)).transpose(0, 1)
        return torch.mm(A_norm, B_norm.transpose(0, 1))

    else:

        batch_size = max(1, max_size // (n * d))
        batch_results = []
        for i in range((m - 1) // batch_size + 1):
            id_left = i * batch_size
            id_rigth = min((i + 1) * batch_size, m)
            batch_results.append(outer_cosine_similarity(A, B[id_left:id_rigth]))

        return torch.cat(batch_results, dim=1)


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

        tmp = positive_loss.sum() / positive_loss.size(0)
        #positive_loss = (F.relu(positive_loss - self.margin * self.kpos)).sum()
        positive_loss = positive_loss.sum()
        
        negative_loss = F.relu(
            self.margin * self.kneg - F.pairwise_distance(embeddings[negative_pairs[:, 0]], embeddings[negative_pairs[:, 1]])
        ).pow(2).sum()
        
        #print(f'average distance beetween augments {tmp.item()}')
        return positive_loss, negative_loss

    def step(self, gamma_pos=1, gamma_neg=1):
        self.kpos *= gamma_pos
        self.kneg /=gamma_neg

    def get_margings(self):
        return f'pos {self.margin * self.kpos}, neg {self.margin *self.kneg}'


class ContrastiveLossOriginal(nn.Module):
    """
    Contrastive loss
    
    "Signature verification using a siamese time delay neural network", NIPS 1993
    https://papers.nips.cc/paper/769-signature-verification-using-a-siamese-time-delay-neural-network.pdf
    """

    def __init__(self, margin, pair_selector):
        super(ContrastiveLossOriginal, self).__init__()
        self.margin = margin
        self.pair_selector = pair_selector

    def forward(self, embeddings, target):
        
        positive_pairs, negative_pairs = self.pair_selector.get_pairs(embeddings, target)
        positive_loss = F.pairwise_distance(embeddings[positive_pairs[:, 0]], embeddings[positive_pairs[:, 1]]).pow(2)

        #positive_loss = (F.relu(positive_loss - self.margin)).sum()
        positive_loss = positive_loss.sum()
        
        negative_loss = F.relu(
            self.margin - F.pairwise_distance(embeddings[negative_pairs[:, 0]], embeddings[negative_pairs[:, 1]])
        ).pow(2).sum()
        
        return positive_loss, negative_loss


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


class ClusterisationLoss(nn.Module):

    def __init__(self, margin, input_dim, num_classes, device):
        super(ClusterisationLoss, self).__init__()
        self.margin = margin
        self.num_classes = num_classes
        self.device = device

        kwargs = {
            'neg_count' : 1,
        }
        kwargs = {k:v for k,v in kwargs.items() if v is not None}
        self.pair_selector = HardNegativePairSelector(**kwargs)

        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, embeddings):

        # predict classes
        embeddings = self.fc(embeddings) 
        n = embeddings.size(0)
        m = embeddings.size(-1)
        # very important because argmax are not differentiable
        # so we have to calculate distance not in metric learning space
        probs = F.softmax(embeddings, dim=-1)
        lbls = torch.argmax(probs.detach(), dim=-1)
        
        # prepare classes mask
        mask = (lbls.expand(self.num_classes, n) == torch.arange(self.num_classes).expand(n, self.num_classes).transpose(0, 1).to(self.device)).float().to(self.device)
        weights = mask.sum(-1)
        idxs = (weights != 0).expand(self.num_classes, m).transpose(0,1)

        weights = weights.expand(m, self.num_classes).transpose(0,1)
        weights = weights.masked_fill(weights == 0, 1) # not infly results

        # everage embedings in class
        mean_embeddings = torch.matmul(mask, embeddings)
        mean_embeddings = torch.div(mean_embeddings, weights)

        
        # remove zero embeddings (threir classes aren't in batch)
        full_mean_embeddings = mean_embeddings
        mean_embeddings = mean_embeddings.masked_select(idxs).view(-1, m)

        # calculate negative loss (between cluster centers)
        if mean_embeddings.size(0) != 1:
            _, negative_pairs = self.pair_selector.get_pairs(mean_embeddings, 
                                                            torch.arange(mean_embeddings.size(0)).to(self.device))
            
            distances = F.pairwise_distance(mean_embeddings[negative_pairs[:, 0]], 
                                            mean_embeddings[negative_pairs[:, 1]]).pow(2)

            # check only smallest distance
            negative_loss = F.relu(
                self.margin - distances.min()
            ).pow(2).sum()
        else:
            negative_loss = torch.Tensor([0.0]).to(self.device)

        # calculate positive loss (cluster variance)

        expects = torch.index_select(full_mean_embeddings, 0, lbls)
        embeddings2 = embeddings - expects #center
        
        idxr = torch.arange(n).repeat(n).to(self.device)
        idxl = torch.arange(n).unsqueeze(-1).repeat(1, n).view(-1).to(self.device)
        Dl = torch.index_select(embeddings2, 0, idxl)
        Dr = torch.index_select(embeddings2, 0, idxr)
        
        D = F.pairwise_distance(Dl, Dr).view(n, n)
        D1 = torch.matmul(torch.matmul(mask, D), mask.transpose(0, 1)).diagonal()
        
        weights2 = mask.sum(-1) - 1 # unbiased estimation
        weights3 = weights2.masked_fill(weights2 <= 0, 1) # not infly results
        D1 = torch.div(D1, weights3)
        
        positive_loss = D1.sum()/D1.size(0)
        
        return positive_loss, negative_loss


class ClusterisationLoss2(nn.Module):

    def __init__(self, device):
        super(ClusterisationLoss2, self).__init__()
        self.norm = L2Normalization()
        self.device = device

    # embeddings: Nxd, centroids: Kxd
    def forward(self, embeddings, centroids):
        N, K = embeddings.size(0), self.centroids.size(0)

        centers = centroids
                
        # distances to each centroid (cosine)
        D = torch.matmul(centers, embeddings.transpose(0, 1)) # KxN

        # mask
        centr_idx = torch.argmin(D, dim=0).detach()
        centr_mask = (centr_idx.repeat(K).view(K, -1) == torch.arange(K).view(-1, 1).repeat(1, N).to(self.device)).int() # KxN
        weights = centr_mask.sum(-1) + 1

        Dmasked = D * centr_mask

        # in cluster distance
        in_cluster_dist = -1 * torch.div(Dmasked.sum(-1), weights).sum()/K # -1 - because cosine distance

        # between cluster dist
        Dc = torch.matmul(centers, centers.transpose(0, 1)) # KxK
        between_cluster_dist = Dc.sum()/(K*K) # max - because cosine distance

        return in_cluster_dist, between_cluster_dist


class InClusterisationLoss(nn.Module):

    def __init__(self, device):
        super(InClusterisationLoss, self).__init__()
        self.device = device

    # embeddings: Nxd, centroids: Kxd
    def forward(self, embeddings, centroids):
        N, K = embeddings.size(0), centroids.size(0)

        centers = centroids
                
        # distances to each centroid
        #D = torch.matmul(centers, embeddings.transpose(0, 1)) # KxN, (cosine)
        D = outer_pairwise_distance(centers, embeddings) # KxN, (l2)

        # mask
        centr_idx = torch.argmin(D, dim=0).detach()
        centr_mask = (centr_idx.repeat(K).view(K, -1) == torch.arange(K).view(-1, 1).repeat(1, N).to(self.device)).int() # KxN
        weights = centr_mask.sum(-1) + 1

        Dmasked = D * centr_mask

        # in cluster distance
        in_cluster_dist =  torch.div(Dmasked.sum(-1), weights).sum()/K # *-1 for cosine distance

        # centroid's sphere radius estimation and elements count
        centroids_radius = Dmasked.max(-1)[0].mean()
        centroids_elements_count = centr_mask.sum(-1).float().mean()

        return in_cluster_dist, centroids_radius, centroids_elements_count


class BasisClusterisationLoss(nn.Module):

    def __init__(self, device):
        super(BasisClusterisationLoss, self).__init__()
        self.device = device
        self.norm = L2Normalization()

    # embeddings: Nxd, centroids: Kxd
    def forward(self, embeddings, centroids):
        # distances to each centroid
        coefs = outer_cosine_similarity(embeddings, centroids) # NxK

        # entropy
        H = -1 * F.softmax(coefs, dim=-1) * F.log_softmax(coefs, dim=-1)
        H = H.sum()

        return H, coefs