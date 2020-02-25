import torch
import torch.nn.functional as F
from domyshnik.models import * 
from domyshnik.data import *
from dltranz.metric_learn.sampling_strategies import HardNegativePairSelector, PairSelector
from domyshnik.constants import *
from domyshnik.losses import *


def outer_pairwise_kl_distance(A, B=None):
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

        return F.kl_div(
            F.log_softmax(A[:, None].expand(n, m, d).reshape((-1, d)), dim=-1),
            B.expand(n, m, d).reshape((-1, d)),
            reduction='none'
        ).sum(-1).reshape((n, m))

    else:

        batch_size = max(1, max_size // (n * d))
        batch_results = []
        for i in range((m - 1) // batch_size + 1):
            id_left = i * batch_size
            id_rigth = min((i + 1) * batch_size, m)
            batch_results.append(outer_pairwise_kl_distance(A, B[id_left:id_rigth]))

        return torch.cat(batch_results, dim=1)

class HardNegativeKLDivPairSelector(PairSelector):
    """
    Generates all possible possitive pairs given labels and
         neg_count hardest negative example for each example
    """
    def __init__(self, neg_count = 1):
        super(HardNegativeKLDivPairSelector, self).__init__()
        self.neg_count = neg_count

    def get_pairs(self, embeddings, labels):
        
        # construct matrix x, such as x_ij == 0 <==> labels[i] == labels[j]
        n = labels.size(0)
        x = labels.expand(n,n) - labels.expand(n,n).t()        
            
        # positive pairs
        positive_pairs = torch.triu((x == 0).int(), diagonal = 1).nonzero()
        
        # hard negative minning
        mat_distances = outer_pairwise_kl_distance(embeddings.detach()) # pairwise_distance
        
        upper_bound = int((2*n) ** 0.5) + 1
        mat_distances = ((upper_bound - mat_distances) * (x != 0).type(mat_distances.dtype)) # filter: get only negative pairs
        
        values, indices = mat_distances.topk(k = self.neg_count, dim = 0, largest = True)
        negative_pairs = torch.stack([
            torch.arange(0,n, dtype = indices.dtype, device = indices.device).repeat(self.neg_count),
            torch.cat(indices.unbind(dim = 0))
        ]).t()

        return positive_pairs, negative_pairs

# ----------------------------------------------------------------------------------------------

def get_sampling_strategy(params='HardNegativePair'):
    if params == 'HardNegativePair':
        kwargs = {
            'neg_count' : NEGATIVES_COUNT,
        }
        kwargs = {k:v for k,v in kwargs.items() if v is not None}
        sampling_strategy = HardNegativePairSelector(**kwargs)
        return sampling_strategy
    elif params == 'HardNegativePairKlDiv':
        kwargs = {
            'neg_count' : NEGATIVES_COUNT,
        }
        kwargs = {k:v for k,v in kwargs.items() if v is not None}
        sampling_strategy = HardNegativeKLDivPairSelector(**kwargs)
        return sampling_strategy
    return None

class LaunchInfo:
    
    def __init__(self, model, loss, optimizer, scheduler, train_loader, test_loader, epochs, device, mode, model_name, add_info=None):
        self.model = model
        self.loss = loss
        self.epochs = epochs
        self.mode = mode
        self.model_name = model_name
        self.add_info = add_info
        
        if optimizer is not None:
            self.optimizer = optimizer
        else:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
            
        if scheduler is not None:
            self.scheduler = scheduler
        else:
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                             step_size=STEP_SIZE,
                                                             gamma=GAMMA)
            
        self.train_loader = train_loader
        self.test_loader = test_loader
        
        if device == 'cuda':
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
            
        self.model.to(device)
        self.loss.to(device)
        
# -------------------------- predefind configs --------------------------------

def get_launch_info():
    if CURRENT_PARAMS == 'mnist_classification':
        mnist_classification_lunch_info = LaunchInfo(model=MnistClassificationNet(), 
                                                    loss=torch.nn.NLLLoss(), 
                                                    optimizer=None, 
                                                    scheduler=None, 
                                                    train_loader=get_mnist_train_loader(batch_size=BATCH_SIZE, 
                                                                                        n_augments=N_AUGMENTS), 
                                                    test_loader=get_mnist_test_loader(batch_size=BATCH_SIZE, 
                                                                                    n_augments=N_AUGMENTS), 
                                                    epochs=EPOCHS, 
                                                    device='cuda',
                                                    mode='classification',
                                                    model_name='mnist_classification.w',
                                                    add_info=ADD_INFO)
        return mnist_classification_lunch_info
    
    elif CURRENT_PARAMS == 'classification_metric_learning_per_sampl': 
        mnist_classification_metriclearning_lunch_info = LaunchInfo(model=MnistClassificationMetricLearningModel(), 
                                                                    loss=torch.nn.NLLLoss(), 
                                                                    optimizer=None, 
                                                                    scheduler=None, 
                                                                    train_loader=get_mnist_train_loader(batch_size=BATCH_SIZE, 
                                                                                                        n_augments=N_AUGMENTS), 
                                                                    test_loader=get_mnist_test_loader(batch_size=BATCH_SIZE, 
                                                                                                    n_augments=0), 
                                                                    epochs=EPOCHS, 
                                                                    device='cuda',
                                                                    mode='classification',
                                                                    model_name='mnist_classification_metriclearning.w',
                                                                    add_info=ADD_INFO)
        return mnist_classification_metriclearning_lunch_info   

    elif CURRENT_PARAMS in ['metric_learning_per_sampl', 'metric_learning_per_class']:                                         
        mnist_metriclearning_lunch_info = LaunchInfo(model=MnistMetricLearningNet3(), 
                                                    loss=ContrastiveLoss(margin=MARGING, 
                                                                         pair_selector=get_sampling_strategy(SAMPLING_STRATEGY)), 
                                                    optimizer=None, 
                                                    scheduler=None, 
                                                    train_loader=get_mnist_train_loader(batch_size=BATCH_SIZE, 
                                                                                        n_augments=N_AUGMENTS), 
                                                    test_loader=get_mnist_test_loader(batch_size=BATCH_SIZE, 
                                                                                    n_augments=N_AUGMENTS), 
                                                    epochs=EPOCHS, 
                                                    device='cuda',
                                                    mode='metric_learning',
                                                    model_name='mnist_metric_learning.w',
                                                    add_info=ADD_INFO)
        return mnist_metriclearning_lunch_info

    elif CURRENT_PARAMS in ['cifar10_metric_learning_per_sampl', 'cifar10_metric_learning_per_class']:                                         
        cifar10_metriclearning_lunch_info = LaunchInfo(model=Cifar10MetricLearningNet(), 
                                                    loss=ContrastiveLoss(margin=MARGING, 
                                                                        pair_selector=get_sampling_strategy(SAMPLING_STRATEGY)), 
                                                    optimizer=None, 
                                                    scheduler=None, 
                                                    train_loader=get_mnist_train_loader(batch_size=BATCH_SIZE, 
                                                                                        n_augments=N_AUGMENTS), 
                                                    test_loader=get_mnist_test_loader(batch_size=BATCH_SIZE, 
                                                                                    n_augments=N_AUGMENTS), 
                                                    epochs=EPOCHS, 
                                                    device='cuda',
                                                    mode='metric_learning',
                                                    model_name='cifar10_metric_learning.w',
                                                    add_info=ADD_INFO)
        return cifar10_metriclearning_lunch_info

    elif CURRENT_PARAMS == 'domyshnik':
        mnist_domyshnik_lunch_info = LaunchInfo(model=MnistDomyshnikNetNet3(), 
                                                    loss=ContrastiveLoss(margin=MARGING, 
                                                                         pair_selector=get_sampling_strategy(SAMPLING_STRATEGY)), 
                                                    optimizer=None, 
                                                    scheduler=None, 
                                                    train_loader=get_mnist_train_loader(batch_size=BATCH_SIZE, 
                                                                                        n_augments=N_AUGMENTS,
                                                                                        augment_labels=True), 
                                                    test_loader=get_mnist_test_loader(batch_size=BATCH_SIZE, 
                                                                                    n_augments=N_AUGMENTS,
                                                                                    augment_labels=True), 
                                                    epochs=EPOCHS, 
                                                    device='cuda',
                                                    mode='domyshnik',
                                                    model_name='mnist_domushnik.w',
                                                    add_info=ADD_INFO)

        return mnist_domyshnik_lunch_info   

    elif CURRENT_PARAMS == 'cifar10_domyshnik':
        cifar10_domyshnik_lunch_info = LaunchInfo(model=MnistDomyshnikNetNet3(), 
                                                    loss=ContrastiveLoss(margin=MARGING, 
                                                                         pair_selector=get_sampling_strategy(SAMPLING_STRATEGY)), 
                                                    optimizer=None, 
                                                    scheduler=None, 
                                                    train_loader=get_mnist_train_loader(batch_size=BATCH_SIZE, 
                                                                                        n_augments=N_AUGMENTS,
                                                                                        augment_labels=True), 
                                                    test_loader=get_mnist_test_loader(batch_size=BATCH_SIZE, 
                                                                                    n_augments=N_AUGMENTS,
                                                                                    augment_labels=True), 
                                                    epochs=EPOCHS, 
                                                    device='cuda',
                                                    mode='domyshnik',
                                                    model_name='cifar10_domushnik.w',
                                                    add_info=ADD_INFO)
        return cifar10_domyshnik_lunch_info

                                             