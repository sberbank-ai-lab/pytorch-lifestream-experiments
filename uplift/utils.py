import torch
import torch.nn.functional as F
from uplift.models import * 
from uplift.data import *
from dltranz.metric_learn.sampling_strategies import HardNegativePairSelector, PairSelector
from uplift.constants import *
from uplift.losses import *


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
            B.expand(n, m, d).reshape((-1, d)), # i think that this was a misatake because B is not a valid distribution
            #F.softmax(B.expand(n, m, d).reshape((-1, d)), -1),
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

class RandomNegativePairSelector(PairSelector):
    """
    Generates all possible possitive pairs given labels and
         neg_count hardest negative example for each example
    """
    def __init__(self, neg_count = 1):
        super(RandomNegativePairSelector, self).__init__()
        self.neg_count = neg_count

    def get_pairs(self, embeddings, labels):
        
        # construct matrix x, such as x_ij == 0 <==> labels[i] == labels[j]
        n = labels.size(0)
        x = labels.expand(n,n) - labels.expand(n,n).t()        
            
        # positive pairs
        positive_pairs = torch.triu((x == 0).int(), diagonal = 1).nonzero()
        
        # hard negative minning
        mat_distances = outer_pairwise_distance(embeddings.detach()) # pairwise_distance
        
        upper_bound = int((2*n) ** 0.5) + 1
        mat_distances = ((upper_bound - mat_distances) * (x != 0).type(mat_distances.dtype)) # filter: get only negative pairs
        
        indices = (torch.randperm(mat_distances.size(0)*mat_distances.size(1))%mat_distances.size(1)).view(-1, mat_distances.size(1))[:self.neg_count, :]
        indices.to(mat_distances.device)
        
        negative_pairs = torch.stack([
            torch.arange(0,n, dtype = indices.dtype, device = indices.device).repeat(self.neg_count),
            torch.cat(indices.unbind(dim = 0))
        ]).t()

        return positive_pairs, negative_pairs

# ----------------------------------------------------------------------------------------------

def get_sampling_strategy(params='HardNegativePair', neg_count=None):

    if params == 'HardNegativePair':
        kwargs = {
            'neg_count' : NEGATIVES_COUNT if neg_count is None else neg_count,
        }
        kwargs = {k:v for k,v in kwargs.items() if v is not None}
        #sampling_strategy = HardNegativePairSelector(**kwargs)
        sampling_strategy = RandomNegativePairSelector(**kwargs)
        return sampling_strategy
    elif params == 'HardNegativePairKlDiv':
        kwargs = {
            'neg_count' : NEGATIVES_COUNT if neg_count is None else neg_count,
        }
        kwargs = {k:v for k,v in kwargs.items() if v is not None}
        #sampling_strategy = HardNegativeKLDivPairSelector(**kwargs)
        sampling_strategy = RandomNegativePairSelector(**kwargs)
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

        self.device = torch.device(device)
        
        '''if device == 'cuda':
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')'''
            
        self.model.to(device)
        self.loss.to(device)
        
# -------------------------- predefind configs --------------------------------

def get_launch_info():
    # setup losses
    if 'losses' in ADD_INFO:
        losses = []
        for loss in ADD_INFO['losses']:
            if 'ContrastiveLossOriginal' in loss['name']:
                l = ContrastiveLossOriginal(margin=loss['marging'], pair_selector=get_sampling_strategy(loss['sampling_strategy'], loss['neg_count']))
                losses.append((l, loss['name']))

            if 'InClusterisationLoss' in loss['name']:
                losses.append((InClusterisationLoss(torch.device(DEVICE)), loss['name']))

            if 'BasisClusterisationLoss' in loss['name']:
                losses.append((BasisClusterisationLoss(torch.device(DEVICE)), loss['name']))

        ADD_INFO['losses'] = losses

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
                                                    device=DEVICE,
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
                                                                    device=DEVICE,
                                                                    mode='classification',
                                                                    model_name='mnist_classification_metriclearning.w',
                                                                    add_info=ADD_INFO)
        return mnist_classification_metriclearning_lunch_info  

    elif CURRENT_PARAMS == 'cifar10_classification_metric_learning_per_sampl': 
        mnist_classification_metriclearning_lunch_info = LaunchInfo(model=Cifar10ClassificationMetricLearningModelGlobal(), 
                                                                    loss=torch.nn.NLLLoss(), 
                                                                    optimizer=None, 
                                                                    scheduler=None, 
                                                                    train_loader=get_cifar10_train_loader(batch_size=BATCH_SIZE, 
                                                                                                          n_augments=N_AUGMENTS), 
                                                                    test_loader=get_cifar10_test_loader(batch_size=BATCH_SIZE, 
                                                                                                        n_augments=0),  
                                                                    epochs=EPOCHS, 
                                                                    device=DEVICE,
                                                                    mode='classification',
                                                                    model_name='cifar10_classification_metriclearning.w',
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
                                                    device=DEVICE,
                                                    mode='metric_learning',
                                                    model_name='mnist_metric_learning.w',
                                                    add_info=ADD_INFO)
        return mnist_metriclearning_lunch_info

    elif CURRENT_PARAMS in ['cifar10_metric_learning_per_sampl', 'cifar10_metric_learning_per_class']:                                         
        cifar10_metriclearning_lunch_info = LaunchInfo(model=Cifar10MetricLearningNet3(), 
                                                    loss=ContrastiveLoss(margin=MARGING, 
                                                                        pair_selector=get_sampling_strategy(SAMPLING_STRATEGY)), 
                                                    optimizer=None, 
                                                    scheduler=None, 
                                                    train_loader=get_cifar10_train_loader(batch_size=BATCH_SIZE, 
                                                                                        n_augments=N_AUGMENTS), 
                                                    test_loader=get_cifar10_test_loader(batch_size=BATCH_SIZE, 
                                                                                    n_augments=N_AUGMENTS), 
                                                    epochs=EPOCHS, 
                                                    device=DEVICE,
                                                    mode='metric_learning',
                                                    model_name='cifar10_metric_learning.w',
                                                    add_info=ADD_INFO)
        return cifar10_metriclearning_lunch_info

    elif CURRENT_PARAMS in ['cifar10_metric_learning_global', 'cifar10_metric_learning_global_basis']: 
        #loader = get_cifar10_train_loader(1000, n_augments=-2, augment_labels=False)
        # centroids will not be in train dataset
        loader = get_cifar10_test_loader(1000, n_augments=-2, augment_labels=False)
        centroids = get_cifar10_centroids(ADD_INFO['centroids_count']*2, loader)
        centroids = centroids[:ADD_INFO['centroids_count']]
        print(f'centroids count {centroids.size(0)}')

        cifar10_metriclearning_lunch_info = LaunchInfo(model=Cifar10MetricLearningNetCentroids(), 
                                                    loss=ContrastiveLossOriginal(margin=MARGING, 
                                                                                 pair_selector=get_sampling_strategy(SAMPLING_STRATEGY)), 
                                                    optimizer=None, 
                                                    scheduler=None, 
                                                    train_loader=get_cifar10_train_global_loader(batch_size=BATCH_SIZE, 
                                                                                                 centroids=centroids, 
                                                                                                 n_augments_centroids=N_AUGMENTS,
                                                                                                 n_augments_images=ADD_INFO['n_augs_imgs']),
                                                    test_loader=get_cifar10_test_global_loader(batch_size=BATCH_SIZE, 
                                                                                               centroids=centroids),
                                                    epochs=EPOCHS, 
                                                    device=DEVICE,
                                                    mode='metric_learning_global',
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
                                                    device=DEVICE,
                                                    mode='domyshnik',
                                                    model_name='mnist_domushnik.w',
                                                    add_info=ADD_INFO)

        return mnist_domyshnik_lunch_info   

    elif CURRENT_PARAMS == 'cifar10_domyshnik':
        cifar10_domyshnik_lunch_info = LaunchInfo(model=Cifar10DomyshnikNetNetCentroids(),#Cifar10DomyshnikNetNet(), 
                                                    loss=ContrastiveLoss(margin=MARGING, 
                                                                         pair_selector=get_sampling_strategy(SAMPLING_STRATEGY)), 
                                                    optimizer=None, 
                                                    scheduler=None, 
                                                    train_loader=get_cifar10_train_loader(batch_size=BATCH_SIZE, 
                                                                                        n_augments=N_AUGMENTS,
                                                                                        augment_labels=True), 
                                                    test_loader=get_cifar10_test_loader(batch_size=BATCH_SIZE, 
                                                                                    n_augments=N_AUGMENTS,
                                                                                    augment_labels=True), 
                                                    epochs=EPOCHS, 
                                                    device=DEVICE,
                                                    mode='domyshnik',
                                                    model_name='cifar10_domushnik.w',
                                                    add_info=ADD_INFO)
        return cifar10_domyshnik_lunch_info

    elif CURRENT_PARAMS == 'okko_metric_learning':
        okko_train, okko_test = get_okko_metrlearn_loaders(BATCH_SIZE, N_AUGMENTS)
        okko_metric_learnin_lunch_info = LaunchInfo(model=okko_metrlearn_model(), 
                                                    loss=ContrastiveLoss(margin=MARGING, 
                                                                         pair_selector=get_sampling_strategy(SAMPLING_STRATEGY)), 
                                                    optimizer=None, 
                                                    scheduler=None, 
                                                    train_loader=okko_train, 
                                                    test_loader=okko_test, 
                                                    epochs=EPOCHS, 
                                                    device=DEVICE,
                                                    mode='metric_learning',
                                                    model_name='okko.w',
                                                    add_info=ADD_INFO)
        return okko_metric_learnin_lunch_info

    elif CURRENT_PARAMS == 'okko_domyshik':
        okko_train, okko_test = get_okko_domyshnik_loaders(BATCH_SIZE, N_AUGMENTS)
        okko_metric_learnin_lunch_info = LaunchInfo(model=okko_domyshnik_model(), 
                                                    loss=ContrastiveLoss(margin=MARGING, 
                                                                         pair_selector=get_sampling_strategy(SAMPLING_STRATEGY)), 
                                                    optimizer=None, 
                                                    scheduler=None, 
                                                    train_loader=okko_train, 
                                                    test_loader=okko_test, 
                                                    epochs=EPOCHS, 
                                                    device=DEVICE,
                                                    mode='metric_learning',
                                                    model_name='okko.w',
                                                    add_info=ADD_INFO)
        return okko_metric_learnin_lunch_info
                                             