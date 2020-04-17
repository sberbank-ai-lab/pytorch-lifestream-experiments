import torch
import torch.nn.functional as F
from uplift.models import * 
from uplift.data import *
from dltranz.metric_learn.sampling_strategies import HardNegativePairSelector, PairSelector
from uplift.constants import *
from uplift.losses import *
import os
import datetime
import os.path as ops
from torch.utils.tensorboard import SummaryWriter

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

            if 'ContrastiveLoss' == loss['name']:
                l = ContrastiveLoss(margin=loss['marging'], pair_selector=get_sampling_strategy(loss['sampling_strategy'], loss['neg_count']))
                losses.append((l, loss['name']))

            if 'PositiveContrastiveLoss' in loss['name']:
                l = PositiveContrastiveLoss(margin=loss['marging'], pair_selector=get_sampling_strategy(loss['sampling_strategy'], loss['neg_count']))
                losses.append((l, loss['name']))

            if 'InClusterisationLoss' in loss['name']:
                losses.append((InClusterisationLoss(torch.device(DEVICE)), loss['name']))

            if 'BasisClusterisationLoss' in loss['name']:
                losses.append((BasisClusterisationLoss(torch.device(DEVICE)), loss['name']))

            if 'KLDivLoss' in loss['name']:
                losses.append((nn.KLDivLoss(), loss['name']))

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
        return cifar10_domyshnik_lunch_info,

    elif CURRENT_PARAMS == 'cifar10_vae_domyshnik':
        cifar10_vae_domyshnik_lunch_info = LaunchInfo(model=Cifar10VAEDomyshnikNet(),
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
                                                    model_name='cifar10_vae_domushnik.w',
                                                    add_info=ADD_INFO)
        return cifar10_vae_domyshnik_lunch_info

    elif CURRENT_PARAMS == 'cifar10_metric_learning':     
                                      
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

        # set loger info
        train = get_cifar10_train_loader(batch_size=800, n_augments=5)
        test = get_cifar10_train_loader(batch_size=800, n_augments=5)
        (train_imgs, train_lbls) = next(iter(train))
        (test_imgs, test_lbls) = next(iter(test))
        device = torch.device(DEVICE)
        set_loger(Logger(
                           (train_imgs.to(device), train_lbls.to(device)), 
                           (test_imgs.to(device), test_lbls.to(device))
                         )
                  )

        return cifar10_metriclearning_lunch_info

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

    elif CURRENT_PARAMS == 'okko_classification':
        okko_train, okko_test = get_okko_classification_loaders(BATCH_SIZE)
        okko_metric_learnin_lunch_info = LaunchInfo(model=okko_classification_model(), 
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

    elif CURRENT_PARAMS == 'criteo_metric_learning':
        criteo_train, criteo_test = get_criteo_data_loaders(BATCH_SIZE, N_AUGMENTS)
        okko_metric_learnin_lunch_info = LaunchInfo(model=criteo_metrlearn_model(), 
                                                    loss=ContrastiveLoss(margin=MARGING, 
                                                                         pair_selector=get_sampling_strategy(SAMPLING_STRATEGY)), 
                                                    optimizer=None, 
                                                    scheduler=None, 
                                                    train_loader=criteo_train, 
                                                    test_loader=criteo_test, 
                                                    epochs=EPOCHS, 
                                                    device=DEVICE,
                                                    mode='metric_learning',
                                                    model_name=f"criteo_{ADD_INFO['augment_type']}.w",
                                                    add_info=ADD_INFO)
        return okko_metric_learnin_lunch_info

    elif CURRENT_PARAMS == 'criteo_domyshnik':
        criteo_train, criteo_test = get_criteo_domyshnik_data_loaders(BATCH_SIZE, N_AUGMENTS, True)
        okko_metric_learnin_lunch_info = LaunchInfo(model=criteo_domyshnik_model(), 
                                                    loss=ContrastiveLoss(margin=MARGING, 
                                                                         pair_selector=get_sampling_strategy(SAMPLING_STRATEGY)), 
                                                    optimizer=None, 
                                                    scheduler=None, 
                                                    train_loader=criteo_train, 
                                                    test_loader=criteo_test, 
                                                    epochs=EPOCHS, 
                                                    device=DEVICE,
                                                    mode='domyshnik',
                                                    model_name=f"criteo_{ADD_INFO['augment_type']}.w",
                                                    add_info=ADD_INFO)
        return okko_metric_learnin_lunch_info

class Logger:
    
    def __init__(self, train_sample, test_sample):
        self.class_names = ['airplane',
                            'automobile',
                            'bird',
                            'cat',
                            'deer',
                            'dog',
                            'frog',
                            'horse',
                            'ship',
                            'truck']
        dt = datetime.datetime.now()
        base_dir = '/mnt/data/molchanov/logs'
        prefix = f'{CURRENT_PARAMS}_{dt.year}_{dt.month}_{dt.day}_{dt.hour}_{dt.minute}_{dt.second}'
        log_dir = osp.join(base_dir, prefix)
        os.mkdir(log_dir)
        self.writer = SummaryWriter(log_dir=log_dir, comment="cifar10")
        self.train_sample = train_sample
        self.test_sample = test_sample
        self.prefix = ''
        self.step = 0
        self.log_step = 0
        self.writer.add_text(tag='COMMENT', text_string=COMMENT, global_step=0)
        self.log_index = 0

    def _get_centroids(self, embeds, lbls):
        l, e = lbls, embeds
        num_lbls = l.max().int().item() + 1

        # get mask
        m = l.expand(num_lbls, l.size(0))
        k = torch.arange(num_lbls).expand(l.size(0), num_lbls).transpose(0, 1).to(m.device)
        m = (m == k).int()

        # get weights
        w = m.sum(-1).expand(e.size(-1), num_lbls).transpose(0, 1)

        # get centroids
        avrg = torch.matmul(m.float(), e.float())
        centroids = torch.div(avrg, w)
        return centroids


    def _log_metrics(self, prefix, metrics, step):
        self.writer.add_scalars(f'{prefix}_metrics_info', metrics, step)

    def _log_centroids_distance(self, centroids, step):
        D_centroids = outer_pairwise_distance(centroids)
        for i in range(D_centroids.size(0)):
            d = {}
            if i != self.log_index:
                continue
            for j in range(D_centroids.size(0)):
                if i == j:
                    continue
                ci, cj = self.class_names[i].upper(), self.class_names[j].upper()
                d[f'{ci}_{cj}'] = D_centroids[i, j]
            self.writer.add_scalars(f'{self.prefix}_class{ci}_centroids_distance', d, step)

    def _log_incluster_average_distance(self, embs, imgs, lbls, step):
        # mask 0
        ll = lbls.expand(lbls.size(0), lbls.size(0))
        m0 = (ll == ll.transpose(0, 1)).int()
        
        # mask 1
        num_lbls = lbls.max().int().item() + 1
        m = lbls.expand(num_lbls, lbls.size(0))
        k = torch.arange(num_lbls).expand(lbls.size(0), num_lbls).transpose(0, 1).to(lbls.device)
        m1 = (m == k).int()
        
        # all distances
        D = outer_pairwise_distance(embs)
        D2 = D.pow(2)
        
        # mask distances
        D = D * m0
        D2 = D2 * m0
        
        # final dists
        weights = m1.sum(-1).pow(2)
        D = torch.matmul(torch.matmul(m1.float(), D), m1.float().transpose(0, 1)).diag()
        D = torch.div(D, weights)
        D2 = torch.matmul(torch.matmul(m1.float(), D2), m1.float().transpose(0, 1)).diag()
        D2 = torch.div(D2, weights)
        
        log_D = {f'class_{self.class_names[i].upper()}': val.item() for i, val in enumerate(D)}
        self.writer.add_scalars(f'{self.prefix}_inner_classes_distance', log_D, step)

        # dispertion
        varD = (D2 - D.pow(2)).pow(0.5)
        for i, (delta, d) in enumerate(zip(varD, D)):
            if i != self.log_index:
                continue
            t = {"low": (d - delta).item(),
                 "up":  (d + delta).item(),
                 "avrg": d.item()
                }
            self.writer.add_scalars(f'{self.prefix}_class_{self.class_names[i].upper()}_spread', t, step)

        # image tracks
        D = outer_pairwise_distance(embs)
        num_log_class = weights[self.log_index].pow(0.5).int()
        start_log_class = 0
        for i in range(self.log_index):
            start_log_class += weights[i].pow(0.5).int()

        for other_class in range(num_lbls):
            num_other_class = weights[other_class].pow(0.5).int()
            start_other_class = 0
            for i in range(other_class):
                start_other_class += weights[i].pow(0.5).int()
            # all distances among elements in log_class vs other_class
            D_elems = D[start_log_class: start_log_class + num_log_class, start_other_class: start_other_class + num_other_class]
            
            # tracked images
            log_idxs, other_idxs = [1], [i for i in range(0, num_other_class, int(num_other_class/10))]
            if step == 1:
                for log_i in log_idxs:
                    log2other_img = [imgs[start_log_class + log_i]]
                    for other_j in other_idxs:
                        log2other_img.append(imgs[start_other_class + other_j])

                    log_img_batch = torch.stack(log2other_img, dim=0)
                    self.writer.add_images(tag=f'{self.class_names[self.log_index]}_{self.class_names[other_class]}', 
                                           img_tensor=log_img_batch, 
                                           global_step=step)
            # track distances
            for log_i in log_idxs:
                log_dist = {}
                for other_j in other_idxs:
                    log_dist[f'{self.class_names[self.log_index]}{log_i}_{self.class_names[other_class]}{other_j}'] = D_elems[log_i, other_j]

                self.writer.add_scalars(f'{self.prefix}_{self.class_names[self.log_index]}_{self.class_names[other_class]}_distance', log_dist, step)

    def _log_augmented_statistics(self, imgs, embs, lbls, step):
        # imgs size: B, n_augs, C, H, W
        # lbls - sorted
        n_augs = imgs.size(1)

        # get positions of log_index items
        num_lbls = lbls.max().int().item() + 1
        m = lbls.expand(num_lbls, lbls.size(0))
        k = torch.arange(num_lbls).expand(lbls.size(0), num_lbls).transpose(0, 1).to(lbls.device)
        m1 = (m == k).int().sum(-1)
        start_pos = 0
        for i in range(self.log_index):
            start_pos += m1[i]
        num_log_idx_elems = m1[self.log_index]
        
        # select
        elems = imgs[start_pos : start_pos + num_log_idx_elems]
        elems_embs = embs[start_pos : start_pos + num_log_idx_elems]
        elems_embs = elems_embs.view(-1, elems_embs.size(-1))
        D_elems = outer_pairwise_distance(elems_embs)

        elems_lbls = torch.arange(elems.size(0)).expand(n_augs, elems.size(0)).transpose(0, 1).contiguous().view(-1).to(lbls.device)
        ll = elems_lbls.expand(elems_lbls.size(0), elems_lbls.size(0))
        m0 = (ll == ll.transpose(0, 1)).int()

        D_elems *= m0
        D_elems2 = D_elems.pow(2)

        D_elems /= (n_augs * n_augs)
        D_elems2 /= (n_augs * n_augs)

        stds = []
        for i in range(0, D_elems.size(-1), n_augs):
            D = D_elems[i: i + n_augs, i: i + n_augs].sum()
            D2 = D_elems2[i: i + n_augs, i: i + n_augs].sum()         
            val = (D2 - D.pow(2)).pow(0.5)
            stds.append(val.item())

        stdMean, stdStd = np.array(stds).mean(), np.array(stds).std() 
        log_std = {
            'mean': stdMean,
            'low' : stdMean - stdStd,
            'up'  : stdMean + stdStd
        }
        self.writer.add_scalars(f'{self.prefix}_{self.class_names[self.log_index]}_augment_distance_spread', log_std, step)
 
    def _forward(self, imgs, lbls, model):
        lbls, indices = lbls.sort() # лейблы только оригинальных картинок (для аугментированных копий - нет)
        n_augs = imgs.size(1)

        with torch.no_grad():
            embs = model(imgs)
        embs = embs.view(-1, n_augs, embs.size(-1))

        # sort by lbls
        imgs = torch.index_select(input=imgs, index=indices, dim=0)
        embs = torch.index_select(input=embs, index=indices, dim=0)

        # get original images/embeds
        idx = torch.arange(start=0, step=n_augs, end=lbls.size(0) * n_augs).to(imgs.device)
        _imgs = imgs.view(lbls.size(0)* n_augs, imgs.size(-3), imgs.size(-2), imgs.size(-1))
        orig_imgs = torch.index_select(input=_imgs, index=idx, dim=0)

        _embs = embs.view(lbls.size(0)* n_augs, embs.size(-1))
        orig_embs = torch.index_select(input=_embs, index=idx, dim=0)

        return orig_imgs, orig_embs, imgs, embs, lbls

    def _log(self, imgs, lbls, model, step):
        orig_imgs, orig_embs, imgs, embs, lbls = self._forward(imgs, lbls, model)

        # get class centers
        centroids = self._get_centroids(orig_embs, lbls)

        # log classes centroids distances
        self._log_centroids_distance(centroids, step)

        # log inclass average distances
        self._log_incluster_average_distance(orig_embs, orig_imgs, lbls, step)

        # log augmented statistics
        self._log_augmented_statistics(imgs, embs, lbls, step)

    def log(self, model, metrics, metrics_prefix):
        b_log = False
        if self.step < 300:
            if self.step % 3 == 0:
                b_log = True
        else:
            if self.step % LOGGER_STEP == 0:
                b_log = True

        if b_log:
            #self.prefix = 'TRAIN'
            #self._log(self.train_sample[0], self.train_sample[1], model, self.log_step)

            self.prefix = 'TEST'
            self._log(self.test_sample[0], self.test_sample[1], model, self.log_step)

            self._log_metrics(metrics_prefix, metrics, self.log_step)

            self.log_step += 1

        self.step += 1



                                             