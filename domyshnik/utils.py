import torch
from domyshnik.models import * 
from domyshnik.data import *
from dltranz.metric_learn.losses import ContrastiveLoss
from dltranz.metric_learn.sampling_strategies import HardNegativePairSelector
from domyshnik.constants import *

def get_sampling_strategy(params='HardNegativePair'):
    if params == 'HardNegativePair':
        kwargs = {
            'neg_count' : NEGATIVES_COUNT,
        }
        kwargs = {k:v for k,v in kwargs.items() if v is not None}
        sampling_strategy = HardNegativePairSelector(**kwargs)
        return sampling_strategy
    return None

class LaunchInfo:
    
    def __init__(self, model, loss, optimizer, scheduler, train_loader, test_loader, epochs, device, mode, model_name):
        self.model = model
        self.loss = loss
        self.epochs = epochs
        self.mode = mode
        self.model_name = model_name
        
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
                                             model_name='mnist_classification.w')
    
mnist_metriclearning_lunch_info = LaunchInfo(model=MnistMetricLearningNet(), 
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
                                             model_name='mnist_metric_learning.w')