N_AUGMENTS = 15#5
LEARNING_RATE = 0.0002#0.002
GAMMA = 0.9025
BATCH_SIZE = 32#128
EPOCHS = 20
SAMPLING_STRATEGY = 'HardNegativePair'
NEGATIVES_COUNT = 3#N_AUGMENTS + 1
MARGING = 0.5#0.1
STEP_SIZE = 5
WEIGHTS_PATH = '/mnt/data/molchanov/models'
MODEL_POSTFIX = ''
NUM_CLASSES = 10
ADD_INFO = None
BAD_REWARD = -0.8
DEVICE = 'cuda'
ERROR_RATE = 0.5

SAVE_MODELS = True
#CURRENT_PARAMS = 'cifar10_metric_learning_per_sampl'
#CURRENT_PARAMS = 'cifar10_classification_metric_learning_per_sampl'
#CURRENT_PARAMS = 'cifar10_domyshnik'
CURRENT_PARAMS = 'cifar10_metric_learning_global'
#CURRENT_PARAMS = 'cifar10_metric_learning_global_basis'

class config_params:

    def __init__(self, 
                 n_augments, 
                 lr, 
                 gamma,
                 batch_size,
                 epochs,
                 sampling_strategy,
                 negatives_cnt,
                 marging,
                 step_size,
                 model_postfix,
                 add_info=None,
                 device='cuda'):
        self.n_augments = n_augments
        self.lr = lr
        self.gamma = gamma
        self.batch_size = batch_size
        self.epochs = epochs
        self.sampling_strategy = sampling_strategy
        self.negatives_cnt = negatives_cnt
        self.marging = marging
        self.step_size = step_size
        self.model_postfix = model_postfix
        self.device = device
        self.add_info = add_info


PARAMS = {
    "metric_learning_per_class": config_params(n_augments=20,
                                               lr=0.002,
                                               gamma=0.9025,
                                               batch_size=512,
                                               epochs=15,
                                               sampling_strategy='HardNegativePair',
                                               negatives_cnt=10,
                                               marging=0.3,
                                               step_size=3,
                                               model_postfix='mnist_per_class'),

    "metric_learning_per_sampl": config_params(n_augments=15,
                                               lr=0.002,
                                               gamma=0.9025,
                                               batch_size=256,
                                               epochs=15,
                                               sampling_strategy='HardNegativePair',
                                               negatives_cnt=10,
                                               marging=0.5,
                                               step_size=3,
                                               model_postfix='mnist_per_sampl'),

    "classification_metric_learning_per_sampl": config_params(n_augments=1,
                                                             lr=0.002,
                                                             gamma=0.9025,
                                                             batch_size=128,
                                                             epochs=20,
                                                             sampling_strategy='HardNegativePair',
                                                             negatives_cnt=3,
                                                             marging=0.5,
                                                             step_size=1,
                                                             model_postfix='mnist_per_sampl'),

    "domyshnik": config_params(n_augments=20,
                               lr=0.0004,
                               gamma=0.9025,
                               batch_size=128,
                               epochs=90,
                               sampling_strategy='HardNegativePairKlDiv',
                               negatives_cnt=10,
                               marging=0.1,
                               step_size=5,
                               model_postfix='mnist',
                               add_info={'refresh_reward_step': 40,
                                         'k_pos' : 0.25, 
                                         'k_neg': 500, 
                                         'k_reward': 5}),

    "domyshnik_mnist_not_bad_params": config_params(n_augments=20,
                               lr=0.0004,
                               gamma=0.9025,
                               batch_size=128,
                               epochs=90,
                               sampling_strategy='HardNegativePairKlDiv',
                               negatives_cnt=10,
                               marging=0.1,
                               step_size=5,
                               model_postfix='mnist',
                               add_info={'refresh_reward_step': 40,
                                         'k_pos' : 0.25, 
                                         'k_neg': 500, 
                                         'k_reward': 5}),

    "cifar10_metric_learning_per_class": config_params(n_augments=5,
                                               lr=0.002,
                                               gamma=0.9025,
                                               batch_size=512,
                                               epochs=30,
                                               sampling_strategy='HardNegativePair',
                                               negatives_cnt=10,
                                               marging=0.3,
                                               step_size=5,
                                               model_postfix='cifar10_per_class'),

    "cifar10_metric_learning_per_sampl": config_params(n_augments=5,
                                               lr=0.002,
                                               gamma=0.9025,
                                               batch_size=800,
                                               epochs=250,
                                               sampling_strategy='HardNegativePair',
                                               negatives_cnt=5,
                                               marging=0.5,
                                               step_size=5,
                                               model_postfix='cifar10_per_sampl_hinton_150_centroids',
                                               device='cuda:2',
                                               add_info={'k_pos' : 1, 
                                                         'k_neg': 10,
                                                         'use_clusterisation_loss': True,
                                                         'k_clust_neg': 70,
                                                         'k_clust_pos': 10000#1
                                                         }
                                                         ),

    "cifar10_metric_learning_global": config_params(
        n_augments=5,
        lr=0.002,
        gamma=0.9025,
        batch_size=128,
        epochs=50,
        sampling_strategy='HardNegativePair',
        negatives_cnt=5,
        marging=0.5,
        step_size=5,
        model_postfix='cifar10_global_c500_caug5_iaug5_cmrg05_imrg02_v2',
        device='cuda:0',
        add_info={  'centroids_count': 500,
                    'losses': [
                        {'name': 'InClusterisationLoss'}, 
                        {'name': 'ContrastiveLossOriginal_centroids', 'marging': 0.5, 'neg_count': 5, 'sampling_strategy': 'HardNegativePair'},
                        {'name': 'ContrastiveLossOriginal_images', 'marging': 0.2, 'neg_count': 5, 'sampling_strategy': 'HardNegativePair'}
                        ],
                    'n_augs_imgs': 5
                    }
                    
        ),

    "cifar10_metric_learning_global_basis": config_params(
        n_augments=5,
        lr=0.002,
        gamma=0.9025,
        batch_size=128,
        epochs=50,
        sampling_strategy='HardNegativePair',
        negatives_cnt=5,
        marging=0.5,
        step_size=5,
        model_postfix='cifar10_global_c100_caug5_iaug5_cmrg05_imrg01_basis',
        device='cuda:3',
        add_info={  'centroids_count': 100,
                    'losses': [
                        {'name': 'BasisClusterisationLoss'}, 
                        {'name': 'ContrastiveLossOriginal_centroids', 'marging': 0.5, 'neg_count': 5, 'sampling_strategy': 'HardNegativePair'},
                        {'name': 'ContrastiveLossOriginal_images', 'marging': 0.1, 'neg_count': 5, 'sampling_strategy': 'HardNegativePair'}
                        ],
                    'n_augs_imgs': 5
                    }
                    
        ),

    "cifar10_classification_metric_learning_per_sampl": config_params(n_augments=1,
                                                                    lr=0.0004,#0.002,
                                                                    gamma=0.9025,
                                                                    batch_size=128,
                                                                    epochs=20,
                                                                    sampling_strategy='HardNegativePair',
                                                                    negatives_cnt=3,
                                                                    marging=0.5,
                                                                    step_size=1,
                                                                    model_postfix='cifar10_per_sampl',
                                                                    add_info={}),

    "cifar10_domyshnik": config_params(n_augments=9,
                               lr=0.0004,
                               gamma=0.9025,
                               batch_size=128,
                               epochs=150,
                               sampling_strategy='HardNegativePair',#'HardNegativePairKlDiv',
                               negatives_cnt=10,
                               marging=0.1,
                               step_size=5,
                               model_postfix='cifar10_domyshnik',
                               device='cuda:1',
                               add_info={'refresh_reward_step': 400,
                                         'k_pos' : 1,#1,#0.25*0.3, 
                                         'k_neg': 1,#1,#200*0.3, 
                                         'k_reward': 15,#15,
                                         'centroids_count': 150,
                                         'factor': 1})                                                                                                           
}

cparams = PARAMS[CURRENT_PARAMS]
N_AUGMENTS = cparams.n_augments
LEARNING_RATE = cparams.lr
GAMMA = cparams.gamma
BATCH_SIZE = cparams.batch_size
EPOCHS = cparams.epochs
SAMPLING_STRATEGY = cparams.sampling_strategy
NEGATIVES_COUNT = cparams.negatives_cnt
MARGING = cparams.marging
STEP_SIZE = cparams.step_size
MODEL_POSTFIX = cparams.model_postfix
ADD_INFO = cparams.add_info
DEVICE = cparams.device

print(f'model _params:\n\
        CURRENT_PARAMS {CURRENT_PARAMS}\n\
        N_AUGMENTS {N_AUGMENTS}\n\
        LEARNING_RATE {LEARNING_RATE}\n\
        GAMMA {GAMMA}\n\
        BATCH_SIZE {BATCH_SIZE}\n\
        EPOCHS {EPOCHS}\n\
        SAMPLING_STRATEGY {SAMPLING_STRATEGY}\n\
        NEGATIVES_COUNT {NEGATIVES_COUNT}\n\
        MARGING {MARGING}\n\
        STEP_SIZE {STEP_SIZE}\n\
        MODEL_POSTFIX {MODEL_POSTFIX}\n\
        ADD_INFO {ADD_INFO}\n\
        ERROR_RATE {ERROR_RATE}\n\
       ')
if not SAVE_MODELS:
    print('Attention: model will not be saved')