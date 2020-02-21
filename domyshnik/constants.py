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

SAVE_MODELS = True
CURRENT_PARAMS = 'metric_learning_per_class'

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
                 model_postfix):
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

PARAMS = {
    "metric_learning_per_class": config_params(n_augments=5,
                                               lr=0.002,
                                               gamma=0.9025,
                                               batch_size=128,
                                               epochs=20,
                                               sampling_strategy='HardNegativePair',
                                               negatives_cnt=6,
                                               marging=0.5,
                                               step_size=5,
                                               model_postfix='per_class'),

    "metric_learning_per_sampl": config_params(n_augments=15,
                                               lr=0.002,
                                               gamma=0.9025,
                                               batch_size=32,
                                               epochs=20,
                                               sampling_strategy='HardNegativePair',
                                               negatives_cnt=3,
                                               marging=0.5,
                                               step_size=1,
                                               model_postfix='per_sampl')
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

print(f'model _params:\n\
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
       ')