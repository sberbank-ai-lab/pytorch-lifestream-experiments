import logging
import torch

from agg_features_ts_preparation import load_agg_data
from dltranz.baselines.cpc import CPC_Encoder, run_experiment
from dltranz.custom_layers import MLP, ReshapeWrapper
from dltranz.data_load.fast_tensor_data_loader import FastTensorDataLoader
from dltranz.util import init_logger, get_conf, switch_reproducibility_on

logger = logging.getLogger(__name__)

if __name__ == '__main__':
    switch_reproducibility_on()


def create_data_loaders(conf, load_data_function):
    data = load_data_function(conf)
    r = torch.randperm(len(data))
    data = data[r]

    valid_size = int(len(data) * conf['dataset.valid_size'])
    train_data, valid_data = torch.split(data, [len(data) - valid_size, valid_size])
    logger.info(f'Train data len: {len(train_data)}, Valid data len: {len(valid_data)}')

    train_loader = FastTensorDataLoader(
        train_data, torch.zeros(len(train_data)),
        batch_size=conf['params.train.batch_size'],
        shuffle=True
    )

    valid_loader = FastTensorDataLoader(
        valid_data, torch.zeros(len(valid_data)),
        batch_size=conf['params.valid.batch_size']
    )

    return train_loader, valid_loader


def main(args=None):
    conf = get_conf(args)

    train_loader, valid_loader = create_data_loaders(conf, load_data_function=load_agg_data)

    encoder = ReshapeWrapper(
        MLP(conf['dataset.features_count'], conf['params.mlp'])
    )

    cpc_e = CPC_Encoder(encoder, conf['params.cpc'])

    run_experiment(train_loader, valid_loader, cpc_e, conf, is_padded=False)

    if conf.get('save_model', False):
        agg_model = torch.load(conf['model_path.agg_model'])
        enc_agr_model = torch.nn.Sequential(agg_model, encoder.model)

        torch.save(enc_agr_model, conf['model_path.model'])
        logger.info(f'Model saved to "{conf["model_path.model"]}"')


if __name__ == '__main__':
    init_logger(__name__)
    init_logger('dltranz')

    main()
