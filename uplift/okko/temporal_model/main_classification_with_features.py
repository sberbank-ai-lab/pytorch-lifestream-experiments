from data_load import load_temporal_classification_with_features_data
from train import fit_model_classification
from model import get_model_calssification_with_features
from loss import KlLoss
import constants as cns
import torch
import torch.nn as nn
from copy import deepcopy

torch.autograd.set_detect_anomaly(True)

cns.BATCH_SIZE = 64
cns.EMBEDDING_DIM = 256
cns.B_TEST_DATA = False
print(f'Attention: using predefinde constants: BATCH_SIZE = {cns.BATCH_SIZE}, EMBEDDING_DIM = {cns.EMBEDDING_DIM}, SMALL_DATASET = {cns.B_TEST_DATA}')
print(f' features:')
for k, v in cns.FEATURES:
    print(k)
print('-')


def main():

    data_loader, valid_data_loader = load_temporal_classification_with_features_data()

    model = get_model_calssification_with_features()

    loss = KlLoss()

    fit_model_classification(model, data_loader, valid_data_loader, loss)


def main_ext():
    cns.N_EPOCH = 2
    #train_features = [[], ['t_delta_1'], ['t_delta_1'], ['t_delta_1', 't_delta_2'], ['rating'], ['watched_rating'],
    #                  ['feature_1'], ['feature_2'], ['feature_4'], ['feature_5'], ['bookmark'], ['device_type'],
    #                  ['device_manufacturer'], ['purchase'], ['rent'], ['subscription'], ['feature_3']]
    train_features = [['t_delta_1', 't_delta_2', 'feature_1']]

    for f in train_features:
        cns.FEATURES = {k: v for k, v in cns.ALL_FEATURES.items() if k in f}
        print(f'\nTRAIN ON FEATURES: {cns.FEATURES.keys()}\n')
        main()

main()
#main_ext()

