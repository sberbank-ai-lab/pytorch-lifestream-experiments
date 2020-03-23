from data_load import load_temporal_ordinal_with_features_data
from train import fit_model_ordinal
from model import get_model_ordinal_with_features
from loss import OrdinalLoss
import constants as cns
import torch
import torch.nn as nn
from copy import deepcopy

torch.autograd.set_detect_anomaly(True)

cns.BATCH_SIZE = 64
cns.EMBEDDING_DIM = 16#256
cns.B_TEST_DATA = False
print(f'Attention: using predefinde constants: BATCH_SIZE = {cns.BATCH_SIZE}, EMBEDDING_DIM = {cns.EMBEDDING_DIM}, SMALL_DATASET = {cns.B_TEST_DATA}')


def main():

    data_loader, valid_data_loader = load_temporal_ordinal_with_features_data()

    model = get_model_ordinal_with_features()

    loss = OrdinalLoss()

    fit_model_ordinal(model, data_loader, valid_data_loader, loss)

main()