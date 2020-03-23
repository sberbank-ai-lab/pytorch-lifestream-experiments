from data_load import load_temporal_classification_data
from train import fit_model_classification
from model import get_model_calssification
from loss import KlLoss
import constants as cns
import torch
import torch.nn as nn
from copy import deepcopy

torch.autograd.set_detect_anomaly(True)

cns.BATCH_SIZE = 64
cns.EMBEDDING_DIM = 512
print(f'Attention: using predefinde constants: BATCH_SIZE = {cns.BATCH_SIZE}, EMBEDDING_DIM = {cns.EMBEDDING_DIM}')


def main():

    data_loader, valid_data_loader = load_temporal_classification_data()

    model = get_model_calssification()

    loss = KlLoss()

    fit_model_classification(model, data_loader, valid_data_loader, loss)


main()