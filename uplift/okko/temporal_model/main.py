from data_load import load_temporal_data
from train import fit_model
from model import get_model
from loss import PairwiseMarginRankingLoss
import constants as cns
import torch
from copy import deepcopy

torch.autograd.set_detect_anomaly(True)

cns.BATCH_SIZE = 64
cns.EMBEDDING_DIM = 1024
print(f'Attention: using predefinde constants: BATCH_SIZE = {cns.BATCH_SIZE}, EMBEDDING_DIM = {cns.EMBEDDING_DIM}')


def main():

    data_loader, valid_data_loader = load_temporal_data()

    model = get_model()

    loss = PairwiseMarginRankingLoss(margin=cns.MARGIN)

    fit_model(model, data_loader, valid_data_loader, loss)


main()