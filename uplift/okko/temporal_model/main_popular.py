from data_load import load_popular_data
from train import fit_model_popular
import constants as cns
import torch
import torch.nn as nn
from copy import deepcopy

torch.autograd.set_detect_anomaly(True)

cns.BATCH_SIZE = 512


def main():

    data_loader, popular = load_popular_data()

    fit_model_popular(data_loader, popular)


main()