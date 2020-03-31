import torch
import json
import constants as cns
import tqdm
import pickle
import pandas as pd
import numpy as np
from data_load import load_temporal_data, PaddedBatch
from train import batch_to_device
from model import NormEncoder
import torch.nn as nn
import matplotlib.pyplot as plt
from ml_metrics import mapk

N_CEIDS, N_UIDS = cns.N_CEIDS, cns.N_UIDS
TOP_K = 20

cns.BATCH_SIZE = 1


def get_model():
    model_path = f'/data/molchanov/okko/ils_via_backprop/model_weights_several_test_with_separation_epoch_9'
    #model_path = f'/data/molchanov/okko/ils_via_backprop/test_weights'
    print(f'loading model: {model_path}')
    model = torch.load(model_path)
    model = model[:-1]
    # not good but ...
    model[-1][-1].features = ['ceid']
    model.cuda()
    model.eval()
    return model


def encode_ceids(model):
    eids = torch.arange(1, N_CEIDS).cuda()
    data = PaddedBatch({
        'ceid': eids.view(1, -1)
    }, None)

    s_model = model[0][1]
    eids_codes = s_model(data).payload
    #eids_codes = model(data)[1].payload
    eids_codes = NormEncoder()(eids_codes)
    return eids_codes


def validate_several_temporal_on_validation_dataset():
    # get model
    model = get_model()
    n = NormEncoder()
    sig = nn.Sigmoid()

    # encode cieds with model
    eids_codes = encode_ceids(model)

    # data
    top1_acc, top5_acc, top10_acc, top20_acc = 0.0, 0.0, 0.0, 0.0
    ea = 0.0
    data_train, data_valid = load_temporal_data()
    l = len(data_valid)
    recomendations1, recomendations5, recomendations10, recomendations20 = [], [], [], []
    with tqdm.tqdm(total=l) as steps:

        for k, input_data in enumerate(data_valid):

            known_codes = set(input_data[0].payload['ceid'].view(-1)[:-cns.TEST_WINDOW].numpy())
            codes = set(input_data[0].payload['ceid'].view(-1)[-cns.TEST_WINDOW:].numpy())

            cds = input_data[0].payload['ceid'].view(-1)[-cns.TEST_WINDOW:].numpy().tolist()

            device_data = batch_to_device(input_data, torch.device('cuda'), True)
            e1, _ = model(device_data[0])
            last_state = e1.payload[:, -cns.TEST_WINDOW - 1:-cns.TEST_WINDOW:, :]
            last_state = n(last_state).transpose(1, 2)
            scores = torch.matmul(eids_codes, last_state).squeeze()
            scores = sig(scores)

            out, idxs = torch.sort(scores, descending=True)

            ots = []
            top_1, top_5, top_10, top_20 = set(), set(), set(), set()
            for prob, idx in zip(out, idxs):
                if idx.item() + 1 not in known_codes:
                    ots.append(idx.item() + 1)
                    if len(top_1) < 1:
                        top_1.add(idx.item() + 1)

                    if len(top_5) < 5:
                        top_5.add(idx.item() + 1)

                    if len(top_10) < 10:
                        top_10.add(idx.item() + 1)

                    if len(top_20) < 20:
                        top_20.add(idx.item() + 1)
                    else:
                        break

            if len(codes.intersection(top_1)) > 0:
                top1_acc += 1

            if len(codes.intersection(top_5)) > 0:
                top5_acc += 1

            if len(codes.intersection(top_10)) > 0:
                top10_acc += 1

            if len(codes.intersection(top_20)) > 0:
                top20_acc += 1

            ea += mapk([cds], [ots[cns.TEST_WINDOW:]], cns.TEST_WINDOW)

            top1_acc_ = float(top1_acc) / (k+1)
            top5_acc_ = float(top5_acc) / (k+1)
            top10_acc_ = float(top10_acc) / (k+1)
            top20_acc_ = float(top20_acc) / (k+1)

            recomendations1.extend(list(top_1))
            recomendations5.extend(list(top_5))
            recomendations10.extend(list(top_10))
            recomendations20.extend(list(top_20))

            steps.set_description('working...')
            steps.set_postfix({'top1_acc': top1_acc_, 'top5_acc': top5_acc_, 'top10_acc': top10_acc_, 'top20_acc': top20_acc_, 'MAP@k': ea/(k+1)})
            steps.update()

            # plt.hist(out.detach().cpu().numpy(), bins=100)
            # plt.show()
            # plt.close()

    plt.hist(recomendations1.numpy(), bins=100)
    plt.show()
    plt.close()

    plt.hist(recomendations5.numpy(), bins=100)
    plt.show()
    plt.close()

    plt.hist(recomendations10.numpy(), bins=100)
    plt.show()
    plt.close()

    plt.hist(recomendations20.numpy(), bins=100)
    plt.show()
    plt.close()

    top1_acc = float(top1_acc) / l
    top5_acc = float(top5_acc) / l
    top10_acc = float(top10_acc) / l
    top20_acc = float(top20_acc) / l

    print(f'top1 {top1_acc}, top5 {top5_acc}, top10 {top10_acc}, top20 {top20_acc}')


validate_several_temporal_on_validation_dataset()