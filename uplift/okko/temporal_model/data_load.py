import pandas as pd
import numpy as np
import tqdm
import re
from torch.utils.data import Dataset, DataLoader
import torch
import pickle
import constants as cns
import random
from copy import deepcopy
from constants import extend_add_info as exai

# ----------------------------------------------------------------------------------------------------------------------------------------------------------


class PaddedBatch:
    def __init__(self, payload, length, add_info=None):
        self._payload = payload
        self._length = length
        self._add_info = add_info

    @property
    def payload(self):
        return self._payload

    @property
    def seq_lens(self):
        return self._length

    @property
    def add_info(self):
        return self._add_info


# ----------------------------------------------------------------------------------------------------------------------------------------------------------
# DataSet class

class TemporalDataSet(Dataset):

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        rec = self.data[idx]['feature_arrays']

        ceids1, ceids2 = rec['pos_ceid'][:, -100:], rec['neg_ceid'][:, -100:]
        target1, target2 = rec['target'][-100:], rec['neg_target'][-100:]
        rating = rec['rating'][-101:]

        # get random subset from neg ceids
        idx = np.random.choice(cns.TRAIN_WINDOW * cns.NEG_COEF, size=cns.TRAIN_WINDOW, replace=False)
        ceids2 = ceids2[idx]

        ceids = np.split(ceids1, ceids1.shape[0]) + np.split(ceids2, ceids2.shape[0])
        x = {}
        for i, ar in enumerate(ceids):
            x[f'ceids_{i + 1}'] = torch.LongTensor(ar).view(-1)
        x['ceid'] = torch.LongTensor(rec['ceid'][-101:])

        y = {'target1': torch.LongTensor(target1),
             'target2': torch.LongTensor(target2),
             'rating' : torch.Tensor(rating)
             }
        return x, y


class TemporalClassificationDataSet(Dataset):

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        rec = self.data[idx]['feature_arrays']

        pos_ceid = torch.LongTensor(rec['pos_ceid'][:, -100:])

        idx_0 = torch.arange(pos_ceid.size(1) * pos_ceid.size(0)) % pos_ceid.size(1)
        idx_1 = pos_ceid.view(-1)

        target = torch.zeros(pos_ceid.size(1), cns.N_CEIDS)
        target[[idx_0, idx_1]] = float(1) / pos_ceid.size(0)
        target.cuda()

        x = {'ceid': torch.LongTensor(rec['ceid'][-101:])}

        rating = rec['rating'][-101:]

        y = {'target': target, 'rating' : torch.Tensor(rating)}
        return x, y


class TemporalClassificationWithFeaturesDataSet(Dataset):

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        rec = self.data[idx]['feature_arrays']

        pos_ceid = torch.LongTensor(rec['pos_ceid'][:, -100:]).contiguous()

        idx_0 = torch.arange(pos_ceid.size(1) * pos_ceid.size(0)) % pos_ceid.size(1)
        idx_1 = pos_ceid.view(-1)

        target = torch.zeros(pos_ceid.size(1), cns.N_CEIDS)
        target[[idx_0, idx_1]] = float(1) / pos_ceid.size(0)
        target.cuda()

        def to_tensor(val, key):
            if cns.FEATURES[key]['type'] == 'cat':
                return torch.LongTensor(val)
            else:
                return torch.FloatTensor(val)

        # other feaures
        x = {k: to_tensor(rec[k][-101:], k) for k in cns.FEATURES.keys()}
        x['ceid'] = torch.LongTensor(rec['ceid'][-101:])

        #rating = rec['rating'][-101:]

        y = {'target': target}#, 'rating': torch.Tensor(rating)}

        return x, y


class RecomendationTemporalClassificationWithFeaturesDataSet(Dataset):

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        rec = self.data[idx]['feature_arrays']

        pos_ceid = torch.LongTensor(rec['pos_ceid'][:, -cns.MAX_SEQ_LEN:]).contiguous()

        idx_0 = torch.arange(pos_ceid.size(1) * pos_ceid.size(0)) % pos_ceid.size(1)
        idx_1 = pos_ceid.view(-1)

        target = torch.zeros(pos_ceid.size(1), cns.N_CEIDS)
        target[[idx_0, idx_1]] = float(1) / pos_ceid.size(0)
        target.cuda()

        def to_tensor(val, key):
            if cns.FEATURES[key]['type'] == 'cat':
                return torch.LongTensor(val)
            else:
                return torch.FloatTensor(val)

        # other feaures
        x = {k: to_tensor(rec[k][-(cns.MAX_SEQ_LEN + 1):], k) for k in cns.FEATURES.keys()}
        x['ceid'] = torch.LongTensor(rec['ceid'][-(cns.MAX_SEQ_LEN + 1):])

        #rating = rec['rating'][-101:]

        y = {'target': target}#, 'rating': torch.Tensor(rating)}

        # for prediction model only
        if cns.DEPLOY_MODE and cns.USER_ID_COLUMN in self.data[idx]:
            y[cns.USER_ID_COLUMN] = torch.LongTensor([self.data[idx][cns.USER_ID_COLUMN]]) # to map prediction to specific user
            y['prev_ceids'] = torch.LongTensor(rec['ceid']) # to not predict already seen items
        return x, y


class TemporalOrdinalWithFeaturesDataSet(Dataset):

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        rec = self.data[idx]['feature_arrays']

        pos_ceid = torch.LongTensor(rec['pos_ceid'][:, -100:])

        idx_0 = torch.arange(pos_ceid.size(1) * pos_ceid.size(0)) % pos_ceid.size(1)
        idx_1 = pos_ceid.view(-1)

        target_mask = torch.zeros(pos_ceid.size(1), cns.N_CEIDS)
        target_mask[[idx_0, idx_1]] = float(1)
        target_mask.cuda()

        def to_tensor(val, key):
            if cns.FEATURES[key]['type'] == 'cat':
                return torch.LongTensor(val)
            else:
                return torch.FloatTensor(val)

        # other features
        x = {k: to_tensor(rec[k][-101:], k) for k in cns.FEATURES.keys()}
        x['ceid'] = torch.LongTensor(rec['ceid'][-101:])

        y = {f'poc_ceid_{i}': pc for i, pc in zip(range(pos_ceid.size(0)), pos_ceid)}
        y['target_mask'] = target_mask.view(-1)
        return x, y


class TemporalPopularDataSet(Dataset):

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        rec = self.data[idx]['feature_arrays']

        pos_ceid = torch.LongTensor(rec['pos_ceid'][:, -100:])

        x = {'ceid': torch.LongTensor(rec['ceid'][-101:])}
        y = {f'poc_ceid_{i}': pc for i, pc in zip(range(pos_ceid.size(0)), pos_ceid)}

        return x, y


# ----------------------------------------------------------------------------------------------------------------------------------------------------------
# Combine in Batch method

def padded_temporal_collate(batch):
    new_x_ = {}
    for x, _ in batch:
        for k, v in x.items():
            if k in new_x_:
                new_x_[k].append(v)
            else:
                new_x_[k] = [v]

    new_y_ = {}
    for _, y in batch:
        for k, v in y.items():
            if k in new_y_:
                new_y_[k].append(v)
            else:
                new_y_[k] = [v]

    lengths = torch.IntTensor([len(e) for e in next(iter(new_x_.values()))])

    new_x = {k: torch.nn.utils.rnn.pad_sequence(v, batch_first=True) for k, v in new_x_.items()}
    new_y = {k: torch.nn.utils.rnn.pad_sequence(v, batch_first=True) for k, v in new_y_.items()}

    d = None

    return PaddedBatch(new_x, lengths, exai(None, 'distances', d)), new_y


# ----------------------------------------------------------------------------------------------------------------------------------------------------------
# Create loader method

def create_temporal_loader(dataset):
    dataset = TemporalDataSet(dataset)

    valid_loader = DataLoader(
        dataset,
        batch_size=cns.BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        collate_fn=padded_temporal_collate)

    return valid_loader


def create_temporal_classification_loader(dataset):
    dataset = TemporalClassificationDataSet(dataset)

    valid_loader = DataLoader(
        dataset,
        batch_size=cns.BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        collate_fn=padded_temporal_collate)

    return valid_loader


def create_temporal_classification_with_features_loader(dataset):
    dataset = TemporalClassificationWithFeaturesDataSet(dataset)

    valid_loader = DataLoader(
        dataset,
        batch_size=cns.BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        collate_fn=padded_temporal_collate)

    return valid_loader


def create_temporal_ordinal_with_features_loader(dataset):
    dataset = TemporalOrdinalWithFeaturesDataSet(dataset)

    valid_loader = DataLoader(
        dataset,
        batch_size=cns.BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        collate_fn=padded_temporal_collate)

    return valid_loader


def create_temporal_popular_loader(dataset):
    dataset = TemporalPopularDataSet(dataset)

    valid_loader = DataLoader(
        dataset,
        batch_size=cns.BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        collate_fn=padded_temporal_collate)

    return valid_loader


# ----------------------------------------------------------------------------------------------------------------------------------------------------------
# Load Data Method

def load_temporal_data():
    with open(f'/data/molchanov/okko/ils_via_backprop/data/temporal_dataset_several_strong_negative_sampling_pickle2', 'rb') as handle:
        data = pickle.load(handle)
        if cns.B_TEST_DATA:
            data = data[:2000]
        data = [d for d in data if len(d['feature_arrays']['target']) >= cns.TRAIN_WINDOW*2 + cns.TEST_WINDOW]
        #data = data[:30000]
        valid_data = [deepcopy(d) for d in data]
        return create_temporal_loader(data), create_temporal_loader(valid_data)


def load_temporal_classification_data():
    with open(f'/data/molchanov/okko/ils_via_backprop/data/temporal_dataset_several_strong_negative_sampling_pickle2', 'rb') as handle:
        data = pickle.load(handle)
        if cns.B_TEST_DATA:
            data = data[:2000]
        data = [d for d in data if len(d['feature_arrays']['target']) >= cns.TRAIN_WINDOW*2 + cns.TEST_WINDOW]
        #data = data[:30000]
        valid_data = [deepcopy(d) for d in data]
        return create_temporal_classification_loader(data), create_temporal_classification_loader(valid_data)


def load_temporal_classification_with_features_data():
    with open(f'/data/molchanov/okko/ils_via_backprop/data/temporal_dataset_with_features_pickle', 'rb') as handle:
        data = pickle.load(handle)
        if cns.B_TEST_DATA:
            data = data[:2000]
        data = [d for d in data if len(d['feature_arrays']['target']) >= cns.TRAIN_WINDOW*2 + cns.TEST_WINDOW]
        #data = data[:30000]
        valid_data = [deepcopy(d) for d in data]
        return create_temporal_classification_with_features_loader(data), create_temporal_classification_with_features_loader(valid_data)


def load_temporal_ordinal_with_features_data():
    with open(f'/data/molchanov/okko/ils_via_backprop/data/temporal_dataset_with_features_pickle', 'rb') as handle:
        data = pickle.load(handle)
        if cns.B_TEST_DATA:
            data = data[:2000]
        data = [d for d in data if len(d['feature_arrays']['target']) >= cns.TRAIN_WINDOW*2 + cns.TEST_WINDOW]
        #data = data[:30000]
        valid_data = [deepcopy(d) for d in data]
        return create_temporal_ordinal_with_features_loader(data), create_temporal_ordinal_with_features_loader(valid_data)


def load_popular_data():
    with open(f'/data/molchanov/okko/ils_via_backprop/data/temporal_dataset_with_features_pickle', 'rb') as handle:
        data = pickle.load(handle)
        if cns.B_TEST_DATA:
            data = data[:2000]
        data = [d for d in data if len(d['feature_arrays']['target']) >= cns.TRAIN_WINDOW*2 + cns.TEST_WINDOW]
        #data = data[:30000]

        popular = pd.read_csv(f'/data/molchanov/okko/ils_via_backprop/data/popular.csv')
        popular = list(popular['ceid'])
        return create_temporal_popular_loader(data), popular

