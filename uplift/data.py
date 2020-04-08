import torch
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import random
import pickle
from collections import defaultdict
import ast
import pandas as pd
from torch.distributions.multivariate_normal import MultivariateNormal

from uplift.constants import *


def mnist_train_dataset():
    tmp = torchvision.datasets.MNIST('/mnt/data/molchanov/datasets/mnist', 
                                           train=True, 
                                           transform=None, 
                                           target_transform=None, 
                                           download=True)
    return tmp

def mnist_test_dataset():
    tmp = torchvision.datasets.MNIST('/mnt/data/molchanov/datasets/mnist', 
                                           train=False, 
                                           transform=None, 
                                           target_transform=None, 
                                           download=True)
    return tmp

def cifar10_train_dataset():
    tmp = torchvision.datasets.CIFAR10('/mnt/data/molchanov/datasets/cifar10', 
                                               train=True, 
                                               transform=None, 
                                               target_transform=None, 
                                               download=True)
    return tmp

def cifar10_test_dataset():
    tmp = torchvision.datasets.CIFAR10('/mnt/data/molchanov/datasets/cifar10', 
                                               train=False, 
                                               transform=None, 
                                               target_transform=None, 
                                               download=True)
    return tmp

def okko_metriclearn_datasets(n_augments):
    with open(f'/mnt/data/molchanov/datasets/okko/rekko_challenge_rekko_challenge_2019/transactions_grouped.pkl', 'rb') as handle:
        data = pickle.load(handle)
        data = [d for d in data if d['len'] >= 30]
        data = np.random.permutation(data)
        
        split_len = int(len(data)*0.7)
        train_data, test_data = data[:split_len], data[:split_len]
    return OkkoMetricLearnDataSet(train_data, n_augments), OkkoMetricLearnDataSet(test_data, n_augments)

def okko_domyshnik_datasets(n_augments):
    with open(f'/mnt/data/molchanov/datasets/okko/rekko_challenge_rekko_challenge_2019/transactions_grouped.pkl', 'rb') as handle:
        data = pickle.load(handle)
        data = [d for d in data if d['len'] >= 30]
        data = np.random.permutation(data)
        
        split_len = int(len(data)*0.7)
        train_data, test_data = data[:split_len], data[split_len:]
    return OkkoDomyshnikDataSet(train_data, n_augments), OkkoDomyshnikDataSet(test_data, n_augments)

def criteo_control_datasets():
    df = pd.read_csv('/mnt/data/molchanov/datasets/criteo/criteo-uplift-v2.1.csv')
    df_t = df[df['treatment'] == 1]

    # zeros sampling
    n0 = df_t[df_t['conversion'] == 0]['conversion'].values.shape[0]
    n1 = df_t[df_t['conversion'] == 1]['conversion'].values.shape[0]
    n = int((n0 + n1) * float(n1)/n0)
    A = df_t[df_t['conversion'] == 0].values
    zeros = A[np.random.choice(A.shape[0], n, replace=False), :]
    zeros = pd.DataFrame(zeros, columns=df_t.columns)
    ones = df_t[df_t['conversion'] == 1]
    df_t = pd.concat((ones, zeros))

    # split train test
    B = df_t.values
    idx = np.random.permutation(B.shape[0])
    n_train = int(B.shape[0] * 0.7)
    train, test = B[idx[:n_train], :], B[idx[n_train:], :] 
    df_train = pd.DataFrame(train, columns=df_t.columns)
    df_test  = pd.DataFrame(test, columns=df_t.columns)

    y_train, y_test = df_train['conversion'].values, df_test['conversion'].values
    df_train = df_train.drop(['visit', 'treatment', 'conversion'], axis=1).values
    df_test = df_test.drop(['visit', 'treatment', 'conversion'], axis=1).values
    
    return df_train, y_train, df_test, y_test
# -------------------------------------------------------------------------------------------------

def mnist_torch_augmentation(p=1):
    return torchvision.transforms.Compose([
        transforms.RandomChoice([
            transforms.RandomAffine(degrees=20, 
                                translate=(0.25, 0.25), 
                                scale=(0.8, 0.8), 
                                shear=None, 
                                resample=False, 
                                fillcolor=0),
            transforms.RandomPerspective(distortion_scale=0.5, p=0.5, interpolation=3)
        ]),
        transforms.ColorJitter(brightness=0.7, contrast=0.5, saturation=0.5, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        #transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False)
    ])

def cifar_torch_augmentation_strong(p=1):
    return torchvision.transforms.Compose([
                
        transforms.RandomApply([
            transforms.RandomChoice([
                transforms.RandomAffine(degrees=15,#45, 
                                        translate=(0.1, 0.1),#(0.2, 0.2), 
                                        scale=(0.7, 1.3), 
                                        shear=None,#10, 
                                        resample=False),
                transforms.RandomResizedCrop(size=32, 
                                             scale=(0.8, 1.2),#(0.6, 1.5), 
                                             ratio=(0.75, 1.3), 
                                             interpolation=2)]
            )], p=0.5
        ),
        
        transforms.RandomApply([
            transforms.ColorJitter(brightness=0.1, 
                                   contrast=0.1,#0.7, 
                                   saturation=(0.0, 0.5),#(0.5, 1), 
                                   hue=0.1)], p=0.5
        ),
        
        transforms.RandomApply([
            transforms.RandomVerticalFlip(),
            transforms.RandomHorizontalFlip()], p=0.5
        ),
        
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
    ])

def cifar_torch_augmentation(p=1):
    return torchvision.transforms.Compose([
        transforms.RandomApply([
            transforms.RandomResizedCrop(size=32, scale=(0.5, 1.0))
            ], p=0.5),
        
        transforms.RandomApply([
            transforms.ColorJitter(brightness=0.3, 
                                   contrast=0.3,#0.7, 
                                   saturation=(0.3, 0.5),#(0.5, 1), 
                                   hue=0.5)
            ], p=0.5),
        
        transforms.RandomApply([
            transforms.RandomHorizontalFlip()
        ], p=0.7),
        
        transforms.ToTensor(),
        #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
    ])

def cifa10_base_aug():
    return torchvision.transforms.Compose([
        transforms.ToTensor(),
        #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
    ]
    )

def cifar10_normilise(img):
    return transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))(img) 

def padded_collate_okko_metrlearn(batch):
    
    new_x = defaultdict(list)
    lengths = []
    for sample in batch:
        for i, (feature_name, feature_augments_collection) in enumerate(sample.items()):
            for augment_val in feature_augments_collection:
                new_x[feature_name].append(augment_val)
                if i == 0:
                    lengths.append(augment_val.size(0))

    lengths = torch.IntTensor(lengths)

    out = {k: torch.nn.utils.rnn.pad_sequence(v, batch_first=True) for k, v in new_x.items()}

    return PaddedBatch(out, lengths)

def padded_collate_criteo_metrlearn(batch):
    
    new_x = defaultdict(list)
    lengths = []
    targets = []
    for sample, y in batch:
        targets.append(y)
        for i, (feature_name, feature_augments_collection) in enumerate(sample.items()):
            for augment_val in feature_augments_collection:
                new_x[feature_name].append(augment_val)
                if i == 0:
                    lengths.append(augment_val.size(0))

    lengths = torch.IntTensor(lengths)

    out = {k: torch.nn.utils.rnn.pad_sequence(v, batch_first=True) for k, v in new_x.items()}

    return PaddedBatch(out, lengths), torch.LongTensor(targets)

def padded_collate_okko_domyshnik(batch):
    
    new_x = defaultdict(list)
    lengths = []
    for sample, _ in batch:
        for i, (feature_name, feature_augments_collection) in enumerate(sample.items()):
            for augment_val in feature_augments_collection:
                new_x[feature_name].append(augment_val)
                if i == 0:
                    lengths.append(augment_val.size(0))

    lst_true_lbls, lst_fake_labels, lst_rewards = [], [], []
    for _, (true_lbls, fake_labels, rewards) in batch:
        lst_true_lbls.append(true_lbls)
        lst_fake_labels.append(fake_labels)
        lst_rewards.append(rewards)
    trues = torch.cat(lst_true_lbls, 0)
    fakes = torch.cat(lst_fake_labels, 0)
    rewards = torch.cat(lst_rewards, 0)

    lengths = torch.IntTensor(lengths)

    out = {k: torch.nn.utils.rnn.pad_sequence(v, batch_first=True) for k, v in new_x.items()}

    return PaddedBatch(out, lengths), (trues, fakes, rewards)

def padded_collate_criteo_rnn_domyshnik(batch):
    
    new_x = defaultdict(list)
    lengths = []
    for sample, _ in batch:
        for i, (feature_name, feature_augments_collection) in enumerate(sample.items()):
            for augment_val in feature_augments_collection:
                new_x[feature_name].append(augment_val)
                if i == 0:
                    lengths.append(augment_val.size(0))

    lst_true_lbls, lst_fake_labels, lst_rewards = [], [], []
    for _, (true_lbls, fake_labels, rewards) in batch:
        lst_true_lbls.append(true_lbls)
        lst_fake_labels.append(fake_labels)
        lst_rewards.append(rewards)
    trues = torch.cat(lst_true_lbls, 0)
    fakes = torch.cat(lst_fake_labels, 0)
    rewards = torch.cat(lst_rewards, 0)

    lengths = torch.IntTensor(lengths)

    out = {k: torch.nn.utils.rnn.pad_sequence(v, batch_first=True) for k, v in new_x.items()}

    return PaddedBatch(out, lengths), (trues, fakes, rewards)
# -------------------------------------------------------------------------------------------------
class MetrLearnDataset(torch.utils.data.Dataset):
    
    def __init__(self, dataset, augmenter, n_augments=10, augment_labels=False):
        self.data = dataset
        self.aug = augmenter
        self.n_augments = n_augments
        self.augment_labels = augment_labels
        self.base_aug = cifa10_base_aug()
        
    def draw(self, idx):
        imgs, _ = self[idx]
        fig = plt.figure()
        rows, columns = 1, imgs.shape[0]
        for i in range(imgs.shape[0]):
            fig.add_subplot(rows, columns, i+1)
            plt.imshow(imgs[i])
        plt.show()
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, lbl = self.data[idx]

        if self.n_augments > 0:
            #imgs = [self.aug(img) for i in range(self.n_augments + 1)]
            imgs = [self.base_aug(img)] + [self.aug(img) for i in range(self.n_augments - 1)]
            b_img = torch.stack(imgs).squeeze()

        elif self.n_augments == -1: # no augments, return original image
            b_img = self.base_aug(img)

        elif self.n_augments == 0: # one augment with probability
            b_img = self.aug(img) if random.random() > 0.5 else self.base_aug(img)

        elif self.n_augments == -2: # raw pil image
            b_img = transforms.ToTensor()(img)

        if not self.augment_labels:
            return b_img, lbl
        else:
            if random.random() > ERROR_RATE:
                new_lbl = random.randint(0, NUM_CLASSES - 1)
                while new_lbl == lbl:
                    new_lbl = random.randint(0, NUM_CLASSES - 1)
            else:
                new_lbl = lbl
            reward = -1.0 if lbl == new_lbl else BAD_REWARD
            return b_img, (new_lbl, lbl, reward)
        
class DataLoaderWrapper:

    def __init__(self, base_loader, centroids, n_augments=0, augmenter=None):
        self.loader = base_loader
        self.base_aug = cifa10_base_aug()

        # centroids for augmentation
        toPil = transforms.ToPILImage()
        self.centroids = [toPil(centroid) for centroid in centroids]
        self.centroids_count = len(self.centroids)

        # original centroids
        #norm = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
        #self.centers = [norm(centroid) for centroid in centroids]
        #self.centers = torch.stack(self.centers)
        self.centers = centroids

        self.n_augments = n_augments
        self.aug = augmenter

    def get_centroids(self):
        if self.aug is not None:
            augmentations = []
            for centroid in self.centroids:
                centroid_augments = [self.base_aug(centroid)] + [self.aug(centroid) for i in range(self.n_augments - 1)]
                augmentations += centroid_augments
            augmented_centroids = torch.stack(augmentations, 0)

            labels = torch.arange(self.centroids_count).view(1, -1).repeat(self.n_augments, 1).transpose(0, 1).flatten()

            return augmented_centroids, labels
        return self.centers, torch.arange(self.centroids_count)

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        for data in self.loader:
            centrs = self.get_centroids()
            yield data, centrs

class OkkoMetricLearnDataSet(torch.utils.data.Dataset):

    def __init__(self, data, n_augments):
        self.data = data
        self.n_augments = n_augments

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        rec = self.data[idx]['feature_arrays']
        l = self.data[idx]['len']

        def to_tensor(val, key):
            if FEATURES[key]['type'] == 'cat':
                return torch.LongTensor(val)
            else:
                return torch.FloatTensor(val)

        x = defaultdict(list)
        for _ in range(self.n_augments):
            idx = torch.randint(0, 2, (l,)).bool()

            for k in sorted(list(FEATURES.keys())):
                tmp = to_tensor(rec[k], k)
                tmp = tmp.masked_select(idx)
                x[k].append(tmp[-125:])

        return x

class CriteoMetricLearnDataSet(torch.utils.data.Dataset):

    def __init__(self, X, y, n_augments, augment_labels=False):
        self.X = X
        self.y = y
        self.n_augments = n_augments
        self.noise = None
        self.augment_labels = augment_labels

    def __len__(self):
        return self.y.shape[0]

    def augment(self, x):
        if self.noise is None:
            self.noise = MultivariateNormal(torch.zeros(x.size(0)), torch.eye(x.size(0))*0.05)
        x += self.noise.sample()

        # sub sample
        while True:
            codes = torch.arange(x.size(0))
            mask = torch.bernoulli(0.75 * torch.ones(x.size(0))).bool()
            mask[-1] = True # take exposure feature
            if mask.int().sum() > 6:
                break

        mcodes = codes[mask]
        mx = x[mask]

        # random permute
        perm = torch.randperm(mcodes.size(0))
        tmp = mcodes[perm[0]].item()
        mcodes[perm[0]] = mcodes[perm[1]]
        mcodes[perm[1]] = tmp

        tmp = mx[perm[0]].item()
        mx[perm[0]] = mx[perm[1]]
        mx[perm[1]] = tmp

        return mx, mcodes

    def __getitem__(self, idx):
        rec = torch.Tensor(self.X[idx])
        y = torch.LongTensor([self.y[idx]])

        x = defaultdict(list)
        for _ in range(self.n_augments):
            rec_aug, codes = self.augment(rec)
            x['codes'].append(codes)
            x['features'].append(rec_aug)

        if self.augment_labels:
            if random.random() > ERROR_RATE:
                return x, (1 - y, y, torch.Tensor([BAD_REWARD])) # augs, new_lbl, true_lbl, reward
            else:
                return x, (y, y, torch.Tensor([-1.0])) # augs, new_lbl, true_lbl, reward

        return x, y

class CriteoMetricLearnDataSet2(torch.utils.data.Dataset):

    def __init__(self, X, y, n_augments, augment_labels=False):
        self.X = X
        self.y = y
        self.n_augments = n_augments
        self.noise = None
        self.augment_labels = augment_labels

    def __len__(self):
        return self.y.shape[0]

    def augment(self, x):
        # noise
        if self.noise is None:
            self.noise = MultivariateNormal(torch.zeros(x.size(0)), torch.eye(x.size(0))*0.05)
        x += self.noise.sample()

        # sub sample
        while True:
            mask = torch.bernoulli(0.75 * torch.ones(x.size(0)))
            mask[-1] = True # take exposure feature
            if mask.int().sum() > 6:
                break

        x *= mask
        return x

    def __getitem__(self, idx):
        rec = torch.Tensor(self.X[idx])
        y = torch.LongTensor([self.y[idx]])

        augs = [rec] + [self.augment(rec) for i in range(self.n_augments - 1)]
        augs = torch.stack(augs, dim=0)

        if self.augment_labels:
            if random.random() > ERROR_RATE:
                return augs, (y, 1 - y, torch.Tensor([BAD_REWARD])) # augs, true_lbl, new_lbl, reward
            else:
                return augs, (y, y, torch.Tensor([-1.0])) # augs, true_lbl, new_lbl, reward

        return augs, y

class OkkoDomyshnikDataSet(torch.utils.data.Dataset):

    def __init__(self, data, n_augments):
        self.data = data
        self.n_augments = n_augments

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        rec = self.data[idx]['feature_arrays']
        l = self.data[idx]['len']

        def to_tensor(val, key):
            if FEATURES[key]['type'] == 'cat':
                return torch.LongTensor(val)
            else:
                return torch.FloatTensor(val)

        x = defaultdict(list)
        n = 4
        for _ in range(self.n_augments):
            idx = torch.randint(0, 2, (l,)).bool()
            idx[-n:] = False # don't take last 4 films they are for test

            for k in sorted(list(FEATURES.keys())):
                tmp = to_tensor(rec[k], k)
                tmp = tmp.masked_select(idx)
                x[k].append(tmp[-9:])

        # target
        y_idx = torch.zeros(l).bool()
        y_idx[-n:] = True
        tmp = to_tensor(rec['element_uid'], 'element_uid')
        lbls = tmp.masked_select(y_idx)

        true_lbls = lbls.expand(self.n_augments, lbls.size(0))
        rewards = torch.Tensor([-1]*n)
        if random.random() > ERROR_RATE:
            fake_labels = torch.randperm(FEATURES['element_uid']['in'])[:n]
            fake_labels = fake_labels.expand(self.n_augments, lbls.size(0))
            
            rewards = torch.Tensor([BAD_REWARD]*n)
        else:
            fake_labels = true_lbls
        rewards = rewards.expand(self.n_augments, lbls.size(0))

        return x, (true_lbls, fake_labels, rewards)

def get_criteo_data_loaders(batch_size, n_augments, augment_labels=False):
    x_train, y_train, x_test, y_test = criteo_control_datasets()
    if ADD_INFO['augment_type'] == 'rnn':
        print('use rnn dataset')
        data_train = CriteoMetricLearnDataSet(x_train, y_train, n_augments)
        data_test = CriteoMetricLearnDataSet(x_test, y_test, n_augments)

        train_data_loader = torch.utils.data.DataLoader(data_train,
                                            batch_size=batch_size,
                                            shuffle=True,
                                            num_workers=16,
                                            collate_fn=padded_collate_criteo_metrlearn)

        test_data_loader = torch.utils.data.DataLoader(data_test,
                                            batch_size=batch_size,
                                            shuffle=True,
                                            num_workers=4,
                                            collate_fn=padded_collate_criteo_metrlearn)
    else:
        print('use perceptron dataset')
        data_train = CriteoMetricLearnDataSet2(x_train, y_train, n_augments)
        data_test = CriteoMetricLearnDataSet2(x_test, y_test, n_augments)

        train_data_loader = torch.utils.data.DataLoader(data_train,
                                            batch_size=batch_size,
                                            shuffle=True,
                                            num_workers=16)

        test_data_loader = torch.utils.data.DataLoader(data_test,
                                            batch_size=batch_size,
                                            shuffle=True,
                                            num_workers=4)
    
    return train_data_loader, test_data_loader

def get_criteo_domyshnik_data_loaders(batch_size, n_augments, augment_labels=False):
    x_train, y_train, x_test, y_test = criteo_control_datasets()
    if ADD_INFO['augment_type'] == 'rnn':
        print('use rnn dataset')
        data_train = CriteoMetricLearnDataSet(x_train, y_train, n_augments, augment_labels)
        data_test = CriteoMetricLearnDataSet(x_test, y_test, n_augments, augment_labels)

        train_data_loader = torch.utils.data.DataLoader(data_train,
                                            batch_size=batch_size,
                                            shuffle=True,
                                            num_workers=16,
                                            collate_fn=padded_collate_criteo_rnn_domyshnik)

        test_data_loader = torch.utils.data.DataLoader(data_test,
                                            batch_size=batch_size,
                                            shuffle=True,
                                            num_workers=4,
                                            collate_fn=padded_collate_criteo_rnn_domyshnik)
    else:
        print('use perceptron dataset')
        data_train = CriteoMetricLearnDataSet2(x_train, y_train, n_augments, augment_labels)
        data_test = CriteoMetricLearnDataSet2(x_test, y_test, n_augments, augment_labels)

        train_data_loader = torch.utils.data.DataLoader(data_train,
                                            batch_size=batch_size,
                                            shuffle=True,
                                            num_workers=16)

        test_data_loader = torch.utils.data.DataLoader(data_test,
                                            batch_size=batch_size,
                                            shuffle=True,
                                            num_workers=4)
    
    return train_data_loader, test_data_loader

def get_mnist_train_loader(batch_size, n_augments=4, augment_labels=False):
    data_train = MetrLearnDataset(dataset=mnist_train_dataset(), 
                            augmenter=mnist_torch_augmentation(p=1), 
                            n_augments=n_augments,
                            augment_labels=augment_labels)
    
    train_data_loader = torch.utils.data.DataLoader(data_train,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          num_workers=16)

    test_data_loader = torch.utils.data.DataLoader(data_test,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              num_workers=5)
    return train_data_loader, test_data_loader
    
def get_mnist_test_loader(batch_size, n_augments=4, augment_labels=False):
    data_test = MetrLearnDataset(dataset=mnist_test_dataset(), 
                            augmenter=mnist_torch_augmentation(p=1), 
                            n_augments=n_augments,
                            augment_labels=augment_labels)
        
    test_data_loader = torch.utils.data.DataLoader(data_test,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              num_workers=5)
    return test_data_loader

def get_cifar10_train_loader(batch_size, n_augments=4, augment_labels=False):
    data_train = MetrLearnDataset(dataset=cifar10_train_dataset(), 
                            augmenter=cifar_torch_augmentation(p=1), 
                            n_augments=n_augments,
                            augment_labels=augment_labels)
    
    train_data_loader = torch.utils.data.DataLoader(data_train,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          num_workers=1)
    return train_data_loader
    
def get_cifar10_test_loader(batch_size, n_augments=4, augment_labels=False):
    data_test = MetrLearnDataset(dataset=cifar10_test_dataset(), 
                            augmenter=cifar_torch_augmentation(p=1), 
                            n_augments=n_augments,
                            augment_labels=augment_labels)
        
    test_data_loader = torch.utils.data.DataLoader(data_test,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              num_workers=5)
    return test_data_loader

def get_cifar10_test_loader_without_augmentation(batch_size=1):
    data_test = MetrLearnDataset(dataset=cifar10_test_dataset(), 
                            augmenter=transforms.ToTensor(), 
                            n_augments=0,
                            augment_labels=False)
        
    test_data_loader = torch.utils.data.DataLoader(data_test,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              num_workers=5)
    return test_data_loader

def get_cifar10_train_global_loader(batch_size, centroids, n_augments_centroids, n_augments_images):
    data_train = MetrLearnDataset(dataset=cifar10_train_dataset(), 
                                  augmenter=cifar_torch_augmentation(p=1), 
                                  n_augments=n_augments_images - 1,
                                  augment_labels=False)
    
    base_train_data_loader = torch.utils.data.DataLoader(data_train,
                                                         batch_size=batch_size,
                                                         shuffle=True,
                                                         num_workers=16)

    train_loader = DataLoaderWrapper(base_loader=base_train_data_loader,
                                     centroids=centroids,
                                     n_augments=n_augments_centroids,
                                     augmenter=cifar_torch_augmentation(p=1))

    return train_loader
    
def get_cifar10_test_global_loader(batch_size, centroids):
    data_test = MetrLearnDataset(dataset=cifar10_test_dataset(), 
                                  augmenter=cifar_torch_augmentation(p=1), 
                                  n_augments=-1,
                                  augment_labels=False)
    
    base_test_data_loader = torch.utils.data.DataLoader(data_test,
                                                         batch_size=batch_size,
                                                         shuffle=False,
                                                         num_workers=5)

    test_loader = DataLoaderWrapper(base_loader=base_test_data_loader,
                                     centroids=centroids,
                                     n_augments=0)

    return test_loader

def get_cifar10_centroids(count, loader):
    b_count =  int(float(count)/(len(loader))) # number of samples to take from batch
    centroids = []
    for imgs, _ in loader:

        b_rate = min(float(b_count)/imgs.size(0), 1.0)
        mask = torch.bernoulli(b_rate * torch.ones(imgs.size(0))).bool()

        b_centroids = imgs[mask].contiguous()
        centroids.append(b_centroids)

    centroids = torch.cat(centroids, 0)
    return centroids

def get_okko_metrlearn_loaders(batch_size, n_augments):
    train_data, test_data = okko_metriclearn_datasets(n_augments=n_augments)

    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=16,
        collate_fn=padded_collate_okko_metrlearn)

    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=padded_collate_okko_metrlearn)

    return train_loader, test_loader

def get_okko_domyshnik_loaders(batch_size, n_augments):
    train_data, test_data = okko_domyshnik_datasets(n_augments=n_augments)

    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=16,
        collate_fn=padded_collate_okko_domyshnik)

    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=padded_collate_okko_domyshnik)

    return train_loader, test_loader