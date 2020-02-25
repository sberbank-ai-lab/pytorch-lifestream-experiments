import torch
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import random

from domyshnik.constants import *


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

def cifar_torch_augmentation(p=1):
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

# -------------------------------------------------------------------------------------------------
class MetrLearnDataset(torch.utils.data.Dataset):
    
    def __init__(self, dataset, augmenter, n_augments=10, augment_labels=False):
        self.data = dataset
        self.aug = augmenter
        self.n_augments = n_augments
        self.augment_labels = augment_labels
        
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
        imgs = [transforms.ToTensor()(img)] + [self.aug(img) for i in range(self.n_augments)]
        b_img = torch.stack(imgs).squeeze()
        if not self.augment_labels:
            return b_img, lbl
        else:
            new_lbl = random.randint(0, NUM_CLASSES - 1)
            reward = -1.0 if lbl == new_lbl else -0.8
            return b_img, (new_lbl, lbl, reward)
        
    
def get_mnist_train_loader(batch_size, n_augments=4, augment_labels=False):
    data_train = MetrLearnDataset(dataset=mnist_train_dataset(), 
                            augmenter=mnist_torch_augmentation(p=1), 
                            n_augments=n_augments,
                            augment_labels=augment_labels)
    
    train_data_loader = torch.utils.data.DataLoader(data_train,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          num_workers=16)
    return train_data_loader
    
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
                                          num_workers=16)
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