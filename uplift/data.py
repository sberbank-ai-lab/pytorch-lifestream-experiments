import torch
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import random

from constants import *


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
            imgs = [self.base_aug(img)] + [self.aug(img) for i in range(self.n_augments)]
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