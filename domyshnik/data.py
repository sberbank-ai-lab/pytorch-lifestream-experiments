import torch
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np


mnist_train_dataset = torchvision.datasets.MNIST('/mnt/data/molchanov/datasets/mnist', 
                                           train=True, 
                                           transform=None, 
                                           target_transform=None, 
                                           download=True)

mnist_test_dataset = torchvision.datasets.MNIST('/mnt/data/molchanov/datasets/mnist', 
                                           train=False, 
                                           transform=None, 
                                           target_transform=None, 
                                           download=True)

# -------------------------------------------------------------------------------------------------

def mnist_torch_augmentation(p=1):
    return torchvision.transforms.Compose([
        transforms.RandomChoice([
            transforms.RandomAffine(degrees=7, 
                                translate=(0.1, 0.1), 
                                scale=(0.9, 0.9), 
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
    
    def __init__(self, dataset, augmenter, n_augments=10):
        self.data = dataset
        self.aug = augmenter
        self.n_augments = n_augments
        
    def draw(self, idx):
        imgs, lbl = self[idx]
        print(imgs.shape, lbl)
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
        return b_img, lbl
        
    
def get_mnist_train_loader(batch_size, n_augments=4):
    data_train = MetrLearnDataset(dataset=mnist_train_dataset, 
                            augmenter=mnist_torch_augmentation(p=1), 
                            n_augments=n_augments)
    
    train_data_loader = torch.utils.data.DataLoader(data_train,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          num_workers=16)
    return train_data_loader
    
def get_mnist_test_loader(batch_size, n_augments=4):
    data_test = MetrLearnDataset(dataset=mnist_test_dataset, 
                            augmenter=mnist_torch_augmentation(p=1), 
                            n_augments=n_augments)
        
    test_data_loader = torch.utils.data.DataLoader(data_test,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              num_workers=5)
    return test_data_loader