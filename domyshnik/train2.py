import sys
sys.path.insert(0, "/mnt/data/molchanov/dltranz")

import torch
import torch.nn.functional as F
import tqdm
import numpy as np
from domyshnik.utils import *
from domyshnik.constants import *
from dltranz.metric_learn.metric import metric_Recall_top_K
import copy



class AverageLoss:

    def __init__(self, window=1000):
        self.history = []
        self.window = window

    def update(self, new_val):
        if not isinstance(new_val, tuple):
            self.history.append(new_val.item())
            arr = np.array(self.history)[-self.window:]
            res = np.convolve(arr, np.ones((arr.shape[0],)))/arr.shape[0]
            return res[0]
        else:
            if len(self.history) == 0:
                for _ in new_val:
                    self.history.append([])
            outs = []
            for val, hist in zip(new_val, self.history):
                hist.append(val.item())
                arr = np.array(hist)[-self.window:]
                res = np.convolve(arr, np.ones((arr.shape[0],)))/arr.shape[0]
                outs.append(res[0])
            return tuple(outs)
            


class LossesCalculator:

    def __init__(self):
        self.losses_stat = {}   
        self.losses = {}

    def update_loss(self, loss_name, new_val):
        return self.losses_stat[loss_name].update(new_val)

    def add_loss(self, loss, loss_name):
        self.losses_stat[loss_name] = AverageLoss()
        self.losses[loss_name] = loss

    def get_loss(self, loss_name):
        return self.losses[loss_name]


class Learner:
    
    def __init__(self, launch_info):
        self.info = launch_info
        
        self.model = self.info.model
        self.loss = self.info.loss
        self.optimizer = self.info.optimizer
        self.scheduler = self.info.scheduler
        self.epochs = self.info.epochs
        self.train_loader = self.info.train_loader
        self.test_loader = self.info.test_loader
        self.device = self.info.device
        self.mode = self.info.mode
        self.model_name = self.info.model_name
        self.add_info = self.info.add_info

        self.los_calc = LossesCalculator()
        for loss, loss_name in self.add_info['losses']:
            self.los_calc.add_loss(loss.to(self.device), loss_name)

    def update_loss(self, loss_name, val):
        return self.los_calc.update_loss(loss_name, val)

    def get_loss(self, loss_name):
        return self.los_calc.get_loss(loss_name)

    def train_metric_learning_global(self, data, itr, step):
        (imgs, lbls), (c_imgs, c_lbls) = data
        imgs = imgs.to(self.device)
        lbls = lbls.to(self.device)
        c_imgs = c_imgs.to(self.device)
        c_lbls = c_lbls.to(self.device)

        embs, c_embs = self.model(imgs), self.model(c_imgs)

        # local metric learning loss on centroids
        clust_los_pos, clust_los_neg = self.get_loss('ContrastiveLossOriginal_centroids')(c_embs, c_lbls)
        clust_los_pos_val, clust_los_neg_val = self.update_loss('ContrastiveLossOriginal_centroids', (clust_los_pos, clust_los_neg))

        # other images loss relating to centroids
        img_los = self.get_loss('InClusterisationLoss')(embs, c_embs.detach()) # centroids independent from other images
        img_los_val = self.update_loss('InClusterisationLoss', img_los)

        # centroids recall
        centroids_recal = metric_Recall_top_K(c_embs, c_lbls, N_AUGMENTS)

        loss = clust_los_pos + clust_los_neg + img_los

        return {
            'clust_los_pos': clust_los_pos_val,
            'clust_los_neg': clust_los_neg_val,
            'img2cntr'     : img_los_val,
            'centr_recal': centroids_recal
        }, loss

    def test_metric_learning_global(self, data, itr, step):
        return ""
    
    def train_epoch(self, step):
        self.model.train()
        with tqdm.tqdm(total=len(self.train_loader)) as steps:

            for itr, data in enumerate(self.train_loader):
                self.optimizer.zero_grad()

                if CURRENT_PARAMS == 'cifar10_metric_learning_global':
                    message, loss = self.train_metric_learning_global(data, itr, step)

                loss.backward()
                self.optimizer.step()

                steps.set_postfix(message)
                steps.update()
        
    def test_epoch(self, step):
        self.model.eval()
        with torch.no_grad():
            with tqdm.tqdm(total=len(self.test_loader)) as steps:

                for itr, data in enumerate(self.test_loader):

                    if CURRENT_PARAMS == 'cifar10_metric_learning_global':
                        message = self.test_metric_learning_global(data, itr, step)

                    steps.set_postfix(message)
                    steps.update()
        
    def fit(self):
        for step in range(self.epochs):
            self.train_epoch(step + 1)
            self.test_epoch(step + 1)
            self.scheduler.step()


def main():
    learner = Learner(launch_info=get_launch_info())
    learner.fit()

main()