import sys
sys.path.insert(0, "/mnt/data/molchanov/dltranz")


import torch
import torch.nn.functional as F
import tqdm
import numpy as np
from domyshnik.utils import *
from domyshnik.constants import *
from dltranz.metric_learn.metric import metric_Recall_top_K

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
        
        
    def train_epoch(self, step):
        self.model.train()
        losses = []
        with tqdm.tqdm(total=len(self.train_loader)) as steps:
            for itr, data in enumerate(self.train_loader):
                self.optimizer.zero_grad()
                
                device_data = self.data_to_device(data)
                labels = device_data[1]
                
                out = self.model(device_data[0])
                if self.mode == 'metric_learning':
                    #labels = torch.arange(int(out.size(0)/(N_AUGMENTS + 1))).view(1, -1).repeat(N_AUGMENTS+1, 1).transpose(0, 1).flatten().to(self.device)
                    labels = labels.view(1, -1).repeat(N_AUGMENTS+1, 1).transpose(0, 1).flatten().to(self.device)
                
                loss = self.loss(out, labels)
                if self.mode == 'metric_learning':
                    loss, pos_neg_len = loss[0], loss[1]
                losses.append(loss.item())
                loss_val = self.running_average(losses)
                loss.backward()
                
                self.optimizer.step()
                
                steps.set_description(f"train: epoch {step}, step {itr}/{len(self.train_loader)}")
                if self.mode == 'classification':
                    steps.set_postfix({"loss": '{:.5E}'.format(loss_val)})
                elif self.mode == 'metric_learning':
                    steps.set_postfix({"loss": '{:.5E}'.format(loss_val)})
                steps.update()
        
    def test_epoch(self, step):
        self.model.eval()
        losses, total, corrects = [], 0, 0
        total_recall = 0
        with torch.no_grad():
            with tqdm.tqdm(total=len(self.test_loader)) as steps:
                for itr, data in enumerate(self.test_loader):
                    device_data = self.data_to_device(data)
                    labels = device_data[1]
                    
                    out = self.model(device_data[0])
                    if self.mode == 'metric_learning':
                        #labels = torch.arange(int(out.size(0)/(N_AUGMENTS + 1))).view(1, -1).repeat(N_AUGMENTS+1, 1).transpose(0, 1).flatten().to(self.device)
                        labels = labels.view(1, -1).repeat(N_AUGMENTS+1, 1).transpose(0, 1).flatten().to(self.device)
                
                    loss = self.loss(out, labels)
                    if self.mode == 'metric_learning':
                        loss, pos_neg_len = loss[0], loss[1]
                    losses.append(loss.item())
                    loss_val = self.running_average(losses)
                    
                    if self.mode == 'classification':
                        pred = F.softmax(out, dim=-1).argmax(dim=-1)
                        corrects += pred.eq(device_data[1].view_as(pred)).sum().item()
                        total += pred.size(0)
                        accuracy = corrects/total
                    elif self.mode == 'metric_learning':
                        batch_recall = metric_Recall_top_K(out, labels, N_AUGMENTS*int(BATCH_SIZE/10))
                        total_recall += batch_recall
                    
                    steps.set_description(f"test: epoch {step}, step {itr}/{len(self.train_loader)}")
                    if self.mode == 'classification':
                        steps.set_postfix({"loss": '{:.5E}'.format(loss_val), "accuracy": accuracy})
                    elif self.mode == 'metric_learning':
                        steps.set_postfix({"loss": '{:.5E}'.format(loss_val),
                                           "RECALL": total_recall/(itr+1)})
                    steps.update()
        
    def fit(self):
        for step in range(self.epochs):
            self.train_epoch(step + 1)
            self.test_epoch(step + 1)
            self.scheduler.step()
        if SAVE_MODELS:
            save_model_params(self.model, self.model_name)
            
    def data_to_device(self, data):
        return data[0].to(self.device).float(), data[1].to(self.device)
    
    def running_average(self, x):
        arr = np.array(x)[-1000:]
        res = np.convolve(arr, np.ones((arr.shape[0],)))/arr.shape[0]
        return res[0]


def main():
    mnist_metriclearning_learner = Learner(launch_info=mnist_metriclearning_lunch_info)
    mnist_metriclearning_learner.fit()

main()