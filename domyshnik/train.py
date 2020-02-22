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
        losses, rewrds = [], []
        losses_pos, losses_neg = [], []
        total_recall, total, corrects = 0, 0, 0
        with tqdm.tqdm(total=len(self.train_loader)) as steps:
            for itr, data in enumerate(self.train_loader):
                self.optimizer.zero_grad()
                device_data = self.data_to_device(data)
                labels = device_data[1]

                if self.mode == 'classification':
                    out = self.model(device_data[0])

                    loss = self.loss(out, labels)
                    losses.append(loss.item())
                    loss_val = self.running_average(losses)                    

                    pred = F.softmax(out, dim=-1).argmax(dim=-1)
                    corrects += pred.eq(old_labels.view_as(pred)).sum().item()
                    total += pred.size(0)
                    accuracy = corrects/total

                    steps.set_postfix({"loss": '{:.5E}'.format(loss_val),
                                    "accuracy": '{:.5E}'.format(accuracy)})

                elif self.mode == 'metric_learning': 
                    #labels = torch.arange(int(out.size(0)/(N_AUGMENTS + 1))).view(1, -1).repeat(N_AUGMENTS+1, 1).transpose(0, 1).flatten().to(self.device)
                    labels = labels.view(1, -1).repeat(N_AUGMENTS+1, 1).transpose(0, 1).flatten().to(self.device)

                    out = self.model(device_data[0])

                    loss = self.loss(out, labels)
                    loss, _ = loss[0], loss[1]
                    losses.append(loss.item())
                    loss_val = self.running_average(losses)

                    batch_recall = metric_Recall_top_K(out, labels, N_AUGMENTS*int(BATCH_SIZE/10))
                    total_recall += batch_recall

                    steps.set_postfix({"loss": '{:.5E}'.format(loss_val),
                                       "recall": '{:.5E}'.format(total_recall)})

                elif self.mode == 'domyshnik':
                    out = self.model(device_data[0])

                    labels, old_labels, rewards = labels[0], labels[1], labels[2]
                    old_labels = old_labels.view(1, -1).repeat(N_AUGMENTS+1, 1).transpose(0, 1).flatten().to(self.device)
                    rewards = rewards.view(1, -1).repeat(N_AUGMENTS+1, 1).transpose(0, 1).flatten().to(self.device)
                    p0_labels = labels.view(1, -1).repeat(N_AUGMENTS+1, 1).transpose(0, 1).flatten().to(self.device)
                    labels = torch.arange(int(out.size(0)/(N_AUGMENTS + 1))).view(1, -1).repeat(N_AUGMENTS+1, 1).transpose(0, 1).flatten().to(self.device)

                    loss_pos, loss_neg = self.loss(F.softmax(out, dim=-1), labels)
                    losses_pos.append(loss_pos.item())
                    losses_neg.append(loss_neg.item())
                    loss_pos_val = 0#self.running_average(losses_pos)
                    loss_neg_val = 0#self.running_average(losses_neg)

                    reward = ((F.softmax(out, dim=-1)[torch.arange(p0_labels.size(0)), p0_labels]) * rewards).sum()/p0_labels.size(0)
                    rewrds.append(reward)
                    reward_val = 0#self.running_average(rewrds)

                    pred = F.softmax(out, dim=-1).argmax(dim=-1)
                    corrects += pred.eq(old_labels.view_as(pred)).sum().item()
                    total += pred.size(0)
                    accuracy = corrects/total

                    loss = loss_pos + 0*loss_neg + 10000*reward

                    steps.set_postfix({"local loss pos": '{:.5E}'.format(loss_pos_val),
                                       "local loss neg": '{:.5E}'.format(loss_neg_val),
                                       "reward loss": '{:.5E}'.format(reward_val),
                                       "accuracy": '{:.5E}'.format(accuracy)})

                steps.set_description(f"train: epoch {step}, step {itr}/{len(self.train_loader)}")
                loss.backward()
                self.optimizer.step()
                steps.update()
        
    def test_epoch(self, step):
        self.model.eval()
        losses, rewrds = [], []
        losses_pos, losses_neg = [], []
        total_recall, total, corrects = 0, 0, 0
        with torch.no_grad():
            with tqdm.tqdm(total=len(self.test_loader)) as steps:
                for itr, data in enumerate(self.test_loader):
                    device_data = self.data_to_device(data)
                    labels = device_data[1]

                    if self.mode == 'classification':
                        out = self.model(device_data[0])

                        loss = self.loss(out, labels)
                        losses.append(loss.item())
                        loss_val = self.running_average(losses)

                        pred = F.softmax(out, dim=-1).argmax(dim=-1)
                        corrects += pred.eq(old_labels.view_as(pred)).sum().item()
                        total += pred.size(0)
                        accuracy = corrects/total

                        steps.set_postfix({"loss": '{:.5E}'.format(loss_val),
                                        "accuracy": '{:.5E}'.format(accuracy)})

                    elif self.mode == 'metric_learning': 
                        #labels = torch.arange(int(out.size(0)/(N_AUGMENTS + 1))).view(1, -1).repeat(N_AUGMENTS+1, 1).transpose(0, 1).flatten().to(self.device)
                        labels = labels.view(1, -1).repeat(N_AUGMENTS+1, 1).transpose(0, 1).flatten().to(self.device)

                        out = self.model(device_data[0])

                        loss = self.loss(out, labels)
                        loss, _ = loss[0], loss[1]
                        losses.append(loss.item())
                        loss_val = self.running_average(losses)

                        batch_recall = metric_Recall_top_K(out, labels, N_AUGMENTS*int(BATCH_SIZE/10))
                        total_recall += batch_recall

                        steps.set_postfix({"loss": '{:.5E}'.format(loss_val),
                                        "recall": '{:.5E}'.format(total_recall)})

                    elif self.mode == 'domyshnik':
                        out = self.model(device_data[0])

                        labels, old_labels, rewards = labels[0], labels[1], labels[2]
                        old_labels = old_labels.view(1, -1).repeat(N_AUGMENTS+1, 1).transpose(0, 1).flatten().to(self.device)
                        rewards = rewards.view(1, -1).repeat(N_AUGMENTS+1, 1).transpose(0, 1).flatten().to(self.device)
                        p0_labels = labels.view(1, -1).repeat(N_AUGMENTS+1, 1).transpose(0, 1).flatten().to(self.device)
                        labels = torch.arange(int(out.size(0)/(N_AUGMENTS + 1))).view(1, -1).repeat(N_AUGMENTS+1, 1).transpose(0, 1).flatten().to(self.device)

                        loss_pos, loss_neg = self.loss(F.softmax(out, dim=-1), labels)
                        losses_pos.append(loss_pos.item())
                        losses_neg.append(loss_neg.item())
                        loss_pos_val = 0#self.running_average(losses_pos)
                        loss_neg_val = 0#self.running_average(losses_neg)

                        reward = ((F.softmax(out, dim=-1)[torch.arange(p0_labels.size(0)), p0_labels]) * rewards).sum()/p0_labels.size(0)
                        rewrds.append(reward)
                        reward_val = 0#self.running_average(rewrds)

                        pred = F.softmax(out, dim=-1).argmax(dim=-1)
                        corrects += pred.eq(old_labels.view_as(pred)).sum().item()
                        total += pred.size(0)
                        accuracy = corrects/total

                        loss = loss_pos + loss_neg + reward

                        steps.set_postfix({"local loss pos": '{:.5E}'.format(loss_pos_val),
                                        "local loss neg": '{:.5E}'.format(loss_neg_val),
                                        "reward loss": '{:.5E}'.format(reward_val),
                                        "accuracy": '{:.5E}'.format(accuracy)})

                    steps.set_description(f"train: epoch {step}, step {itr}/{len(self.train_loader)}")
                    steps.update()
        
    def fit(self):
        for step in range(self.epochs):
            self.train_epoch(step + 1)
            self.test_epoch(step + 1)
            self.scheduler.step()
        if SAVE_MODELS:
            save_model_params(self.model, self.model_name)
            
    def data_to_device(self, data):
        if self.mode != 'domyshnik':
            return data[0].to(self.device).float(), data[1].to(self.device)
        else:
            return data[0].to(self.device).float(), (data[1][0].to(self.device),
                                                     data[1][1].to(self.device),
                                                     data[1][2].to(self.device))

    
    def running_average(self, x):
        arr = np.array(x)[-1000:]
        res = np.convolve(arr, np.ones((arr.shape[0],)))/arr.shape[0]
        return res[0]


def main():
    #mnist_classification_metriclearning_learner = Learner(launch_info=mnist_classification_metriclearning_lunch_info)
    #mnist_classification_metriclearning_learner.fit()

    #mnist_metriclearning_learner = Learner(launch_info=mnist_metriclearning_lunch_info)
    #mnist_metriclearning_learner.fit()

    mnist_domyshnik_learner = Learner(launch_info=mnist_domyshnik_lunch_info)
    mnist_domyshnik_learner.fit()

main()