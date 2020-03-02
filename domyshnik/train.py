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

'''

'''
class Critic:

    '''
    num_samples = len(train_loader)
    entropy_update_treshold - we want update only when new strategy distribution - less uncertain in comprasion with old one
    '''
    def __init__(self, num_samples, device, entropy_update_treshold=None):
        self.rewards = torch.zeros(num_samples, NUM_CLASSES).fill_(BAD_REWARD)
        self.model = None
        self.p0 = torch.zeros(num_samples, NUM_CLASSES).fill_(0.1)
        self.device = device
        self.entrp_tresh = entropy_update_treshold
        if self.entrp_tresh is None:
            e1 = self.entropy(torch.Tensor([1 / NUM_CLASSES] * NUM_CLASSES))
            e2 = self.entropy(torch.Tensor([1 / (NUM_CLASSES-1)] * (NUM_CLASSES - 1)))
            self.entrp_tresh = (e1 - e2)/2

    def entropy(self, distrib):
        d = -distrib*torch.log(distrib)
        d[torch.isnan(d)] = 0
        return d.sum(-1)

    '''
    correct_labels - good labels (used only for first p0 strategy)
    contexts - batch from dataloader
    idxs - indexes of concrete samples from dataloader
    '''
    def select_action(self, contexts, correct_labels, idxs):
        
        if self.model is None:

            # p0 uncertainty strategy
            new_lbls = torch.randint(0, NUM_CLASSES, (len(correct_labels),))
            rewards = torch.zeros(len(correct_labels)).fill_(-0.8)
            rewards[rewards == correct_labels] = -1.0

            # fill reward matrix
            self.rewards[idxs, new_lbls] = rewards 
            return new_lbls, rewards
        else:

            probs = F.softmax(self.model(contexts), dim=-1)
            new_lbls = probs.argmax(dim=-1)
            
            # update rewards according new strategy 
            # (when action probability becomes less we decrease reward and wise verse)
            p0_probs = self.p0[idxs]
            dlt = F.kl_div(input=F.log_softmax(p0_probs, dim=-1),
                           target=probs,
                           reduction='none')

            # for robastness - update when entropy becomes less (more stable strategy)
            entrp0, entrp = self.entropy(p0_probs), self.entropy(probs)
            entrp_mask = (entrp0 - entrp > self.entrp_tresh).int()

            self.rewards[idxs] -= dlt*entrp_mask # minus is important because rewards are negatives
            self.p0[idxs] = probs * entrp_mask + p0_probs * (1 - entrp_mask)

            return new_lbls.to(self.device), correct_labels, self.rewards[idxs].to(self.device)

    def update_strategy(self, model):
        self.model = copy.deepcopy(self.model)
        self.model.eval()


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
        self.copy_model = None
        self.lamba = 1.0
        self.best_accuracy = 0
        if self.add_info is not None and self.add_info.get('use_clusterisation_loss', False):
            self.clust_loss = ClusterisationLoss(margin=MARGING, 
                                                 input_dim=256, 
                                                 num_classes=NUM_CLASSES, 
                                                 device=self.device)
            self.clust_loss.to(self.device)

        
    def train_epoch(self, step):
        self.model.train()
        losses, rewrds = [], []
        losses_pos, losses_neg = [], []
        total_recall, total, corrects = 0, 0, 0
        clust_neg_losses, clust_pos_losses = [], []
        with tqdm.tqdm(total=len(self.train_loader)) as steps:
            for itr, data in enumerate(self.train_loader):
                self.optimizer.zero_grad()
                device_data = self.data_to_device(data)
                labels = device_data[1]
                out = self.model(device_data[0])

                if self.mode == 'classification':
                    loss = self.loss(out, labels)
                    losses.append(loss.item())
                    loss_val = self.running_average(losses)                    

                    pred = F.softmax(out, dim=-1).argmax(dim=-1)
                    corrects += pred.eq(labels.view_as(pred)).sum().item()
                    total += pred.size(0)
                    accuracy = corrects/total

                    steps.set_postfix({"loss": loss_val,
                                       "accuracy": accuracy})

                elif self.mode == 'metric_learning': 
                    if CURRENT_PARAMS in ['metric_learning_per_sampl', 'cifar10_metric_learning_per_sampl']:
                        labels = torch.arange(int(out.size(0)/(N_AUGMENTS + 1))).view(1, -1).repeat(N_AUGMENTS+1, 1).transpose(0, 1).flatten().to(self.device)
                        K = N_AUGMENTS
                    elif CURRENT_PARAMS in ['metric_learning_per_class', 'cifar10_metric_learning_per_class']:
                        K = N_AUGMENTS * int(BATCH_SIZE/10)
                        labels = labels.view(1, -1).repeat(N_AUGMENTS+1, 1).transpose(0, 1).flatten().to(self.device)
                    
                    loss_pos, loss_neg = self.loss(out, labels)
                    losses_pos.append(loss_pos.item())
                    losses_neg.append(loss_neg.item())
                    loss_pos_val = self.running_average(losses_pos)
                    loss_neg_val = self.running_average(losses_neg)

                    c_loss, c_pos_val, c_neg_val = 0, 0, 0
                    if self.clust_loss is not None and step > 4:

                        x = out.view(out.size(0), -1, out.size(-1))[:, 0, :]
                        c_pos, c_neg = self.clust_loss(x)

                        c_pos *= self.add_info['k_clust_pos']
                        clust_pos_losses.append(c_pos.item())
                        c_pos_val = self.running_average(clust_pos_losses)

                        c_neg *= self.add_info['k_clust_neg']
                        clust_neg_losses.append(c_neg.item())
                        c_neg_val = self.running_average(clust_neg_losses)
                        
                        c_loss = c_pos + c_neg

                    k_pos, k_neg = self.add_info['k_pos'], self.add_info['k_neg']

                    loss = k_pos * loss_pos + k_neg * loss_neg + c_loss

                    total_recall += metric_Recall_top_K(out, labels, K)

                    margings = self.loss.get_margings()
                    steps.set_postfix({"loss_pos": loss_pos_val * k_pos,
                                       "loss neg": loss_neg_val * k_neg,
                                       "clust pos loss": c_pos_val,
                                       "clust neg loss": c_neg_val,
                                       "recall": total_recall/(itr + 1),
                                       #"margings": margings
                                       })

                elif self.mode == 'domyshnik':

                    labels, old_labels, rewards = labels[0], labels[1], labels[2]
                    # refresh rewards according new strategy
                    if self.copy_model is not None:
                        x = device_data[0].view(-1, N_AUGMENTS + 1, device_data[0].size(-2), device_data[0].size(-1))[:, 0, :, :]
                        out_copy = self.copy_model(x)
                        labels = F.softmax(out_copy, dim=-1).argmax(dim=-1)
                        rewards = rewards.masked_fill(labels == old_labels, -1.0)
                    batch_res = (labels == old_labels).sum().float().item()/BATCH_SIZE


                    old_labels = old_labels.view(1, -1).repeat(N_AUGMENTS+1, 1).transpose(0, 1).flatten().to(self.device)
                    rewards = rewards.view(1, -1).repeat(N_AUGMENTS+1, 1).transpose(0, 1).flatten().to(self.device)
                    p0_labels = labels.view(1, -1).repeat(N_AUGMENTS+1, 1).transpose(0, 1).flatten().to(self.device)
                    labels = torch.arange(int(out.size(0)/(N_AUGMENTS + 1))).view(1, -1).repeat(N_AUGMENTS+1, 1).transpose(0, 1).flatten().to(self.device)

                    loss_pos, loss_neg = self.loss(F.softmax(out, dim=-1), labels)
                    losses_pos.append(loss_pos.item())
                    losses_neg.append(loss_neg.item())
                    loss_pos_val = self.running_average(losses_pos)
                    loss_neg_val = self.running_average(losses_neg)

                    reward = ((F.softmax(out, dim=-1)[torch.arange(p0_labels.size(0)), p0_labels]) * rewards).sum()#/p0_labels.size(0)
                    rewrds.append(reward.item())
                    reward_val = self.running_average(rewrds)

                    pred = F.softmax(out, dim=-1).argmax(dim=-1)
                    corrects += pred.eq(old_labels.view_as(pred)).sum().item()
                    total += pred.size(0)
                    accuracy = corrects/total

                    k_pos, k_neg, k_reward = self.add_info['k_pos'], self.add_info['k_neg'], self.add_info['k_reward']

                    loss = (k_pos * loss_pos + k_neg * loss_neg) * self.lamba + k_reward * reward


                    steps.set_postfix({"loss_pos": loss_pos_val * k_pos,
                                       "loss neg": loss_neg_val * k_neg,
                                       "reward loss": reward_val * k_reward,
                                       #"batch_res": batch_res,
                                       #"k": f'{k_pos}, {k_neg}, {k_reward}',
                                       "accuracy": accuracy})

                steps.set_description(f"train: epoch {step}, step {itr}/{len(self.train_loader)}")
                loss.backward()
                self.optimizer.step()
                steps.update()

            if self.mode == 'domyshnik' and self.add_info.get('factor', -1) != -1:
                self.add_info['k_pos'] /= self.add_info['factor']
                self.add_info['k_neg'] /= self.add_info['factor']
                self.add_info['k_reward'] *= self.add_info['factor']
                if step > 1:
                    self.model.allow_grads()
        
    def test_epoch(self, step):
        self.model.eval()
        losses, rewrds = [], []
        losses_pos, losses_neg = [], []
        total_recall, total, corrects = 0, 0, 0
        acc = 0
        with torch.no_grad():
            with tqdm.tqdm(total=len(self.test_loader)) as steps:
                for itr, data in enumerate(self.test_loader):
                    device_data = self.data_to_device(data)
                    labels = device_data[1]
                    out = self.model(device_data[0])

                    if self.mode == 'classification':
                        loss = self.loss(out, labels)
                        losses.append(loss.item())
                        loss_val = self.running_average(losses)

                        pred = F.softmax(out, dim=-1).argmax(dim=-1)
                        corrects += pred.eq(labels.view_as(pred)).sum().item()
                        total += pred.size(0)
                        accuracy = corrects/total
                        acc = accuracy

                        steps.set_postfix({"loss": loss_val,
                                           "accuracy": accuracy})

                    elif self.mode == 'metric_learning': 
                        if CURRENT_PARAMS in ['metric_learning_per_sampl', 'cifar10_metric_learning_per_sampl']:
                            labels = torch.arange(int(out.size(0)/(N_AUGMENTS + 1))).view(1, -1).repeat(N_AUGMENTS+1, 1).transpose(0, 1).flatten().to(self.device)
                            K = N_AUGMENTS
                        elif CURRENT_PARAMS in ['metric_learning_per_class', 'cifar10_metric_learning_per_class']:
                            K = N_AUGMENTS * int(BATCH_SIZE/10)
                            labels = labels.view(1, -1).repeat(N_AUGMENTS+1, 1).transpose(0, 1).flatten().to(self.device)

                        loss_pos, loss_neg = self.loss(out, labels)
                        losses_pos.append(loss_pos.item())
                        losses_neg.append(loss_neg.item())
                        loss_pos_val = self.running_average(losses_pos)
                        loss_neg_val = self.running_average(losses_neg)

                        loss = loss_pos + loss_neg

                        total_recall += metric_Recall_top_K(out, labels, K)
                        acc = total_recall / (itr + 1)

                        margings = self.loss.get_margings()
                        steps.set_postfix({"loss_pos": loss_pos_val,
                                        "loss neg": loss_neg_val,
                                        "recall": total_recall/(itr + 1)})

                    elif self.mode == 'domyshnik':
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
                        acc = accuracy

                        loss = loss_pos + loss_neg + reward

                        steps.set_postfix({
                                        #"local loss pos": '{:.5E}'.format(loss_pos_val),
                                        #"local loss neg": '{:.5E}'.format(loss_neg_val),
                                        #"reward loss": '{:.5E}'.format(reward_val),
                                        "accuracy": '{:.5E}'.format(accuracy)})

                    steps.set_description(f"test: epoch {step}, step {itr}/{len(self.test_loader)}")
                    steps.update()
                #self.loss.step()
                #self.loss.step(gamma_pos=0.85, gamma_neg=0.978)
                if self.best_accuracy < acc:
                    self.best_accuracy = acc
                    if SAVE_MODELS:
                        save_model_params(self.model, self.model_name)

        
    def fit(self):
        for step in range(self.epochs):
            self.train_epoch(step + 1)
            self.test_epoch(step + 1)
            self.scheduler.step()

            if self.add_info is not None and self.add_info.get('refresh_reward_step', False):
                refresh_reward_step = self.add_info['refresh_reward_step']
                if step % refresh_reward_step == refresh_reward_step -1:
                    self.copy_model = copy.deepcopy(self.model)
                    self.copy_model.eval()
                    #self.lamba *= 0.7
                    print('refresh rewards')

        #if SAVE_MODELS:
        #    save_model_params(self.model, self.model_name)
            
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
    learner = Learner(launch_info=get_launch_info())
    learner.fit()

main()