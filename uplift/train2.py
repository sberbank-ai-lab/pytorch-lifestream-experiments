import sys
sys.path.insert(0, "/mnt/data/molchanov/dltranz")

import torch
import torch.nn.functional as F
import tqdm
import numpy as np
from uplift.utils import *
from uplift.constants import *
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
            res = arr.sum()/arr.shape[0]
            return res
            #res = np.convolve(arr, np.ones((arr.shape[0],)))/arr.shape[0]
            #return res[0]
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

    def remove_loss(self, loss_name):
        self.losses.pop(loss_name, None)
        self.losses_stat.pop(loss_name, None)


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

    def add_loss(self, loss, loss_name):
        if loss is not None:
            self.los_calc.add_loss(loss.to(self.device), loss_name)
        else:
            self.los_calc.add_loss(lambda x: x.item(), loss_name)

    def remove_loss(self, loss_name):
        self.los_calc.remove_loss(loss_name)

    def get_loss(self, loss_name):
        return self.los_calc.get_loss(loss_name)

    def split_augmented_embeddings(self, embs, augs_count):
        tmp = embs.view(-1, augs_count, embs.size(-1))
        idx = torch.arange(tmp.size(0))

        mask1 = torch.randint(0, 2, (tmp.size(0),)).bool()
        idx1 = idx.masked_select(mask1).to(self.device)
        idx2 = idx.masked_select(mask1.logical_not()).to(self.device)

        embs_for_centroids = tmp.index_select(0, idx1).view(-1, tmp.size(-1))
        embs_for_self_simular = tmp.index_select(0, idx2).view(-1, tmp.size(-1))

        embs_for_self_simular_lbls = torch.arange(idx2.size(0)).view(1, -1).repeat(augs_count, 1).transpose(0, 1).flatten().to(self.device)

        return embs_for_centroids, embs_for_self_simular, embs_for_self_simular_lbls

    def split_augmented_embeddings2(self, embs, augs_count):
        tmp = embs.view(-1, augs_count, embs.size(-1))
        idx = torch.zeros(1).to(self.device).long()

        embs_for_centroids = tmp.index_select(1, idx).squeeze()
        embs_for_self_simular = embs
        embs_for_self_simular_lbls = torch.arange(tmp.size(0)).view(1, -1).repeat(augs_count, 1).transpose(0, 1).flatten().to(self.device)

        return embs_for_centroids, embs_for_self_simular, embs_for_self_simular_lbls
    
    def train_metric_learning_global(self, data, itr, step):
        (imgs, lbls), (c_imgs, c_lbls) = data
        imgs = imgs.to(self.device)
        lbls = lbls.to(self.device)
        c_imgs = c_imgs.to(self.device)
        c_lbls = c_lbls.to(self.device)

        if step == 1:
            self.add_loss(None, 'centroids_radius')
            self.add_loss(None, 'centroids_elements_count')

        embs, c_embs = self.model(imgs), self.model(c_imgs)
        n_augs_imgs = self.add_info['n_augs_imgs']
        if n_augs_imgs > 1:
            embs_for_centroids, embs_for_self_simular, embs_for_self_simular_lbls = self.split_augmented_embeddings2(embs, n_augs_imgs)
        else:
            embs_for_centroids, embs_for_self_simular, embs_for_self_simular_lbls = embs, None, None

        tmp = c_embs.view(-1, N_AUGMENTS, c_embs.size(-1))
        centroids = tmp.index_select(1, torch.Tensor([0]).long().to(self.device)).squeeze()

        # local metric learning loss on centroids
        clust_los_pos, clust_los_neg = self.get_loss('ContrastiveLossOriginal_centroids')(c_embs, c_lbls)
        clust_los_pos_val, clust_los_neg_val = self.update_loss('ContrastiveLossOriginal_centroids', (clust_los_pos, clust_los_neg))

        # local metric learning loss on images
        imgs_los_pos_val, imgs_los_neg_val, imgs_recal = -1, -1, -1
        if embs_for_self_simular is not None:
            imgs_los_pos, imgs_los_neg = self.get_loss('ContrastiveLossOriginal_images')(embs_for_self_simular, embs_for_self_simular_lbls)
            imgs_los_pos_val, imgs_los_neg_val = self.update_loss('ContrastiveLossOriginal_images', (imgs_los_pos, imgs_los_neg))
            imgs_recal = metric_Recall_top_K(embs_for_self_simular, embs_for_self_simular_lbls, n_augs_imgs)

        # other images loss relating to centroids
        img_2_cents_los, centroids_radius, centroids_elements_count = self.get_loss('InClusterisationLoss')(embs_for_centroids, centroids.detach()) # centroids independent from other images
        img_2_cents_los_val = self.update_loss('InClusterisationLoss', img_2_cents_los)
        k_img_los = 1 if step % 3 == 2 else 0
        k_other_los = 1 - k_img_los

        centr_radius = self.update_loss('centroids_radius', centroids_radius)
        centr_elements_count = self.update_loss('centroids_elements_count', centroids_elements_count)

        # centroids recall
        centroids_recal = metric_Recall_top_K(c_embs, c_lbls, N_AUGMENTS)

        loss = (clust_los_pos + clust_los_neg) * k_other_los + \
               img_2_cents_los * k_img_los + \
               (imgs_los_pos + imgs_los_neg) * k_other_los     

        return {
            'Im2Clst': img_2_cents_los_val * k_img_los,
            'Im_REC': imgs_recal,
            'Clst_REC': centroids_recal,

            'Clst_pos': clust_los_pos_val * k_other_los,
            'Clst_neg': clust_los_neg_val * k_other_los,
            

            'Im_pos': imgs_los_pos_val * k_other_los,
            'Im_neg': imgs_los_neg_val * k_other_los,
            

            'clst_rad': centr_radius,
            'clst_size': centr_elements_count
        }, loss, centroids

    def train_metric_learning_global_basis(self, data, itr, step):
        (imgs, lbls), (c_imgs, c_lbls) = data
        imgs = imgs.to(self.device)
        lbls = lbls.to(self.device)
        c_imgs = c_imgs.to(self.device)
        c_lbls = c_lbls.to(self.device)
        n_augs_imgs = self.add_info['n_augs_imgs']

        embs, c_embs = self.model(imgs), self.model(c_imgs)

        # local metric learning loss on centroids
        clust_los_pos, clust_los_neg = self.get_loss('ContrastiveLossOriginal_centroids')(c_embs, c_lbls)
        clust_los_pos_val, clust_los_neg_val = self.update_loss('ContrastiveLossOriginal_centroids', (clust_los_pos, clust_los_neg))

        # cosine cluster basis loss
        entropy, basis_embs = self.get_loss('BasisClusterisationLoss')(embs, c_embs.detach())
        entropy_val = self.update_loss('BasisClusterisationLoss', entropy)
        k_entropy = 0.01

        # local metric learning loss on images
        ing_lbls = torch.arange(lbls.size(0)).view(1, -1).repeat(n_augs_imgs, 1).transpose(0, 1).flatten().to(self.device)
        imgs_los_pos, imgs_los_neg = self.get_loss('ContrastiveLossOriginal_images')(basis_embs, ing_lbls)
        imgs_los_pos_val, imgs_los_neg_val = self.update_loss('ContrastiveLossOriginal_images', (imgs_los_pos, imgs_los_neg))

        # imgs recall
        imgs_recal = metric_Recall_top_K(basis_embs, ing_lbls, n_augs_imgs)

        # centroids recall
        centroids_recal = metric_Recall_top_K(c_embs, c_lbls, N_AUGMENTS)

        loss = (clust_los_pos + clust_los_neg) + \
               entropy * k_entropy + \
               (imgs_los_pos + imgs_los_neg)    

        tmp = c_embs.view(-1, N_AUGMENTS, c_embs.size(-1))
        centroids = tmp.index_select(1, torch.Tensor([0]).long().to(self.device)).squeeze() 

        return {
            'H': entropy_val * k_entropy,
            'Im_REC': imgs_recal,
            'Clst_REC': centroids_recal,

            'Clst_pos': clust_los_pos_val,
            'Clst_neg': clust_los_neg_val,

            'Im_pos': imgs_los_pos_val,
            'Im_neg': imgs_los_neg_val,
            
        }, loss, centroids

    def test_metric_learning_global(self, data, itr, step):
        return ""

    def traintest_okko_metric_learning(self, data, itr, step):
        samples = data

        embs = self.model(samples) #b*N_AUGMENTS, h
        lbls = torch.arange(int(embs.size(0)/N_AUGMENTS)).view(1, -1).repeat(N_AUGMENTS, 1).transpose(0, 1).flatten().to(self.device)

        # local metric learning loss
        embs_los_pos, embs_los_neg = self.get_loss('ContrastiveLossOriginal')(embs, lbls)
        embs_los_pos_val, embs_los_neg_val = self.update_loss('ContrastiveLossOriginal', (embs_los_pos, embs_los_neg))

        # recall
        embs_recal = metric_Recall_top_K(embs, lbls, N_AUGMENTS)

        loss = embs_los_pos + embs_los_neg

        return {
            'REC': embs_recal,

            'pos': embs_los_pos_val,
            'neg': embs_los_neg_val,
            
        }, loss
    
    def traintest_okko_domyshink(self, data, itr, step):
        samples, (true_lbls, fake_labels, rewards) = data
        true_lbls = true_lbls.to(self.device)
        fake_labels = fake_labels.to(self.device)
        rewards = rewards.to(self.device)

        if itr == 0:
            self.remove_loss('accuracy')
            self.add_loss(None, 'accuracy')

        embs = F.softmax(self.model(samples), -1) #b*N_AUGMENTS, num_classes
        lbls = torch.arange(int(embs.size(0)/N_AUGMENTS)).view(1, -1).repeat(N_AUGMENTS, 1).transpose(0, 1).flatten().to(self.device)

        # local metric learning loss
        embs_los_pos, embs_los_neg = self.get_loss('ContrastiveLossOriginal')(embs, lbls)
        embs_los_pos_val, embs_los_neg_val = self.update_loss('ContrastiveLossOriginal', (embs_los_pos, embs_los_neg))

        # rewards loss
        idx0 = torch.arange(embs.size(0)).expand(fake_labels.size(-1), embs.size(0)).transpose(0, 1).flatten().to(self.device)
        idx1 = fake_labels.flatten()
        topk = embs[idx0, idx1].view_as(rewards)
        rewards_loss = (topk * rewards).sum()
        rewards_loss_val = rewards_loss.item()

        # accuracy
        _, idx = embs.topk(dim=1, k=10, largest=True)
        corrects = 0
        A = idx.detach().cpu().numpy()
        B = true_lbls.detach().cpu().numpy()
        for x, y in zip(A, B):
            corrects += 1 if np.intersect1d(x, y).shape[0] > 0 else 0
        accuracy = float(corrects)/idx.size(0)
        accuracy_val = self.update_loss('accuracy', torch.Tensor([accuracy]))

        loss = (embs_los_pos + embs_los_neg) +\
                rewards_loss

        return {
            'ACC': accuracy_val,
            'acc': accuracy,
            'cor/tot': f'{corrects}/{idx.size(0)}',

            'rew': rewards_loss_val,

            'pos': embs_los_pos_val,
            'neg': embs_los_neg_val,
            
        }, loss

    def train_epoch(self, step):
        self.model.train()
        with tqdm.tqdm(total=len(self.train_loader)) as steps:

            for itr, data in enumerate(self.train_loader):
                self.optimizer.zero_grad()

                if CURRENT_PARAMS == 'cifar10_metric_learning_global':
                    message, loss, centroids_embeddings = self.train_metric_learning_global(data, itr, step)

                if CURRENT_PARAMS == 'cifar10_metric_learning_global_basis':
                    message, loss, centroids_embeddings = self.train_metric_learning_global_basis(data, itr, step)

                if CURRENT_PARAMS == 'okko_metric_learning':
                    message, loss = self.traintest_okko_metric_learning(data, itr, step)

                if CURRENT_PARAMS == 'okko_domyshik':
                    message, loss = self.traintest_okko_domyshink(data, itr, step)

                loss.backward()
                self.optimizer.step()

                steps.set_postfix(message)
                steps.update()
            
            if CURRENT_PARAMS in ['cifar10_metric_learning_global', 'cifar10_metric_learning_global_basis']:
                self.model.set_centroids(centroids_embeddings)
        
    def test_epoch(self, step):
        self.model.eval()
        with torch.no_grad():
            with tqdm.tqdm(total=len(self.test_loader)) as steps:

                for itr, data in enumerate(self.test_loader):

                    if CURRENT_PARAMS in ['cifar10_metric_learning_global', 'cifar10_metric_learning_global_basis']:
                        print(f'test not implemented for {CURRENT_PARAMS}')
                        break
                        #message = self.test_metric_learning_global(data, itr, step)
                    if CURRENT_PARAMS == 'okko_metric_learning':
                        message, _ = self.traintest_okko_metric_learning(data, itr, step)

                    if CURRENT_PARAMS == 'okko_domyshik':
                        message, loss = self.traintest_okko_domyshink(data, itr, step)

                    steps.set_postfix(message)
                    steps.update()
        
    def fit(self):
        for step in range(self.epochs):
            self.train_epoch(step + 1)
            self.test_epoch(step + 1)
            self.scheduler.step()

            save_model_params(self.model, self.model_name + str(step + 1))
            print('------------------------------------------------------')


def main():
    learner = Learner(launch_info=get_launch_info())
    learner.fit()

main()