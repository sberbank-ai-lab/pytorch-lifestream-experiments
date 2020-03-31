import torch
import torch.nn as nn
import numpy as np
import tqdm
from data_load import PaddedBatch

from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import roc_auc_score
import pandas as pd
from bisect import bisect_right
import constants as cns
import torch.nn.functional as F
from loss import PairwiseMarginRankingLoss, ContrastiveLoss


# ----------------------------------------------------------------------------------------------------------------------------------------------------------
# Optimizer/Schedulers layers


def batch_to_device(batch, device, non_blocking):
    x, y = batch
    if isinstance(x, PaddedBatch):
        new_x = {k: v.to(device=device, non_blocking=non_blocking) if isinstance(v, torch.Tensor) else v for k, v in
                 x.payload.items()}
        new_y = {k: v.to(device=device, non_blocking=non_blocking) if isinstance(v, torch.Tensor) else v for k, v in
                 y.items()}
        #if x.add_info is not None:
        #    x.add_info.cuda()
        return PaddedBatch(new_x, x.seq_lens, x.add_info), new_y
    else:
        x = x.to(device=device, non_blocking=non_blocking)
        y = y.to(device=device, non_blocking=non_blocking)
        return x, y


def get_optimizer(model):
    #return torch.optim.Adam(model.parameters(), lr=0.00001)
    return torch.optim.Adam(model.parameters(), lr=0.001)


def get_lr_scheduler(optimizer):
    lr_step_size = 1
    lr_step_gamma = 0.89
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_step_size, gamma=lr_step_gamma)
    return scheduler


# ----------------------------------------------------------------------------------------------------------------------------------------------------------
# Help Struct


class TInfo:

    def __init__(self, epoch, window_size, model, train_loader, valid_loader, loss, optimizer, scheduler, device):
        self.epoch = epoch
        self.window_size = window_size
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.loss = loss
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.writer = SummaryWriter('dlscoring')
        self.device = device
        self.roc_auc = -1
        self.roc_aucs = []
        self.epoh = 1

        self.b = torch.distributions.bernoulli.Bernoulli(probs=cns.SPARCNESS)

    def save_if_best_auc(self, roc_auc_score):
        self.roc_aucs.append(roc_auc_score)
        if roc_auc_score > self.roc_auc and cns.SAVE_MODEL_EACH_EPOCH:
            self.roc_auc = roc_auc_score
            torch.save(self.model, f'./best_{cns.SAVE_WEIGHTS_NAME}')
        torch.save(self.model, f'./{cns.SAVE_WEIGHTS_NAME}_{self.epoch}')

    def log(self):
        import datetime
        import os
        params = {"start_name": str(datetime.datetime.now()) if 'start_name' not in self.params else self.params['start_name'],
                  "used_config": self.params['used_config'],
                  "used_date_splits": self.params['used_date_splits'],
                  "used_optimizer_parameters": self.params['used_optimizer_parameters'],
                  "model_name": self.params['test_model_type'],
                  "best_auc": self.roc_auc,
                  "aucs": ' '.join(map(str, self.roc_aucs)),
                  "lr": self.params['lr'],
                  "step_gamma": self.params['lr_scheduler.step_gamma'],
                  "model_type": self.params['model_type']
                  }
        df = pd.DataFrame(params)
        if os.path.exists(f'./train_results.csv'):
            all_df = pd.read_csv(f'./train_results.csv')
            df = pd.concat((all_df, df), sort=False)
        df.to_csv(f'./train_results.csv')


def on_epoch_completed(tInfo: TInfo):
    tInfo.epoch += 1


def fit_model(model, train_loader, valid_loader, loss):
    device = torch.device('cuda')
    optimizer = get_optimizer(model)
    scheduler = get_lr_scheduler(optimizer)

    model.to(device)
    loss.to(device)

    tInfo = TInfo(None, 20, model, train_loader, valid_loader, loss, optimizer, scheduler, device)

    # test(tInfo)
    for epoch in range(cns.N_EPOCH):
        tInfo.epoch = epoch
        train(tInfo)
        test(tInfo)
        on_epoch_completed(tInfo)
        scheduler.step()

    # tInfo.log()


def running_average(x, tInfo):
    arr = np.array(x)
    #arr = arr[-tInfo.window_size:]
    res = np.convolve(arr, np.ones((arr.shape[0],)))/arr.shape[0]
    return res[0]


# ----------------------------------------------------------------------------------------------------------------------------------------------------------
# Train / Test


def maksing(t, lens, mode, tInfo):
    mask = None

    if mode == 'train':
        idxs = torch.arange(t.size(0) * t.size(1)).view(*t.size()) % t.size(1)
        ls = (lens - cns.TRAIN_WINDOW - cns.TEST_WINDOW - 1).unsqueeze(1).expand(*t.size())
        ls[ls < 0] = 0
        mask = torch.zeros_like(idxs, dtype=torch.float)
        mask = mask.masked_fill(idxs.long() < ls.long(), 1.0).cuda()

        # make sparce mask
        if cns.MAKE_SPARCE_PREDICTIONS:
            '''
            sparce_mask = tInfo.b.sample(mask.size()).float().cuda()
            ls = []
            for i, x in enumerate(lens):
                ls.extend([[i, j] for j in range(max(0, x.item() - cns.TRAIN_WINDOW - cns.TEST_WINDOW - 1), x.item())])
            res = np.array(ls).transpose()
            sparce_mask[res] = 1.0
            mask = mask * sparce_mask
            '''

            sparce_mask = torch.arange(t.size(1)).unsqueeze(0).repeat(t.size(0), 1).cuda()
            m = (sparce_mask % 2 == 1)
            sparce_mask.masked_fill_(m, 1)
            sparce_mask.masked_fill_((1 - m.int()).bool(), 0)
            ls = []
            for i, x in enumerate(lens):
                ls.extend([[i, j] for j in range(max(0, x.item() - cns.TRAIN_WINDOW - cns.TEST_WINDOW - 1), x.item())])
            res = np.array(ls).transpose()
            sparce_mask[res] = 1.0
            mask = mask * sparce_mask.float()

        mask = torch.cat((mask, mask), -1)

    elif mode == 'test':
        mask = torch.zeros(*lens.size(), dtype=torch.float)
        mask = mask.masked_fill((lens - cns.TEST_WINDOW - 1) >= 0, 1.0).cuda()
        mask = mask.unsqueeze(1).repeat(1, 2)

    return mask


def temporal_loss_ranking(x, y, tInfo, mode):
    e1, e2 = x[0].payload, x[1].payload
    y1, y2 = y['target1'], y['target2']
    seq_len = x[0].seq_lens

    mask = maksing(e1, seq_len, mode, tInfo)

    if mode == 'train':

        prediction = torch.cat((e1, e2), -1)
        label = torch.cat((y1, y2), -1)

        error = tInfo.loss(prediction, label, mask.long())
        return error, None, None

    elif mode == 'test':
        st = cns.TEST_WINDOW + 1
        e1_ = e1[range(len(e1)), [l - st if l - st >= 0 else 0 for l in seq_len]]
        e2_ = e2[range(len(e2)), [l - st if l - st >= 0 else 0 for l in seq_len]]
        y1 = y1[range(len(y1)),  [l - st if l - st >= 0 else 0 for l in seq_len]]
        y2 = y2[range(len(y2)),  [l - st if l - st >= 0 else 0 for l in seq_len]]

        pred = torch.cat((e1_, e2_), dim=0)
        true = torch.cat((y1, y2), dim=0)
        mask = mask.view(-1)
        return None, pred, true, mask


def contrastive_loss(x, y, tInfo):
    distns, out = x[0].add_info['distances'], x[0].add_info['rnn_out']
    seq_lens = x[0].seq_lens

    h = out[range(len(out)), [l - 1 for l in seq_lens]]
    #h = F.normalize(h, p=2, dim=1)

    ls = ContrastiveLoss(margin=cns.MARGIN)
    out = ls(embeddings=h, distances=distns)
    return out


def ratings_loss(x, y, tInfo):
    rnn_out = x[0].add_info['rnn_out']
    ceids = x[0].add_info['ceids']
    ratings = y['rating']

    mask = (ratings > 0).nonzero()
    if len(mask) == 0:
        return None

    out = rnn_out[mask[:, 0], mask[:, 1]]
    ratings = ratings[mask[:, 0], mask[:, 1]]
    ceid = ceids[mask[:, 0], mask[:, 1]]

    r = F.pairwise_distance(out, ceid)
    #l = nn.MSELoss()
    l = nn.SmoothL1Loss()

    error = l(r, ratings)
    return error


def train(tInfo):
    tInfo.model.train()
    losses, losses2, rate_losses = [], [], []
    with tqdm.tqdm(total=len(tInfo.train_loader)) as steps:
        for step, input_data in enumerate(tInfo.train_loader):
            tInfo.optimizer.zero_grad()

            # forward-backward pass
            device_data = batch_to_device(input_data, tInfo.device, True)
            e1, e2 = tInfo.model(device_data[0])
            output, _, _ = temporal_loss_ranking((e1, e2), device_data[1], tInfo, 'train')
            #output2 = 0.0005 * contrastive_loss((e1, e2), device_data[1], tInfo)
            rates_loss = ratings_loss((e1, e2), device_data[1], tInfo)
            if rates_loss is not None:
                rates_loss *= 0.001
                (output + rates_loss).backward()
            else:
                #(output + output2).backward()
                output.backward()

            # loss marging
            batch_loss = output.item()
            losses.append(batch_loss)
            average_loss = running_average(losses, tInfo)

            # loss contrustive
            #batch_loss2 = output2.item()
            # losses2.append(batch_loss2)
            average_loss2 = 0  # running_average(losses2, tInfo)

            # loss ratings
            if rates_loss is not None:
                batch_rate_loss = rates_loss.item()
                rate_losses.append(batch_rate_loss)
            average_rate_loss = running_average(rate_losses, tInfo)

            # log to terminal
            steps.set_description(f'train: epoch {tInfo.epoch}, step {step}/{len(tInfo.train_loader)}')
            steps.set_postfix({'marging_loss': '{:.5E}'.format(average_loss), 'ratings_loss': '{:.5E}'.format(average_rate_loss), 'contrustive_loss': '{:.5E}'.format(average_loss2), 'epoch': tInfo.epoch})
            steps.update()

            tInfo.optimizer.step()


def test(tInfo):
    tInfo.model.eval()
    with torch.no_grad():
        trues, preds = [], []
        with tqdm.tqdm(total=len(tInfo.valid_loader)) as steps:
            for step, input_data in enumerate(tInfo.valid_loader):
                # forward-backward pass
                device_data = batch_to_device(input_data, tInfo.device, True)
                e1, e2 = tInfo.model(device_data[0])
                output, pred, true, mask = temporal_loss_ranking((e1, e2), device_data[1], tInfo, 'test')
                steps.update()

                true = true[mask == 1]
                pred = pred[mask == 1]
                trues.append(true)
                preds.append(pred)

            # metric calculation
            trues = torch.cat(trues).detach().cpu().numpy()
            preds = torch.cat(preds).detach().cpu().numpy()
            auc_score = roc_auc_score(trues, preds)
            steps.set_description(f'epoch {tInfo.epoch} step {step}/{len(tInfo.valid_loader)}')
            steps.set_postfix({'epoch': tInfo.epoch, 'ROC_AUC': auc_score})
            tInfo.save_if_best_auc(auc_score)


# ----------------------------------------------------------------------------------------------------------------------------------------------------------
# Train / Test CLASSIFICATION


def maksing_classification(t, lens, mode, tInfo):
    mask = None

    if mode == 'train':
        idxs = torch.arange(t.size(0) * t.size(1)).view(*(t.size()[:-1])) % t.size(1)
        ls = (lens - cns.TRAIN_WINDOW - cns.TEST_WINDOW - 1).unsqueeze(1).expand(*(t.size()[:-1]))
        ls[ls < 0] = 0
        mask = torch.zeros_like(idxs, dtype=torch.float)
        mask = mask.masked_fill(idxs.long() < ls.long(), 1.0).cuda()

        # make sparce mask
        if cns.MAKE_SPARCE_PREDICTIONS:
            '''
            sparce_mask = tInfo.b.sample(mask.size()).float().cuda()
            ls = []
            for i, x in enumerate(lens):
                ls.extend([[i, j] for j in range(max(0, x.item() - cns.TRAIN_WINDOW - cns.TEST_WINDOW - 1), x.item())])
            res = np.array(ls).transpose()
            sparce_mask[res] = 1.0
            mask = mask * sparce_mask
            '''

            sparce_mask = torch.arange(t.size(1)).unsqueeze(0).repeat(t.size(0), 1).cuda()
            m = (sparce_mask % 2 == 1)
            sparce_mask.masked_fill_(m, 1)
            sparce_mask.masked_fill_((1 - m.int()).bool(), 0)
            ls = []
            for i, x in enumerate(lens):
                ls.extend([[i, j] for j in range(max(0, x.item() - cns.TRAIN_WINDOW - cns.TEST_WINDOW - 1), x.item())])
            res = np.array(ls).transpose()
            sparce_mask[res] = 1.0
            mask = mask * sparce_mask.float()

    elif mode == 'test':
        mask = torch.zeros(*lens.size(), dtype=torch.float)
        mask = mask.masked_fill((lens - cns.TEST_WINDOW - 1) >= 0, 1.0).cuda().bool()

    elif mode == 'prediction':
        mask = torch.zeros(*lens.size(), dtype=torch.float)
        mask = mask.masked_fill((lens - 1) >= 0, 1.0).cuda().bool()

    return mask


def temporal_loss_classification(x, y, tInfo, mode):
    e = x.payload
    y = y['target']
    seq_len = x.seq_lens

    mask = maksing_classification(e, seq_len, mode, tInfo)

    if mode == 'train':
        error = tInfo.loss(e, y, mask)
        return error, None, None

    elif mode == 'test':
        st = cns.TEST_WINDOW + 1
        pred = e[range(len(e)), [l - st if l - st >= 0 else 0 for l in seq_len]]
        target = y[range(len(y)),  [l - st if l - st >= 0 else 0 for l in seq_len]]

        pred = torch.exp(pred[mask])
        target = target[mask]

        idx_target = (target > 0).nonzero()[:, 1].view(-1, cns.TEST_WINDOW).detach().cpu().numpy()
        _, idx_pred = torch.sort(pred, descending=True)
        idx_pred_max = idx_pred[:, :cns.TEST_WINDOW].detach().cpu().numpy()
        #idx_pred_max = idx_pred[:, :20].detach().cpu().numpy()
        s, s_iou = 0.0, 0.0
        for p, t in zip(idx_pred_max, idx_target):
            s_pred, s_target = set(p), set(t)
            intersection = len(s_pred.intersection(s_target))
            union = len(s_pred.union(s_target))
            s_iou += float(intersection)/cns.TEST_WINDOW #union
            s += 1.0 if intersection > 0 else 0.0
        return s / idx_target.shape[0], s_iou / idx_target.shape[0]


def fit_model_classification(model, train_loader, valid_loader, loss):
    device = torch.device('cuda')
    optimizer = get_optimizer(model)
    scheduler = get_lr_scheduler(optimizer)

    model.to(device)
    loss.to(device)

    tInfo = TInfo(None, 20, model, train_loader, valid_loader, loss, optimizer, scheduler, device)

    #test_classification(tInfo)
    for epoch in range(cns.N_EPOCH):
        tInfo.epoch = epoch
        train_classification(tInfo)
        test_classification(tInfo)
        on_epoch_completed(tInfo)
        scheduler.step()

    # tInfo.log()


def train_classification(tInfo):
    tInfo.model.train()
    classify_losses, rate_losses = [], [0.0]
    with tqdm.tqdm(total=len(tInfo.train_loader)) as steps:
        for step, input_data in enumerate(tInfo.train_loader):
            tInfo.optimizer.zero_grad()

            # forward-backward pass
            device_data = batch_to_device(input_data, tInfo.device, True)
            e = tInfo.model(device_data[0])
            classify_loss, _, _ = temporal_loss_classification(e, device_data[1], tInfo, 'train')
            #rates_loss = ratings_loss((e, None), device_data[1], tInfo)
            rates_loss = None
            if rates_loss is not None:
                rates_loss *= 0.02
                (classify_loss + rates_loss).backward()
            else:
                classify_loss.backward()

            # loss classification
            batch_loss = classify_loss.item()
            classify_losses.append(batch_loss)
            average_classify_loss = running_average(classify_losses, tInfo)

            # loss ratings
            if rates_loss is not None:
                batch_rate_loss = rates_loss.item()
                rate_losses.append(batch_rate_loss)
            average_rate_loss = running_average(rate_losses, tInfo)

            # log to terminal
            steps.set_description(f'train: epoch {tInfo.epoch}, step {step}/{len(tInfo.train_loader)}')
            steps.set_postfix({'kl divergent loss': '{:.5E}'.format(average_classify_loss), 'rating loss': '{:.5E}'.format(average_rate_loss), 'epoch': tInfo.epoch})
            steps.update()

            tInfo.optimizer.step()


def test_classification(tInfo):
    tInfo.model.eval()
    with torch.no_grad():
        guesses, guessed_iou = 0.0, 0.0
        with tqdm.tqdm(total=len(tInfo.valid_loader)) as steps:
            for step, input_data in enumerate(tInfo.valid_loader):

                # forward-backward pass
                device_data = batch_to_device(input_data, tInfo.device, True)
                e = tInfo.model(device_data[0])
                guesses_, guessed_iou_ = temporal_loss_classification(e, device_data[1], tInfo, 'test')
                guesses += guesses_
                guessed_iou += guessed_iou_

                steps.set_postfix({'guesses': '{:.2E}'.format(guesses / (step + 1)), 'guessed_iou': '{:.2E}'.format(guessed_iou / (step + 1))})
                steps.update()

            tInfo.save_if_best_auc(guesses)


def get_classification_result(x, y, top_k, mode, to_filter_seen_items, allowed_items):
    e = x.payload
    uids = y[cns.USER_ID_COLUMN]
    prev_ceids = y['prev_ceids']
    seq_len = x.seq_lens

    mask = maksing_classification(e, seq_len, mode, None)

    pred = e[range(len(e)), [l - 1 for l in seq_len]]

    pred = torch.exp(pred[mask])

    # mask for already seen films
    if to_filter_seen_items:
        k = (prev_ceids > 0).nonzero().transpose(0, 1)
        mask_already_seen = torch.stack((k[0, :], prev_ceids[k[0, :], k[1, :]]))
        pred[mask_already_seen[0, :], mask_already_seen[1, :]] = -1.0 # set for these items negative score

    # maks not allowed items
    if allowed_items is not None:
        allowed_items.cuda()
        allowed_items.unsqueeze(0).repeat(pred.size(0), 1)
        k = (allowed_items > 0).nonzero().transpose(0, 1)
        mask_allowed_items = torch.stack((k[0, :], prev_ceids[k[0, :], k[1, :]]))
        pred[mask_allowed_items[0, :], mask_allowed_items[1, :]] = -1.0  # set for these items negative score

    scores_pred, idx_pred = torch.sort(pred, descending=True)
    scores_pred, idx_pred = scores_pred[:, :top_k].contiguous(), idx_pred[:, :top_k].contiguous()

    uids = uids.repeat(1, idx_pred.size(1))
    predictions = torch.cat((uids.view(-1).unsqueeze(1).float(),
                             idx_pred.view(-1).unsqueeze(1).float(),
                             scores_pred.view(-1).unsqueeze(1)), -1).detach().cpu().numpy()
    return predictions


def predict_model_classification(top_k, model, data_loader, to_filter_seen_items, allowed_items):
    model.to(torch.device('cuda'))
    model.eval()

    predictions = np.ones((len(data_loader) * cns.BATCH_SIZE * top_k, 3)) * -100
    with torch.no_grad():
        guesses, guessed_iou = 0.0, 0.0
        with tqdm.tqdm(total=len(data_loader)) as steps:
            for step, input_data in enumerate(data_loader):
                device_data = batch_to_device(input_data, torch.device('cuda'), True)
                e = model(device_data[0])
                #guesses_, guessed_iou_ = temporal_loss_classification(e, device_data[1], None, 'test')
                #guesses += guesses_
                #guessed_iou += guessed_iou_

                #steps.set_postfix({'guesses': '{:.2E}'.format(guesses / (step + 1)),
                #                   'guessed_iou': '{:.2E}'.format(guessed_iou / (step + 1))})

                preds = get_classification_result(e, device_data[1], top_k, 'prediction', to_filter_seen_items, allowed_items)
                #predictions[steps.n * cns.BATCH_SIZE * top_k: (steps.n + 1) * cns.BATCH_SIZE * top_k, :] = preds[:, :]
                f_ = steps.n * cns.BATCH_SIZE * top_k
                t_ = f_ + preds.shape[0]
                predictions[f_: t_, :] = preds[:, :]

                steps.update()

    return predictions


# ----------------------------------------------------------------------------------------------------------------------------------------------------------
# Train / Test ORDINAL REGRESSION


def maksing_ordinal(t, lens, mode, tInfo):
    mask = None

    if mode == 'train':
        idxs = torch.arange(t.size(0) * t.size(1)).view(*(t.size()[:-1])) % t.size(1)
        ls = (lens - cns.TRAIN_WINDOW - cns.TEST_WINDOW - 1).unsqueeze(1).expand(*(t.size()[:-1]))
        ls[ls < 0] = 0
        mask = torch.zeros_like(idxs, dtype=torch.float)
        mask = mask.masked_fill(idxs.long() < ls.long(), 1.0).cuda()

        # make sparce mask
        if cns.MAKE_SPARCE_PREDICTIONS:
            '''
            sparce_mask = tInfo.b.sample(mask.size()).float().cuda()
            ls = []
            for i, x in enumerate(lens):
                ls.extend([[i, j] for j in range(max(0, x.item() - cns.TRAIN_WINDOW - cns.TEST_WINDOW - 1), x.item())])
            res = np.array(ls).transpose()
            sparce_mask[res] = 1.0
            mask = mask * sparce_mask
            '''

            sparce_mask = torch.arange(t.size(1)).unsqueeze(0).repeat(t.size(0), 1).cuda()
            m = (sparce_mask % 2 == 1)
            sparce_mask.masked_fill_(m, 1)
            sparce_mask.masked_fill_((1 - m.int()).bool(), 0)
            ls = []
            for i, x in enumerate(lens):
                ls.extend([[i, j] for j in range(max(0, x.item() - cns.TRAIN_WINDOW - cns.TEST_WINDOW - 1), x.item())])
            res = np.array(ls).transpose()
            sparce_mask[res] = 1.0
            mask = mask * sparce_mask.float()

    elif mode == 'test':
        mask = torch.zeros(*lens.size(), dtype=torch.float)
        mask = mask.masked_fill((lens - cns.TEST_WINDOW - 1) >= 0, 1.0).cuda().bool()

    return mask


def temporal_loss_ordinal_(x, y, tInfo, mode):
    e = x.payload

    tresholds = x.add_info['tresholds']
    if cns.USE_CONSTANT_ORDINAL_TRESHOLDS:
        #t = torch.Tensor([1/ (1000 * cns.N_CEIDS), 0.2, 0.6, 0.8]).cuda()
        t = torch.Tensor([1 / (100 * cns.N_CEIDS), 0.1, 0.4, 0.65]).cuda()
        tresholds = t.unsqueeze(0).unsqueeze(0).repeat(tresholds.size(0), tresholds.size(1), 1)

    target = torch.cat([v.unsqueeze(-1) for k, v in y.items() if 'poc_ceid' in k], dim=-1)
    seq_len = x.seq_lens

    mask = maksing_ordinal(e, seq_len, mode, tInfo)
    target_mask = y['target_mask'].view(e.size(0), e.size(1), -1)

    if mode == 'train':
        error = tInfo.loss(predictions=e, tresholds=tresholds, target=target, mask=mask, target_mask=target_mask)
        return error, None, None

    elif mode == 'test':
        st = cns.TEST_WINDOW + 1
        pred = e[range(len(e)), [l - st if l - st >= 0 else 0 for l in seq_len]]
        target = target[range(len(target)),  [l - st if l - st >= 0 else 0 for l in seq_len]]

        pred = pred[mask]
        target = target[mask]

        #idx_target = (target > 0).nonzero()[:, 1].view(-1, cns.TEST_WINDOW).detach().cpu().numpy()
        idx_target = target.view(-1, cns.TEST_WINDOW).detach().cpu().numpy()
        _, idx_pred = torch.sort(pred, descending=True)
        idx_pred_max = idx_pred[:, :cns.TEST_WINDOW].detach().cpu().numpy()
        #idx_pred_max = idx_pred[:, :20].detach().cpu().numpy()
        s, s_iou = 0.0, 0.0
        for p, t in zip(idx_pred_max, idx_target):
            s_pred, s_target = set(p), set(t)
            intersection = len(s_pred.intersection(s_target))
            union = len(s_pred.union(s_target))
            s_iou += float(intersection)/cns.TEST_WINDOW #union
            s += 1.0 if intersection > 0 else 0.0
        return s / idx_target.shape[0], s_iou / idx_target.shape[0]


def temporal_loss_ordinal(x, y, tInfo, mode):
    e = x.payload

    #tresholds = x.add_info['tresholds']
    if cns.USE_CONSTANT_ORDINAL_TRESHOLDS:
        #t = torch.Tensor([1/ (1000 * cns.N_CEIDS), 0.2, 0.6, 0.8]).cuda()
        #tresholds = torch.Tensor([1 / (100 * cns.N_CEIDS), 0.1, 0.4, 0.65]).cuda()
        tresholds = torch.Tensor([1 / (100 * cns.N_CEIDS), 1 / (100 * cns.N_CEIDS), 1 / (100 * cns.N_CEIDS), 1 / (100 * cns.N_CEIDS)]).cuda()

    target = torch.cat([v.unsqueeze(-1) for k, v in y.items() if 'poc_ceid' in k], dim=-1)
    seq_len = x.seq_lens

    mask = maksing_ordinal(e, seq_len, mode, tInfo)

    if mode == 'train':
        error = tInfo.loss(predictions=e, tresholds=tresholds, target=target, mask=mask)
        return error, None, None

    elif mode == 'test':
        st = cns.TEST_WINDOW + 1
        pred = e[range(len(e)), [l - st if l - st >= 0 else 0 for l in seq_len]]
        target = target[range(len(target)),  [l - st if l - st >= 0 else 0 for l in seq_len]]

        pred = pred[mask]
        target = target[mask]

        #idx_target = (target > 0).nonzero()[:, 1].view(-1, cns.TEST_WINDOW).detach().cpu().numpy()
        idx_target = target.view(-1, cns.TEST_WINDOW).detach().cpu().numpy()
        _, idx_pred = torch.sort(pred, descending=True)
        idx_pred_max = idx_pred[:, :cns.TEST_WINDOW].detach().cpu().numpy()
        #idx_pred_max = idx_pred[:, :20].detach().cpu().numpy()
        s, s_iou = 0.0, 0.0
        for p, t in zip(idx_pred_max, idx_target):
            s_pred, s_target = set(p), set(t)
            intersection = len(s_pred.intersection(s_target))
            union = len(s_pred.union(s_target))
            s_iou += float(intersection)/cns.TEST_WINDOW #union
            s += 1.0 if intersection > 0 else 0.0
        return s / idx_target.shape[0], s_iou / idx_target.shape[0]


def fit_model_ordinal(model, train_loader, valid_loader, loss):
    device = torch.device('cuda')
    optimizer = get_optimizer(model)
    scheduler = get_lr_scheduler(optimizer)

    model.to(device)
    loss.to(device)

    tInfo = TInfo(None, 20, model, train_loader, valid_loader, loss, optimizer, scheduler, device)

    #tInfo.model = torch.load(f'/data/molchanov/okko/ils_via_backprop/model_weights_several_test_with_separation_epoch_9')
    #test_ordinal(tInfo)
    for epoch in range(cns.N_EPOCH):
        tInfo.epoch = epoch
        train_ordinal(tInfo)
        test_ordinal(tInfo)
        on_epoch_completed(tInfo)
        scheduler.step()

    # tInfo.log()


def train_ordinal(tInfo):
    tInfo.model.train()
    ordinal_losses, rate_losses = [], [0.0]
    with tqdm.tqdm(total=len(tInfo.train_loader)) as steps:
        for step, input_data in enumerate(tInfo.train_loader):
            tInfo.optimizer.zero_grad()

            # forward-backward pass
            device_data = batch_to_device(input_data, tInfo.device, True)
            e = tInfo.model(device_data[0])
            ordinal_loss, _, _ = temporal_loss_ordinal(e, device_data[1], tInfo, 'train')
            #rates_loss = ratings_loss((e, None), device_data[1], tInfo)
            rates_loss = None
            if rates_loss is not None:
                rates_loss *= 0.02
                (ordinal_loss + rates_loss).backward()
            else:
                ordinal_loss.backward()

            # loss classification
            batch_loss = ordinal_loss.item()
            ordinal_losses.append(batch_loss)
            average_ordinal_loss = running_average(ordinal_losses, tInfo)

            # loss ratings
            if rates_loss is not None:
                batch_rate_loss = rates_loss.item()
                rate_losses.append(batch_rate_loss)
            average_rate_loss = running_average(rate_losses, tInfo)

            # log to terminal
            steps.set_description(f'train: epoch {tInfo.epoch}, step {step}/{len(tInfo.train_loader)}')
            steps.set_postfix({'ordinal loss': '{:.5E}'.format(average_ordinal_loss), 'rating loss': '{:.5E}'.format(average_rate_loss), 'epoch': tInfo.epoch})
            steps.update()

            tInfo.optimizer.step()


def test_ordinal(tInfo):
    tInfo.model.eval()
    with torch.no_grad():
        guesses, guessed_iou = 0.0, 0.0
        with tqdm.tqdm(total=len(tInfo.valid_loader)) as steps:
            for step, input_data in enumerate(tInfo.valid_loader):

                # forward-backward pass
                device_data = batch_to_device(input_data, tInfo.device, True)
                e = tInfo.model(device_data[0])
                guesses_, guessed_iou_ = temporal_loss_ordinal(e, device_data[1], tInfo, 'test')
                guesses += guesses_
                guessed_iou += guessed_iou_

                steps.set_postfix({'guesses': '{:.2E}'.format(guesses / (step + 1)), 'guessed_iou': '{:.2E}'.format(guessed_iou / (step + 1))})
                steps.update()

            tInfo.save_if_best_auc(guesses)


# ----------------------------------------------------------------------------------------------------------------------------------------------------------
# Train / Test POPULAR


def maksing_popular(t, lens, tInfo):
    mask = torch.zeros(*lens.size(), dtype=torch.float)
    mask = mask.masked_fill((lens - cns.TEST_WINDOW - 1) >= 0, 1.0).cuda().bool()
    return mask


def temporal_loss_popular(x, y, tInfo, popular):
    e = x.payload['ceid'][:, :-1]
    target = torch.cat([v.unsqueeze(-1) for k, v in y.items()], dim=-1)
    seq_len = x.seq_lens - 1

    mask = maksing_popular(e, seq_len, tInfo)

    st = cns.TEST_WINDOW + 1
    ls = seq_len - st + 1
    ls[ls < 0] = 0
    ls = ls.unsqueeze(1).repeat(1, e.size(1))
    r = torch.arange(ls.size(1)).unsqueeze(0).repeat(ls.size(0), 1)
    m = (ls.long() > r).long().cuda()
    known = e * m

    #pred = e[range(len(e)), [l - st if l - st >= 0 else 0 for l in seq_len]]
    target = target[range(len(target)),  [l - st if l - st >= 0 else 0 for l in seq_len]]

    known = known[mask].detach().cpu().numpy()
    target = target[mask].detach().cpu().numpy()

    s, s_iou = 0.0, 0.0
    for kn, t in zip(known, target):
        s_know, s_target = set(kn), set(t)
        s_pred = set()
        for pop_item in popular:
            if pop_item not in s_know:
                s_pred.add(pop_item)
                if len(s_pred) == len(s_target):
                    break

        intersection = len(s_pred.intersection(s_target))
        #union = len(s_pred.union(s_target))
        s_iou += float(intersection)/cns.TEST_WINDOW #union
        s += 1.0 if intersection > 0 else 0.0
    return s / target.shape[0], s_iou / target.shape[0]


def fit_model_popular(data_loader, popular):
    device = torch.device('cuda')
    tInfo = TInfo(None, 20, None, None, data_loader, None, None, None, device)

    test_popular(tInfo, popular)


def test_popular(tInfo, popular):
    guesses, guessed_iou = 0.0, 0.0
    with tqdm.tqdm(total=len(tInfo.valid_loader)) as steps:
        for step, input_data in enumerate(tInfo.valid_loader):

            # forward-backward pass
            device_data = batch_to_device(input_data, tInfo.device, True)
            guesses_, guessed_iou_ = temporal_loss_popular(device_data[0], device_data[1], tInfo, popular)
            guesses += guesses_
            guessed_iou += guessed_iou_

            steps.set_postfix({'guesses': '{:.2E}'.format(guesses / (step + 1)), 'guessed_iou': '{:.2E}'.format(guessed_iou / (step + 1))})
            steps.update()
