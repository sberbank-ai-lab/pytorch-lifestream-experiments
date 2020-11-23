import logging
import torch
import torch.nn as nn

from ignite.metrics import Loss, RunningAverage
from torch.nn import functional as F

from dltranz.trx_encoder import PaddedBatch
from dltranz.experiment import update_model_stats, CustomMetric
from dltranz.train import get_optimizer, get_lr_scheduler, fit_model, CheckpointHandler

logger = logging.getLogger(__name__)


class CPC_Encoder(nn.Module):
    def __init__(self, encoder, conf):
        super().__init__()
        self.encoder = encoder
        linear_size = encoder.input_size
        embedding_size = encoder.output_size
        self.linears = nn.ModuleList([nn.Linear(embedding_size, linear_size) for _ in range(conf['n_forward_steps'])])

    def forward(self, x):
        base_embeddings = x
        context_embeddings = self.encoder(base_embeddings)
        mapped_ctx_embeddings = torch.stack([linear_layer(context_embeddings) for linear_layer in self.linears], dim=-1)
        return base_embeddings, context_embeddings, mapped_ctx_embeddings


class CPC_Padded_Encoder(nn.Module):
    def __init__(self, trx_encoder, seq_encoder, linear_size, conf):
        super().__init__()
        self.trx_encoder = trx_encoder
        self.seq_encoder = seq_encoder
        embedding_size = seq_encoder.hidden_size
        linear_size = linear_size
        self.linears = nn.ModuleList([nn.Linear(embedding_size, linear_size) for i in range(conf['n_forward_steps'])])

    def forward(self, x: PaddedBatch):
        base_embeddings = self.trx_encoder(x)
        context_embeddings = self.seq_encoder(base_embeddings)

        me = []
        for l in self.linears:
            me.append(l(context_embeddings.payload))
        mapped_ctx_embeddings = PaddedBatch(torch.stack(me, dim=3), context_embeddings.seq_lens)

        return base_embeddings, context_embeddings, mapped_ctx_embeddings


class CPC_Padded_Loss(nn.Module):
    def __init__(self, n_negatives):
        super().__init__()
        self.n_negatives = n_negatives

    def _get_preds(self, base_embeddings: PaddedBatch, mapped_ctx_embeddings: PaddedBatch):
        batch_size, max_seq_len, emb_size = base_embeddings.payload.shape
        _, _, _, n_forward_steps = mapped_ctx_embeddings.payload.shape
        seq_lens = mapped_ctx_embeddings.seq_lens
        device = mapped_ctx_embeddings.payload.device

        len_mask = torch.arange(max_seq_len).unsqueeze(0).expand(batch_size, -1).to(device)
        len_mask = (len_mask < seq_lens.unsqueeze(1).expand(-1, max_seq_len)).float()

        possible_negatives = base_embeddings.payload.view(batch_size * max_seq_len, emb_size)

        mask = len_mask.unsqueeze(0).expand(batch_size, *len_mask.shape).clone()

        mask = mask.reshape(batch_size, -1)
        sample_ids = torch.multinomial(mask, self.n_negatives)
        neg_samples = possible_negatives[sample_ids]

        positive_preds, neg_preds = [], []
        len_mask_exp = len_mask.unsqueeze(-1).unsqueeze(-1).to(device).expand(-1, -1, emb_size, n_forward_steps)
        trimmed_mce = mapped_ctx_embeddings.payload.mul(len_mask_exp)  # zero context vectors by sequence lengths
        for i in range(1, n_forward_steps + 1):
            ce_i = trimmed_mce[:, 0:max_seq_len - i, :, i - 1]
            be_i = base_embeddings.payload[:, i:max_seq_len]

            positive_pred_i = ce_i.mul(be_i).sum(axis=-1)
            positive_preds.append(positive_pred_i)

            neg_pred_i = ce_i.matmul(neg_samples.transpose(-2, -1))
            neg_pred_i = neg_pred_i
            neg_preds.append(neg_pred_i)

        return positive_preds, neg_preds

    def forward(self, embeddings, _):
        base_embeddings, _, mapped_ctx_embeddings = embeddings
        device = mapped_ctx_embeddings.payload.device
        positive_preds, neg_preds = self._get_preds(base_embeddings, mapped_ctx_embeddings)

        step_losses = []
        for positive_pred_i, neg_pred_i in zip(positive_preds, neg_preds):
            step_loss = -F.log_softmax(torch.cat([positive_pred_i.unsqueeze(-1), neg_pred_i], dim=-1), dim=-1)[:, :,
                         0].mean()
            step_losses.append(step_loss)

        loss = torch.stack(step_losses).mean()
        return loss

    def cpc_accuracy(self, embeddings, _):
        base_embeddings, _, mapped_ctx_embeddings = embeddings
        positive_preds, neg_preds = self._get_preds(base_embeddings, mapped_ctx_embeddings)

        batch_size, max_seq_len, emb_size = base_embeddings.payload.shape
        seq_lens = mapped_ctx_embeddings.seq_lens
        device = mapped_ctx_embeddings.payload.device

        len_mask = torch.arange(max_seq_len).unsqueeze(0).expand(batch_size, -1).to(device)
        len_mask = (len_mask < seq_lens.unsqueeze(1).expand(-1, max_seq_len)).float()

        total, accurate = 0, 0
        for i, (positive_pred_i, neg_pred_i) in enumerate(zip(positive_preds, neg_preds)):
            i_mask = len_mask[:, (i + 1):max_seq_len].to(device)
            total += i_mask.sum().item()
            accurate += (((positive_pred_i.unsqueeze(-1).expand(*neg_pred_i.shape) > neg_pred_i) \
                          .sum(dim=-1) == self.n_negatives) * i_mask).sum().item()
        return accurate / total


class CPC_Loss(nn.Module):
    def __init__(self, n_negatives):
        super().__init__()
        self.n_negatives = n_negatives

    def _get_preds(self, base_embeddings: torch.Tensor, mapped_ctx_embeddings: torch.Tensor):
        batch_size, seq_len, emb_size = base_embeddings.shape
        _, _, _, n_forward_steps = mapped_ctx_embeddings.shape
        device = mapped_ctx_embeddings.device

        len_mask = torch.ones(seq_len).unsqueeze(0).expand(batch_size, -1).to(device).float()

        possible_negatives = base_embeddings.view(batch_size * seq_len, emb_size)

        mask = len_mask.unsqueeze(0).expand(batch_size, *len_mask.shape).clone()

        mask = mask.reshape(batch_size, -1)
        sample_ids = torch.multinomial(mask, self.n_negatives)
        neg_samples = possible_negatives[sample_ids]

        positive_preds, neg_preds = [], []
        len_mask_exp = len_mask.unsqueeze(-1).unsqueeze(-1).to(device).expand(-1, -1, emb_size, n_forward_steps)
        trimmed_mce = mapped_ctx_embeddings.mul(len_mask_exp)  # zero context vectors by sequence lengths

        for i in range(1, n_forward_steps + 1):
            ce_i = trimmed_mce[:, 0:seq_len - i, :, i - 1]
            be_i = base_embeddings[:, i:seq_len]

            positive_pred_i = ce_i.mul(be_i).sum(axis=-1)
            positive_preds.append(positive_pred_i)

            neg_pred_i = ce_i.matmul(neg_samples.transpose(-2, -1))
            neg_pred_i = neg_pred_i
            neg_preds.append(neg_pred_i)

        return positive_preds, neg_preds

    def forward(self, embeddings, _):
        base_embeddings, _, mapped_ctx_embeddings = embeddings
        device = mapped_ctx_embeddings.device
        positive_preds, neg_preds = self._get_preds(base_embeddings, mapped_ctx_embeddings)

        step_losses = []
        for positive_pred_i, neg_pred_i in zip(positive_preds, neg_preds):
            step_loss = -F.log_softmax(torch.cat([positive_pred_i.unsqueeze(-1), neg_pred_i], dim=-1), dim=-1)[:, :,
                         0].mean()
            step_losses.append(step_loss)

        loss = torch.stack(step_losses).mean()
        return loss

    def cpc_accuracy(self, embeddings, _):
        base_embeddings, _, mapped_ctx_embeddings = embeddings
        positive_preds, neg_preds = self._get_preds(base_embeddings, mapped_ctx_embeddings)

        batch_size, seq_len, emb_size = base_embeddings.shape
        device = mapped_ctx_embeddings.device

        len_mask = torch.ones(seq_len).unsqueeze(0).expand(batch_size, -1).to(device).float()

        total, accurate = 0, 0
        for i, (positive_pred_i, neg_pred_i) in enumerate(zip(positive_preds, neg_preds)):
            i_mask = len_mask[:, (i + 1):seq_len].to(device)
            total += i_mask.sum().item()
            accurate += (((positive_pred_i.unsqueeze(-1).expand(*neg_pred_i.shape) > neg_pred_i) \
                          .sum(dim=-1) == self.n_negatives) * i_mask).sum().item()
        return accurate / total


def run_experiment(train_loader, valid_loader, model, conf, is_padded=True):
    import time
    start = time.time()

    params = conf['params']

    if is_padded:
        loss = CPC_Padded_Loss(n_negatives=params['cpc.n_negatives'])
    else:
        loss = CPC_Loss(n_negatives=params['cpc.n_negatives'])

    valid_metric = {
        'loss': RunningAverage(Loss(loss)),
        'cpc accuracy': CustomMetric(lambda outputs, y: loss.cpc_accuracy(outputs, y))
    }

    optimizer = get_optimizer(model, params)
    scheduler = get_lr_scheduler(optimizer, params)

    train_handlers = []
    if 'checkpoints' in params['train']:
        checkpoint = CheckpointHandler(
            model=model,
            **params['train.checkpoints']
        )
        train_handlers.append(checkpoint)

    metric_values = fit_model(
        model,
        train_loader,
        valid_loader,
        loss,
        optimizer,
        scheduler,
        params,
        valid_metric,
        train_handlers=train_handlers)

    exec_sec = time.time() - start

    results = {
        'exec-sec': exec_sec,
        'metrics': metric_values,
    }

    stats_file = conf.get('stats.path', None)

    if stats_file is not None:
        update_model_stats(stats_file, params, results)
    else:
        return results


class CPCShellV2(torch.nn.Module):
    def __init__(self, encoder, embedding_size, k_pos_samples):
        super().__init__()

        self.encoder = encoder
        self.k_pos_samples = k_pos_samples

        history_size = k_pos_samples - 1  # predict one last sample based on all previous
        self.linear_predictor = torch.nn.Linear(embedding_size * history_size, embedding_size)

    def forward(self, x):
        z = self.encoder(x)
        return z


class CPCLossV2(torch.nn.Module):
    def __init__(self, k_pos_samples, m_neg_samples, linear_predictor):
        super(CPCLossV2, self).__init__()

        self.k_pos_samples = k_pos_samples
        self.m_neg_samples = m_neg_samples

        self.linear_predictor = linear_predictor

    def forward(self, embeddings, target):
        embeddings = embeddings

        k_pos_samples = self.k_pos_samples
        n = embeddings.size()[0] // k_pos_samples
        h = embeddings.size()[1]
        m_neg_samples = min(self.m_neg_samples, k_pos_samples * (n - 1))

        # assert m_neg_samples <= (n - 1) * k_pos_samples, (m_neg_samples, (n - 1) * k_pos_samples)

        # pos pred
        history_x_indexes = (torch.arange(n * k_pos_samples) + 1) % k_pos_samples != 0
        history_y_indexes = (torch.arange(n * k_pos_samples) + 1) % k_pos_samples == 0

        hist_x = embeddings[history_x_indexes].reshape(n, -1)  # shape: [n, embedding_size * (k - 1)]
        hist_y = embeddings[history_y_indexes]

        predicts = self.linear_predictor(hist_x)
        positive_pred_logit = predicts.mul(hist_y).sum(axis=-1)

        # negatives
        x = ((target.expand(n * k_pos_samples, n * k_pos_samples) -
              target.expand(n * k_pos_samples, n * k_pos_samples).t()) != 0).nonzero(as_tuple=False)[:, 1]
        neg_samples = x.view(n, k_pos_samples, -1)[:, 0, :]
        perm_ix = torch.cat(
            [torch.stack([torch.ones(m_neg_samples).long() * i,
                          torch.randperm(k_pos_samples * (n - 1))[:m_neg_samples]]).t() for i in range(n)])
        neg_embed = embeddings[neg_samples[perm_ix[:, 0], perm_ix[:, 1]]].view(n, m_neg_samples, h)

        neg_logit = (predicts.unsqueeze(1).repeat(1, m_neg_samples, 1) * neg_embed).sum(-1)

        loss = torch.nn.functional.log_softmax(
            torch.cat([positive_pred_logit.unsqueeze(-1), neg_logit], dim=-1),
            dim=-1)[:, 0]
        return -1.0 * loss.mean(), None
