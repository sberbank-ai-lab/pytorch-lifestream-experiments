import torch
import torch.nn as nn
from collections import OrderedDict
from data_load import PaddedBatch
import constants as cns
import torch.nn.functional as F
import math
import scipy.sparse as sp
from implicit.nearest_neighbours import TFIDFRecommender
import numpy as np
import tqdm
from collections import defaultdict
from constants import extend_add_info as exai

# ----------------------------------------------------------------------------------------------------------------------------------------------------------
# Embedding/Other layers


class NormEncoder(nn.Module):

    def forward(self, x: torch.Tensor):
        return F.normalize(x, p=2, dim=2)
        #y = x.view(-1, x.size(-1))
        #yn = y.pow(2).sum(dim=1).pow(0.5).unsqueeze(-1).expand(*y.size()) + 0.00001
        #yr = y/yn
        #yr = yr.view(*x.size())
        #return yr


class NoisyEmbedding(nn.Embedding):
    """
    Embeddings with additive gaussian noise with mean=0 and user-defined variance.
    *args and **kwargs defined by usual Embeddings
    Args:
        noise_scale (float): when > 0 applies additive noise to embeddings.
            When = 0, forward is equivalent to usual embeddings.
        dropout (float): probability of embedding axis to be dropped. 0 means no dropout at all.

    For other parameters defenition look at nn.Embedding help
    """

    def __init__(self, num_embeddings, embedding_dim, padding_idx=None, max_norm=None,
                 norm_type=2.0, scale_grad_by_freq=False, sparse=False, _weight=None,
                 noise_scale=0, dropout=0):
        super().__init__(num_embeddings, embedding_dim, padding_idx, max_norm,
                         norm_type, scale_grad_by_freq, sparse, _weight)
        self.noise = torch.distributions.Normal(0, noise_scale)
        self.scale = noise_scale
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout(super().forward(x))
        if self.training and self.scale > 0:
            x += self.noise.sample((self.weight.shape[1], )).to(self.weight.device)
        return x


class BaseEncoder(nn.Module):

    def __init__(self, base_encoder, features):
        super().__init__()
        self.encoder = base_encoder
        self.features = features

    def forward(self, x: PaddedBatch):
        processed = []
        for field_name in self.features:
            processed.append(self.encoder(x.payload[field_name].long()))

        out = torch.cat(processed, -1)
        return PaddedBatch(out, x.seq_lens, x.add_info)

    def output_size(self):
        return cns.EMBEDDING_DIM# + len(self.numeric_values.keys())


class FeatureEncoder(nn.Module):

    def __init__(self, base_encoder):
        super().__init__()
        self.encoder = base_encoder

        self.cat_embeddings = nn.ModuleDict({k: NoisyEmbedding(num_embeddings=v['in'],
                                                               embedding_dim=v['out'],
                                                               padding_idx=0,
                                                               max_norm=None,
                                                               noise_scale=0.05)
                                            for k, v in cns.FEATURES.items() if v['type'] == 'cat'})

    def forward(self, x: PaddedBatch):
        # encode base ceids
        out1 = self.encoder(x)

        # encode cat features
        processed = []
        for feature_name in sorted(cns.FEATURES.keys()):
            if cns.FEATURES[feature_name]['type'] == 'cat':
                processed.append(self.cat_embeddings[feature_name](x.payload[feature_name]))
        out2 = torch.cat(processed, dim=-1) if len(processed) > 0 else None

        # encode reg features
        processed = []
        for feature_name in sorted(cns.FEATURES.keys()):
            if cns.FEATURES[feature_name]['type'] == 'reg':
                processed.append(x.payload[feature_name].unsqueeze(-1))
        out3 = torch.cat(processed, dim=-1) if len(processed) > 0 else None

        res = [out1.payload]
        if out2 is not None:
            res.append(out2)
        if out3 is not None:
            res.append(out3)

        out = torch.cat(res, dim=-1)
        return PaddedBatch(out, x.seq_lens, out1.add_info)

    def output_size(self):
        return self.encoder.output_size() + cns.get_features_embedding_size()


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=101):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :x.size(1)]
        return self.dropout(x)


class FModuleList(nn.ModuleList):

    def __init__(self, modules):
        super().__init__(modules)

    def forward(self, inp):
        out = []
        if isinstance(inp, PaddedBatch):
            for layer in self:
                out.append(layer(inp))
        elif isinstance(inp, list):
            for layer, x in zip(self, inp):
                out.append(layer(x))
        else:
            print('warning: not supported operand')
        return out


# ----------------------------------------------------------------------------------------------------------------------------------------------------------
# Seq Encoders


class RnnEncoder(nn.Module):

    def __init__(self, input_size):
        super().__init__()

        self.hidden_size = cns.RNN_HIDDEN_DIM
        self.trainable_starter = 'not static'

        # initialize RNN
        self.rnn = nn.GRU(
            input_size,
            self.hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=False)

        self.full_hidden_size = self.hidden_size

        # initialize starter position if needed
        if self.trainable_starter == 'static':
            self.starter_h = nn.Parameter(torch.randn(1, 1, self.hidden_size))

    def forward(self, x: PaddedBatch):
        if self.trainable_starter == 'static':
            h_0 = self.starter_h.expand(-1, len(x.seq_lens), -1).contiguous()
        else:
            h_0 = None

        out, _ = self.rnn(x.payload, h_0)
        return PaddedBatch(out, x.seq_lens, exai(x.add_info, 'ceids', x.payload))


class BertTransformerEncoder(nn.Module):

    def __init__(self, input_size):
        super().__init__()
        self.input_size = input_size
        encoder_layer = nn.TransformerEncoderLayer(d_model=input_size, nhead=cns.TRANSFORMER_HEADS, dim_feedforward=input_size)
        encoder_norm = nn.LayerNorm(cns.EMBEDDING_DIM)
        self.enc = nn.TransformerEncoder(encoder_layer, num_layers=cns.TRANSFORMER_LAYERS, norm=encoder_norm)
        #self.pe = PositionalEncoding(d_model=input_size)

    def get_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask.cuda()

    def forward(self, inp: PaddedBatch):
        x = torch.transpose(inp.payload, 0, 1)
        #x = self.pe(x)

        out = self.enc(src=x, mask=self.get_mask(x.size(0)))
        out = torch.transpose(out, 0, 1)

        return PaddedBatch(out, inp.seq_lens, exai(inp.add_info, 'ceids', inp.payload))


# ----------------------------------------------------------------------------------------------------------------------------------------------------------
# Heads


class HeadEncoder(nn.Module):

    def __init__(self, window):
        super().__init__()
        self.n = NormEncoder()
        self.f = nn.Sigmoid()

        self.w = nn.Linear(cns.EMBEDDING_DIM, cns.EMBEDDING_DIM, bias=False)
        #self.w = nn.Parameter(torch.randn(cns.EMBEDDING_DIM)).cuda()

        self.avg_pool = nn.AvgPool1d(kernel_size=cns.TRAIN_WINDOW, stride=cns.TRAIN_WINDOW)
        #self.avg_pool = nn.MaxPool1d(kernel_size=cns.TRAIN_WINDOW, stride=cns.TRAIN_WINDOW)

        self.window = window

    def forward(self, l):
        out_, x = l
        out = out_.payload.contiguous()[:, :-1, :]
        out = self.n(out)

        x = x.payload.view(x.payload.size(0), x.payload.size(1), self.window, -1)
        #x = self.n(self.w(x))
        x = self.n(x)

        out_size = x.size()[:-1]
        out = out.unsqueeze(-2).repeat(1, 1, x.size(-2), 1)
        out = out.view(-1, out.size(-1)).unsqueeze(1)
        x = x.view(-1, x.size(-1)).unsqueeze(2)

        # make dor product
        r = torch.matmul(out, x).unsqueeze(2).view(*out_size)
        r = self.f(r)

        # split
        e_pos, e_neg = r[:, :, :cns.TRAIN_WINDOW], r[:, :, cns.TRAIN_WINDOW:]

        e1 = self.avg_pool(e_pos).squeeze()
        e2 = self.avg_pool(e_neg).squeeze()
        #e1 = (-1) * self.avg_pool(e_pos * (-1)).squeeze()
        #e2 = self.avg_pool(e_neg).squeeze()

        return PaddedBatch(e1, out_.seq_lens, exai(out_.add_info, 'rnn_out', out_.payload)), PaddedBatch(e2, out_.seq_lens, exai(out_.add_info, 'rnn_out', out_.payload))


class HeadClassificationEncoder(nn.Module):

    def __init__(self):
        super().__init__()
        self.w = nn.Linear(cns.RNN_HIDDEN_DIM, cns.N_CEIDS)
        self.f = nn.LogSoftmax(dim=-1)

    def forward(self, x: PaddedBatch):
        out = x.payload.contiguous()
        if not cns.DEPLOY_MODE:
            out = out[:, :-1, :]

        out = self.f(self.w(out))

        return PaddedBatch(out, x.seq_lens if cns.DEPLOY_MODE else x.seq_lens - 1, exai(x.add_info, 'rnn_out', x.payload))


class HeadOrdinalEncoder(nn.Module):

    def __init__(self):
        super().__init__()
        self.w = nn.Linear(cns.RNN_HIDDEN_DIM, cns.N_CEIDS)
        self.f = nn.Softmax(dim=-1)
        self.f1 = nn.Sigmoid()

        self.phi = nn.Linear(cns.RNN_HIDDEN_DIM, cns.TRAIN_WINDOW)

    def forward(self, x: PaddedBatch):
        out = x.payload.contiguous()[:, :-1, :]
        out1 = self.f(self.w(out))

        alpha = 1/(1000 * cns.N_CEIDS)
        tresholds = self.f1(self.phi(out)*0.1) * (1 - alpha) + alpha # to [alpha, 1] interval
        tresholds = torch.sort(tresholds)[0]

        return PaddedBatch(out1, x.seq_lens - 1, exai(exai(x.add_info, 'rnn_out', x.payload), 'tresholds', tresholds))


# ----------------------------------------------------------------------------------------------------------------------------------------------------------
# models


def get_model():
    # Embeding layer
    ceid_embedder = NoisyEmbedding(num_embeddings=cns.N_CEIDS,
                                   embedding_dim=cns.EMBEDDING_DIM,
                                   padding_idx=0,
                                   max_norm=None,
                                   noise_scale=0.05)

    #ceid_embedder = nn.Embedding(cns.N_CEIDS, cns.EMBEDDING_DIM, padding_idx=0)
    enc1 = BaseEncoder(ceid_embedder, ['ceid'])

    # seq encoder
    rnn = RnnEncoder(enc1.output_size())
    #rnn = BertTransformerEncoder(enc1.output_size())

    # first part
    p = nn.Sequential(*[enc1, rnn])

    # second part
    features = [f'ceids_{i + 1}' for i in range(cns.TRAIN_WINDOW * 2)]
    enc2 = BaseEncoder(ceid_embedder, features)

    # combine parts
    f = FModuleList([p, enc2])

    # head
    h = HeadEncoder(window=cns.TRAIN_WINDOW*2)

    return nn.Sequential(f, h)


def get_model_calssification():
    # Embeding layer
    ceid_embedder = NoisyEmbedding(num_embeddings=cns.N_CEIDS,
                                   embedding_dim=cns.EMBEDDING_DIM,
                                   padding_idx=0,
                                   max_norm=None,
                                   noise_scale=0.05)

    # ceid_embedder = nn.Embedding(cns.N_CEIDS, cns.EMBEDDING_DIM, padding_idx=0)
    enc1 = BaseEncoder(ceid_embedder, ['ceid'])

    # seq encoder
    rnn = RnnEncoder(enc1.output_size())
    #rnn = BertTransformerEncoder(enc1.output_size())

    # head
    h = HeadClassificationEncoder()

    return nn.Sequential(enc1, rnn,  h)


def get_model_calssification_with_features():
    # Embeding layer
    ceid_embedder = NoisyEmbedding(num_embeddings=cns.N_CEIDS,
                                   embedding_dim=cns.EMBEDDING_DIM,
                                   padding_idx=0,
                                   max_norm=None,
                                   noise_scale=0.05)

    # ceid_embedder = nn.Embedding(cns.N_CEIDS, cns.EMBEDDING_DIM, padding_idx=0)
    enc1 = BaseEncoder(ceid_embedder, ['ceid'])
    fenc = FeatureEncoder(enc1)

    # seq encoder
    rnn = RnnEncoder(fenc.output_size())
    #rnn = BertTransformerEncoder(fenc.output_size())

    # head
    h = HeadClassificationEncoder()

    return nn.Sequential(fenc, rnn,  h)


def get_model_ordinal_with_features():
    # Embeding layer
    ceid_embedder = NoisyEmbedding(num_embeddings=cns.N_CEIDS,
                                   embedding_dim=cns.EMBEDDING_DIM,
                                   padding_idx=0,
                                   max_norm=None,
                                   noise_scale=0.05)

    # ceid_embedder = nn.Embedding(cns.N_CEIDS, cns.EMBEDDING_DIM, padding_idx=0)
    enc1 = BaseEncoder(ceid_embedder, ['ceid'])
    fenc = FeatureEncoder(enc1)

    # seq encoder
    rnn = RnnEncoder(fenc.output_size())
    #rnn = BertTransformerEncoder(fenc.output_size())

    # head
    h = HeadOrdinalEncoder()

    return nn.Sequential(fenc, rnn,  h)