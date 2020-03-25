# coding: utf-8
import logging
import os
import sys
import torch
import torch.nn as nn

from torch.autograd import Function

from dltranz.agg_feature_model import AggFeatureModel
from dltranz.transf_seq_encoder import TransformerSeqEncoder, DateTimeEncoding

script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(script_dir, '../..'))

from dltranz.seq_encoder import RnnEncoder, LastStepEncoder, PerTransTransf, FirstStepEncoder
from dltranz.trx_encoder import TrxEncoder

logger = logging.getLogger(__name__)


# TODO: is the same as dltranz.seq_encoder.NormEncoder
class L2Normalization(nn.Module):
    def __init__(self):
        super(L2Normalization, self).__init__()

    def forward(self, input):
        return input.div(torch.norm(input, dim=1).view(-1, 1))


class Binarization(Function):
    @staticmethod
    def forward(self, x):
        q = (x>0).float()
        return  (2*q - 1)

    @staticmethod
    def backward(self, gradOutput):
        return gradOutput

binary = Binarization.apply


class BinarizationLayer(nn.Module):
    def __init__(self, hs_from, hs_to):
        super(BinarizationLayer, self).__init__()
        self.linear = nn.Linear(hs_from, hs_to, bias = False)

    def forward(self, x):
        return binary(self.linear(x))


def rnn_model(params):
    encoder_layers = [
        TrxEncoder(params['trx_encoder']),
        RnnEncoder(TrxEncoder.output_size(params['trx_encoder']), params['rnn']),
        LastStepEncoder(),
    ]

    layers = [torch.nn.Sequential(*encoder_layers)]
    if 'projection_head' in params:
        logger.info('projection_head included')
        layers.extend([
            torch.nn.Linear(params['rnn.hidden_size'], params['rnn.hidden_size']),
            torch.nn.ReLU(),
            torch.nn.Linear(params['rnn.hidden_size'], params['projection_head.output_size']),
        ])
    if params['use_normalization_layer']:
        layers.append(L2Normalization())
        logger.info('L2Normalization included')
    m = torch.nn.Sequential(*layers)
    return m


class Splitter(torch.nn.Module):
    def __init__(self, embedding_size, strategy, transform_mu_out=None):
        super().__init__()

        self.strategy = strategy
        if self.strategy == 'split':
            pass
        elif self.strategy == 'transform':
            self.l1 = torch.nn.Linear(embedding_size, transform_mu_out * 2)
        elif self.strategy == 'learn_sigma':
            self.l1 = torch.nn.Linear(embedding_size, transform_mu_out)
            self.logvar = torch.nn.Parameter(torch.randn(1, 1))
        else:
            raise NotImplementedError(f'Unknown strategy "{self.strategy}"')

    def forward(self, x):
        if self.strategy == 'split':
            d = x.size()[1] // 2
            mu, logvar = x[:, :d], x[:, d:]
            out = torch.cat([mu, logvar], dim=1)
        elif self.strategy == 'transform':
            out = self.l1(x)
        elif self.strategy == 'learn_sigma':
            x = self.l1(x)
            out = torch.cat([x, self.logvar.expand(*x.size())], dim=1)
        else:
            raise NotImplementedError(f'Unknown strategy "{self.strategy}"')

        return out


def transformer_model(params):
    p = TrxEncoder(params['trx_encoder'])
    trx_size = TrxEncoder.output_size(params['trx_encoder'])
    enc_input_size = params['transf']['input_size']
    if enc_input_size != trx_size:
        inp_reshape = PerTransTransf(trx_size, enc_input_size)
        p = torch.nn.Sequential(p, inp_reshape)

    if 'date_time_encoder' in params:
        p = DateTimeEncoding(p, **params['date_time_encoder'])
        logger.info('DateTimeEncoding included')

    e = TransformerSeqEncoder(enc_input_size, params['transf'])
    l = FirstStepEncoder()
    encoder_layers = [p, e, l]
    if 'splitter' in params:
        encoder_layers.append(Splitter(embedding_size=enc_input_size, **params['splitter']))
        logger.info('Splitter included')
    layers = [torch.nn.Sequential(*encoder_layers)]

    if params['use_normalization_layer']:
        layers.append(L2Normalization())
        logger.info('L2Normalization included')
    m = torch.nn.Sequential(*layers)
    return m


def agg_feature_model(params):
    layers = [
        torch.nn.Sequential(
            AggFeatureModel(params['trx_encoder']),
        ),
        torch.nn.BatchNorm1d(AggFeatureModel.output_size(params['trx_encoder'])),
    ]
    if params['use_normalization_layer']:
        layers.append(L2Normalization())
    m = torch.nn.Sequential(*layers)
    return m


def ml_model_by_type(model_type):
    model = {
        'rnn': rnn_model,
        'transf': transformer_model,
        'agg_features': agg_feature_model,
    }[model_type]
    return model


class ModelEmbeddingEnsemble(nn.Module):
    def __init__(self, submodels):
        super(ModelEmbeddingEnsemble, self).__init__()
        self.models = nn.ModuleList(submodels)

    def forward(self, *args):
        out = torch.cat([m(*args) for m in self.models], dim=1)
        return out
