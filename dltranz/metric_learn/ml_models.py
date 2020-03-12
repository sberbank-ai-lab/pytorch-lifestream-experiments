# coding: utf-8
import logging
import os
import sys
import torch
import torch.nn as nn

from torch.autograd import Function

from dltranz.agg_feature_model import AggFeatureModel
from dltranz.transf_seq_encoder import TransformerSeqEncoder

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
        return input.div(torch.norm(input, dim=1).view(-1, 1) + 1e-9)


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


class ATan(torch.nn.Module):
    def forward(self, x):
        return torch.log1p(x + 1e-9) - torch.log1p(-x + 1e-9)


def rnn_model(params):
    encoder_layers = [
        TrxEncoder(params['trx_encoder']),
        RnnEncoder(TrxEncoder.output_size(params['trx_encoder']), params['rnn']),
        LastStepEncoder(),
    ]
    if params['use_atan']:
        encoder_layers.append(
            ATan(),
        )
        logger.info('+ ATan()')
    if params['use_split']:
        encoder_layers.append(
            torch.nn.Linear(params['rnn.hidden_size'], params['rnn.hidden_size'] * 2),
        )
        logger.info('+ LinearSplit')
    layers = [torch.nn.Sequential(*encoder_layers)]

    if params['use_normalization_layer']:
        layers.append(L2Normalization())
        logger.info('L2Normalization included')
    m = torch.nn.Sequential(*layers)
    return m


def transformer_model(params):
    p = TrxEncoder(params['trx_encoder'])
    trx_size = TrxEncoder.output_size(params['trx_encoder'])
    enc_input_size = params['transf']['input_size']
    if enc_input_size != trx_size:
        inp_reshape = PerTransTransf(trx_size, enc_input_size)
        p = torch.nn.Sequential(p, inp_reshape)

    e = TransformerSeqEncoder(enc_input_size, params['transf'])
    l = FirstStepEncoder()
    layers = [p, e, l]

    if params['use_normalization_layer']:
        layers.append(L2Normalization())
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
