import logging
import math
import torch
from torch import nn

from dltranz.trx_encoder import PaddedBatch

logger = logging.getLogger(__name__)

try:
    from torch.nn import TransformerEncoder, TransformerEncoderLayer, LayerNorm
except ImportError:
    logger.error('Can not import Transformers')


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
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


class DateTimeEncoding(nn.Module):
    """Use it as preliminary layer before TransformerSeqEncoder

    """
    def __init__(self, trx_encoder, hidden_size, div_term_mode, positions, time_shift=0.0, dropout=0.1):
        super(DateTimeEncoding, self).__init__()
        self.trx_encoder = trx_encoder
        self.dropout = nn.Dropout(p=dropout)
        self.positions = positions
        assert positions in ('time', 'ordinal')
        self.time_shift = time_shift

        if div_term_mode == 'exp':
            # 1 day is minimal period
            div_term = 2 * math.pi * torch.exp(
                torch.arange(0, hidden_size, 2).float() * (-math.log(10000.0) / hidden_size))
        elif div_term_mode == 'time':
            dim = hidden_size // 2
            div_term = torch.zeros(dim)
            periods = torch.tensor([1 / 24, 1, 7, 14, 30, 31, 365, 366, 1000])
            if dim > len(periods):
                dim = len(periods)
            div_term[:len(periods)] = 2 * math.pi / periods[:dim]
        else:
            raise NotImplementedError(f'Unknown div_term_mode "{div_term_mode}"')

        div_term = div_term.unsqueeze(0).unsqueeze(0)
        self.register_buffer('div_term', div_term)

    def forward(self, x: PaddedBatch):
        """
        z: B, T, H
        """
        z = self.trx_encoder(x)  # Z: B, T, H

        # event_time = x.payload['event_time'].float()
        # event_time = event_time.unsqueeze(2)  # B, T -> B, T, 1

        if self.positions == 'time':
            event_time = x.payload['event_time'].float()
            event_time = event_time.unsqueeze(2)  # B, T -> B, T, 1
        else:
            event_time = x.payload['event_time']
            event_time = torch.arange(event_time.size()[1], device=event_time.device).float()
            event_time = event_time.unsqueeze(0).unsqueeze(2)  # B, T -> B, T, 1

        if self.training and self.time_shift > 0.0:
            event_time += torch.rand(event_time.size()[0], 1, 1, device=event_time.device) * self.time_shift

        pe = torch.cat([
            torch.sin(event_time * self.div_term),
            torch.cos(event_time * self.div_term),
        ], dim=2)
        out = z.payload + pe
        out = self.dropout(out)
        return PaddedBatch(out, x.seq_lens)


class TransformerSeqEncoder(nn.Module):
    def __init__(self, input_size, params):
        super().__init__()

        self.shared_layers = params['shared_layers']
        self.n_layers = params['n_layers']
        self.use_after_mask = params['use_after_mask']
        self.use_src_key_padding_mask = params['use_src_key_padding_mask']
        self.use_positional_encoding = params['use_positional_encoding']

        self.starter = torch.nn.Parameter(torch.randn(1, 1, input_size)) if params['train_starter'] else None

        self.enc_layer = TransformerEncoderLayer(
            d_model=input_size,
            nhead=params['n_heads'],
            dim_feedforward=params['dim_hidden'],
            dropout=params['dropout'])

        enc_norm = LayerNorm(input_size)
        self.enc = TransformerEncoder(self.enc_layer, params['n_layers'], enc_norm)
        self.pe = PositionalEncoding(max_len=params['max_seq_len'], d_model=input_size, dropout=params['dropout'])

    @staticmethod
    def generate_square_subsequent_mask(sz):
        r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, x):
        batch_size = x.payload.size()[0]
        seq_len_max = x.payload.size()[1]
        x_t = torch.transpose(x.payload, 0, 1)

        if self.starter is not None:
            x_t = torch.cat([self.starter.expand(1, batch_size, -1), x_t], dim=0)

        if self.use_after_mask:
            mask = self.generate_square_subsequent_mask(x_t.size(0)).to(x_t.device)
        else:
            mask = None

        if self.use_src_key_padding_mask:
            src_key_padding_mask = torch.stack([torch.BoolTensor([False] * l + [True] * (seq_len_max - l))
                                                for l in x.seq_lens]).to(x.payload.device)
        else:
            src_key_padding_mask = None

        if self.use_positional_encoding:
            x_t = self.pe(x_t)

        if not self.shared_layers:
            out = self.enc(x_t, mask=mask, src_key_padding_mask=src_key_padding_mask)
        else:
            out = x_t
            for _ in range(self.n_layers):
                out = self.enc_layer(out, mask=mask, src_key_padding_mask=src_key_padding_mask)

        out = torch.transpose(out, 0, 1)

        return PaddedBatch(out, x.seq_lens)
