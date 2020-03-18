import torch

from dltranz.transf_seq_encoder import DateTimeEncoding
from dltranz.trx_encoder import PaddedBatch


class _TestTrxEncoder:
    def __call__(self, x: PaddedBatch):
        return PaddedBatch(x.payload['data'], x.seq_lens)


def get_data():
    return PaddedBatch(
        payload={
            'data': torch.arange(3 * 4 * 10).view(3, 4, 10),
            'event_time': torch.arange(4).unsqueeze(0).repeat(3, 1),
        },
        length=torch.tensor([1, 2, 3])
    )


def test_date_time_encoding_exp():
    encoder = DateTimeEncoding(_TestTrxEncoder(), hidden_size=10, div_term_mode='exp')
    out = encoder(get_data())
    print(out.payload)


def test_date_time_encoding_time():
    encoder = DateTimeEncoding(_TestTrxEncoder(), hidden_size=10, div_term_mode='time')
    out = encoder(get_data())
    print(out.payload)
