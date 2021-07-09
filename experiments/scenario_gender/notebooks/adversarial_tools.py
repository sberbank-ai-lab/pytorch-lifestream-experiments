from itertools import cycle, product

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
import torch
import random
import pytorch_lightning as pl

from dltranz.data_load import IterableChain, IterableAugmentations, padded_collate
from dltranz.data_load.augmentations.build_augmentations import build_augmentations
from dltranz.data_load.data_module.coles_data_module import coles_collate_fn
from dltranz.data_load.iterable_processing.category_size_clip import CategorySizeClip
from dltranz.data_load.iterable_processing.feature_filter import FeatureFilter
from dltranz.data_load.iterable_processing.feature_type_cast import FeatureTypeCast
from dltranz.data_load.iterable_processing.id_filter import IdFilter
from dltranz.data_load.iterable_processing.seq_len_filter import SeqLenFilter
from dltranz.data_load.iterable_processing.target_join import TargetJoin
from dltranz.data_load.iterable_processing_dataset import IterableProcessingDataset
from dltranz.data_load.list_splitter import ListSplitter
from dltranz.data_load.parquet_dataset import ParquetFiles, ParquetDataset
from dltranz.lightning_modules.coles_module import CoLESModule
from dltranz.metric_learn.dataset import split_strategy
from dltranz.metric_learn.dataset.splitting_dataset import MapSplittingDataset


class PretrainedAttack(IterableProcessingDataset):
    def __init__(self, col_id, attacks):
        super().__init__()

        self.col_id = col_id
        self.attacks = attacks

    def process(self, x):
        _id = x[self.col_id]
        a = self.attacks[_id]
        for k, v in a.items():
            x[k] = torch.cat([x[k], torch.tensor(v)])
        return x

class GreadyAttacker:
    def __init__(self, conf_attack, is_sample_mcc=True, mcc_freq=None):
        self._src = None

        self.mcc_len = conf_attack['mcc_len']
        self.max_mcc = conf_attack['max_mcc']
        self.sample_rate = conf_attack['sample_rate']

        self.is_sample_mcc = is_sample_mcc
        if is_sample_mcc:
            self.tr_for_mcc = {}
            for m in range(2, self.max_mcc):
                t = mcc_freq.loc[m]['tr_type_freq']
                t = t.index[torch.multinomial(
                    torch.from_numpy(t.values), 1000, replacement=True).numpy()].values.tolist()
                self.tr_for_mcc[m] = cycle(t)

            self.amount_dist = mcc_freq[['a_min', 'a_max']].to_dict(orient='index')

    def __call__(self, src):
        self._src = src
        return iter(self)

    def __iter__(self):
        for rec in self._src:
            for t in self.process(rec):
                yield t

    def process(self, x):
        # original record
        rec, d = x
        yield rec, {'id': d['id'], 'y': d['y'], 'new_trx': None, 'new_trx_len': 0}

        if not self.is_sample_mcc:
            return

        mcc_len = self.mcc_len
        mcc_product = list(product(*[range(2, self.max_mcc) for _ in range(mcc_len)]))
        n = int(len(mcc_product) * self.sample_rate)
        if n <= 1:
            print(n)
        sel_ids = set(np.random.choice(len(mcc_product), n, replace=False).tolist())
        for i, new_mcc in enumerate(mcc_product):
            if i not in sel_ids:
                continue

            new_rec = {}
            new_rec['mcc_code'] = torch.cat([rec['mcc_code'], torch.tensor(new_mcc)])

            v = rec['event_time'][-1].item()
            new_event_time = [v + 1, v + 2]
            new_rec['event_time'] = torch.cat([rec['event_time'], torch.tensor(new_event_time)])

            new_tr_type = []
            new_amount = []
            for m in new_mcc:
                t = next(self.tr_for_mcc[m])
                new_tr_type.append(t)

                ix = (rec['mcc_code'] == m) & (rec['tr_type'] == t)
                if ix.sum() == 0:
                    # no such trx for this client, sample similar trx
                    v = self.amount_dist[(m, t)]
                    a = random.random() * (v['a_max'] - v['a_min']) + v['a_min']
                else:
                    # take amount from history
                    v = rec['amount'][ix]
                    a = v[random.choice(range(len(v)))].item()
                new_amount.append(a)

            new_rec['tr_type'] = torch.cat([rec['tr_type'], torch.tensor(new_tr_type)])
            new_rec['amount'] = torch.cat([rec['amount'], torch.tensor(new_amount)])

            yield new_rec, {'id': d['id'], 'y': d['y'], 'new_trx': {
                'mcc_code': list(new_mcc),
                'tr_type': new_tr_type,
                'amount': new_amount,
                'event_time': new_event_time,
            }, 'new_trx_len': len(new_mcc)}


class AdversarialAttackDataModule(pl.LightningDataModule):
    def __init__(self, conf, seq_encoder, mcc_freq, *files):
        super().__init__()

        self.setup_conf = conf['data_module']
        self.test_conf = conf['attack']
        self.attack_conf = conf['attack']
        self.mcc_freq = mcc_freq

        self.test_files = files

        self.col_id = self.setup_conf['col_id']
        self.col_id_dtype = {
            'str': str,
            'int': int,
        }[self.setup_conf['col_id_dtype']]
        self.col_target = self.setup_conf['col_target']
        self.y_cast = int
        self.category_names = seq_encoder.category_names
        self.category_names.add('event_time')
        self.category_max_size = seq_encoder.category_max_size

    def prepare_data(self):
        self.load_target()
        self.setup_iterable_files()

    def load_target(self):
        df = pd.read_csv(self.setup_conf['target_file_path'])
        df[self.col_id] = df[self.col_id].astype(self.col_id_dtype)
        df[self.col_target] = df[self.col_target].astype(self.y_cast)
        self._targets = df.set_index(self.col_id)[self.col_target].to_dict()

    def setup_iterable_files(self):
        self.test_dataset = ParquetDataset(
            self.test_files,
            post_processing=IterableChain(*self.build_iterable_processing('test')),
            shuffle_files=False,
        )
        self.history_dataset = ParquetDataset(
            self.test_files,
            post_processing=IterableChain(*self.build_iterable_processing('history')),
            shuffle_files=False,
        )

    def build_iterable_processing(self, part):
        yield FeatureTypeCast({self.col_id: self.col_id_dtype})
        yield IdFilter(id_col=self.col_id, relevant_ids=list(self._targets.keys()))

        yield TargetJoin(self.col_id, self._targets, self.y_cast)
        yield TargetToDict(self.col_id)
        yield FeatureFilter(keep_feature_names=self.category_names)
        yield CategorySizeClip(self.category_max_size)
        yield IterableAugmentations(self.build_augmentations(part))

        if part == 'test':
            yield GreadyAttacker(self.attack_conf, is_sample_mcc=True, mcc_freq=self.mcc_freq)
        else:
            yield GreadyAttacker(self.attack_conf, is_sample_mcc=False)

    def build_augmentations(self, part):
        return build_augmentations(self.attack_conf['augmentations'])

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_dataset,
            collate_fn=padded_collate,
            shuffle=False,
            num_workers=self.attack_conf['num_workers'],
            batch_size=self.attack_conf['batch_size'],
            drop_last=False,
        )


class TargetToDict(IterableProcessingDataset):
    def __init__(self, col_id):
        super().__init__()
        
        self.col_id = col_id
    
    def __iter__(self):
        for x, y in self._src:
            _id = x[self.col_id]
            yield x, {'id': _id, 'y': y}


class Adversarial4PartSplitter:
    def __init__(self, conf):
        self.conf = conf

        train_data_files = ParquetFiles(self.conf['dataset_files.train_data_path']).data_files
        test_data_files = ParquetFiles(self.conf['dataset_files.test_data_path']).data_files

        train_data_files = np.array(train_data_files)
        kf = KFold(n_splits=self.conf['cv_folds'], shuffle=True, random_state=self.conf['split_seed'])

        self.folds = {}
        for i, (i_train, i_valid) in enumerate(kf.split(train_data_files)):
            target_subst_spliter = ListSplitter(
                train_data_files[i_train].tolist(), valid_size=self.conf['substitute_part'],
                seed=self.conf['split_seed'])

            self.folds[i] = {
                'test': test_data_files,
                'valid': train_data_files[i_valid].tolist(),
                'target': target_subst_spliter.train,
                'substitute': target_subst_spliter.valid,
            }


def train_coles_model(fold_id, files, conf):
    target_module = CoLESModule(conf['params'])
    valid_n = int(np.ceil(len(files) * conf['valid_ratio']))
    train_files, valid_files = files[:-valid_n], files[-valid_n:]

    category_names = target_module.seq_encoder.category_names
    category_names.add('event_time')
    train_dataset = ParquetDataset(
        train_files,
        post_processing=IterableChain(
            SeqLenFilter(min_seq_len=conf['train']['min_seq_len']),
            FeatureFilter(keep_feature_names=category_names),
            CategorySizeClip(target_module.seq_encoder.category_max_size),
        ),
        shuffle_files=False,
    )
    valid_dataset = ParquetDataset(
        valid_files,
        post_processing=IterableChain(
            FeatureFilter(keep_feature_names=category_names),
            CategorySizeClip(target_module.seq_encoder.category_max_size),
        ),
        shuffle_files=False,
    )

    train_dataset = list(train_dataset)
    valid_dataset = list(valid_dataset)

    train_dataset = MapSplittingDataset(
        base_dataset=train_dataset,
        splitter=split_strategy.create(**conf['train']['split_strategy']),
        a_chain=lambda x: x,
    )
    valid_dataset = MapSplittingDataset(
        base_dataset=valid_dataset,
        splitter=split_strategy.create(**conf['valid']['split_strategy']),
        a_chain=lambda x: x,
    )

    train_dataloader = DataLoader(
        dataset=train_dataset,
        collate_fn=coles_collate_fn,
        shuffle=True,
        num_workers=conf['train']['num_workers'],
        batch_size=conf['train']['batch_size'],
    )
    val_dataloader = DataLoader(
        dataset=valid_dataset,
        collate_fn=coles_collate_fn,
        num_workers=conf['valid']['num_workers'],
        batch_size=conf['valid']['batch_size'],
    )

    trainer = pl.Trainer(
        logger=pl.loggers.TensorBoardLogger(
            save_dir='lightning_logs',
            name=f'[{fold_id}]: coles_model',
        ),
        **conf['trainer']
    )
    trainer.fit(target_module, train_dataloader=train_dataloader, val_dataloaders=val_dataloader)

    target_model = target_module.seq_encoder
    target_model.is_reduce_sequence = True
    return target_model


def get_mcc_freq(files, seq_encoder):
    category_names = seq_encoder.category_names
    category_names.add('event_time')
    dataset = ParquetDataset(
        files,
        post_processing=IterableChain(
            FeatureFilter(keep_feature_names=category_names),
            CategorySizeClip(seq_encoder.category_max_size),
        ),
        shuffle_files=False,
    )

    df_mcc_freq = [[], [], []]
    for rec in dataset:
        for i, k in enumerate(['mcc_code', 'tr_type', 'amount']):
            df_mcc_freq[i].append(rec[k].numpy())

    df_mcc_freq = pd.DataFrame({
        'mcc_code': np.concatenate(df_mcc_freq[0]),
        'tr_type': np.concatenate(df_mcc_freq[1]),
        'amount': np.concatenate(df_mcc_freq[2]),
    })

    df_mcc_freq1 = df_mcc_freq
    df_mcc_freq = df_mcc_freq.groupby('mcc_code')
    df_mcc_freq = pd.concat([
        df_mcc_freq.apply(lambda x: x['tr_type'].value_counts(normalize=True).rename('tr_type_freq'))
    ], axis=1)

    df_mcc_freq = pd.concat([
        df_mcc_freq,
        df_mcc_freq1.groupby(['mcc_code', 'tr_type']).apply(lambda x: x['amount'].quantile([0.25, 0.75]))
        .rename(columns={0.25: 'a_min', 0.75: 'a_max'}),
    ], axis=1)
    return df_mcc_freq
