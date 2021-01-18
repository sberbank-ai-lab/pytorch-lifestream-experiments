"""
Creates data loaders for contrastive learning
Each record pass throw the sampler and splits into views of the main record
X - is feature_arrays, sub-samples of main event sequences
Y - is Id of main record, for positive item identification

    Input:        -> Output
dict of arrays    ->  X,                 Y
client_1          -> client_1_smpl_1, client_1_id
                     client_1_smpl_2, client_1_id
                     client_1_smpl_3, client_1_id
client_2          -> client_2_smpl_1, client_2_id
                     client_2_smpl_2, client_2_id
                     client_2_smpl_3, client_2_id
"""
import logging

import numpy as np
import pytorch_lightning as pl

from pyhocon.config_parser import ConfigFactory
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from dltranz.data_load import padded_collate, IterableAugmentations, IterableChain, augmentation_chain
from dltranz.data_load.augmentations.dropout_trx import DropoutTrx
from dltranz.data_load.augmentations.seq_len_limit import SeqLenLimit
from dltranz.data_load.iterable_processing.category_size_clip import CategorySizeClip
from dltranz.data_load.iterable_processing.feature_filter import FeatureFilter
from dltranz.data_load.iterable_processing.feature_type_cast import FeatureTypeCast
from dltranz.data_load.iterable_processing.id_filter import IdFilter
from dltranz.data_load.iterable_processing.iterable_shuffle import IterableShuffle
from dltranz.data_load.iterable_processing.seq_len_filter import SeqLenFilter
from dltranz.data_load.iterable_processing.target_extractor import TargetExtractor
from dltranz.data_load.list_splitter import ListSplitter
from dltranz.data_load.parquet_dataset import ParquetDataset, ParquetFiles
from dltranz.data_load.partitioned_dataset import PartitionedDataset, PartitionedDataFiles
from dltranz.metric_learn.dataset import split_strategy, nested_list_to_flat_with_collate
from dltranz.metric_learn.dataset.splitting_dataset import IterableSplittingDataset, MapSplittingDataset

logger = logging.getLogger(__name__)


class ColesDataModuleTrain(pl.LightningDataModule):
    def __init__(self, conf, pl_module):
        super().__init__()

        self._type = conf['type']
        assert self._type in ('map', 'iterable')

        self.setup_conf = conf['setup']
        self.train_conf = conf['train']
        self.valid_conf = conf['valid']

        self.col_id = self.setup_conf['col_id']
        self.category_max_size = pl_module.seq_encoder.category_max_size

        self.train_dataset = None
        self.valid_dataset = None
        self._train_ids = None
        self._valid_ids = None

    def prepare_data(self):
        if 'dataset_files' in self.setup_conf:
            self.setup_iterable_files()
        elif 'dataset_parts' in self.setup_conf:
            self.setup_iterable_parts()
        else:
            raise AttributeError(f'Missing dataset definition. One of `dataset_parts` or `dataset_files` expected')

        if self._type == 'map':
            self.setup_map()

    def setup_iterable_files(self):
        if self.setup_conf['split_by'] == 'files':
            data_files = ParquetFiles(self.setup_conf['dataset_files.data_path']).data_files

            splitter = ListSplitter(
                data_files,
                valid_size=self.setup_conf['valid_size'],
                seed=self.setup_conf['valid_split_seed'],
            )
            logger.info(f'Prepared splits: '
                        f'{len(splitter.train)} files in train, {len(splitter.valid)} files in valid')

            self.train_dataset = ParquetDataset(
                splitter.train,
                post_processing=IterableChain(*self.build_iterable_processing('train')),
                shuffle_files=True if self._type == 'iterable' else False,
            )
            self.valid_dataset = ParquetDataset(
                splitter.valid,
                post_processing=IterableChain(*self.build_iterable_processing('valid')),
                shuffle_files=False,
            )
        elif self.setup_conf['split_by'] == 'rows':
            data_files = ParquetFiles(self.setup_conf['dataset_files.data_path']).data_files

            self.split_by_rows(data_files)

            self.train_dataset = ParquetDataset(
                data_files,
                post_processing=IterableChain(*self.build_iterable_processing('train')),
                shuffle_files=True if self._type == 'iterable' else False,
            )
            self.valid_dataset = ParquetDataset(
                data_files,
                post_processing=IterableChain(*self.build_iterable_processing('valid')),
                shuffle_files=False,
            )

        else:
            raise AttributeError(f'Unknown split strategy: {self.setup_conf.split_by}')

    def setup_iterable_parts(self):
        data_files = PartitionedDataFiles(**self.setup_conf['dataset_parts'])

        if self.setup_conf['split_by'] == 'hash_id':
            splitter = ListSplitter(
                data_files.hash_parts,
                valid_size=self.setup_conf['valid_size'],
                seed=self.setup_conf['valid_split_seed'],
            )
            logger.info(f'Prepared splits: '
                        f'{len(data_files.dt_parts)} parts in dt_train, {len(data_files.dt_parts)} parts in dt_valid, '
                        f'{len(splitter.train)} parts in hash_train, {len(splitter.valid)} parts in hash_valid')

            self.train_dataset = PartitionedDataset(
                data_files.data_path, data_files.dt_parts, splitter.train, col_id=self.col_id,
                post_processing=IterableChain(*self.build_iterable_processing('train')),
                shuffle_files=True if self._type == 'iterable' else False,
            )
            self.valid_dataset = PartitionedDataset(
                data_files.data_path, data_files.dt_parts, splitter.valid, col_id=self.col_id,
                post_processing=IterableChain(*self.build_iterable_processing('valid')),
                shuffle_files=False,
            )
        elif self.setup_conf['split_by'] == 'rows':
            raise NotImplementedError("Split by rows aren't supported for partitioned dataset")
        else:
            raise AttributeError(f'Unknown split strategy: {self.setup_conf.split_by}')

    def split_by_rows(self, data_files):
        """
        This code is correct. Just make reproducible shuffle and split, if needed

        Args:
            data_files:

        Returns:

        """
        ds = ParquetDataset(data_files, post_processing=SeqLenFilter(min_seq_len=self.train_conf['min_seq_len']))
        list_ids = [row for row in ds]
        list_ids = shuffle_client_list_reproducible(ConfigFactory.from_dict({
            'dataset': {
                'client_list_shuffle_seed': 42,
                'col_id': self.col_id,
            }
        }), list_ids)
        list_ids = [row[self.col_id] for row in list_ids]

        valid_ix = np.arange(len(list_ids))
        valid_ix = np.random.RandomState(42).choice(valid_ix, size=int(len(list_ids) * self.setup_conf['valid_size']), replace=False)
        valid_ix = set(valid_ix.tolist())

        logger.info(f'Loaded {len(list_ids)} rows. Split in progress...')
        train_data = [rec for i, rec in enumerate(list_ids) if i not in valid_ix]
        valid_data = [rec for i, rec in enumerate(list_ids) if i in valid_ix]

        logger.info(f'Train data len: {len(train_data)}, Valid data len: {len(valid_data)}')
        self._train_ids = train_data
        self._valid_ids = valid_data

    def build_iterable_processing(self, part):
        if 'dataset_files' in self.setup_conf and self.setup_conf['split_by'] == 'rows':
            if part == 'train':
                yield IdFilter(id_col=self.col_id, relevant_ids=self._train_ids)
            else:
                yield IdFilter(id_col=self.col_id, relevant_ids=self._valid_ids)

        if part == 'train':
            yield SeqLenFilter(min_seq_len=self.train_conf['min_seq_len'])

        yield FeatureTypeCast({self.col_id: int})
        yield TargetExtractor(target_col=self.col_id)
        yield FeatureFilter(drop_non_iterable=True)
        yield CategorySizeClip(self.category_max_size)

        if self._type == 'iterable':
            # all processing in single chain
            if part == 'train':
                yield IterableShuffle(self.train_conf['buffer_size'])

            if part == 'train':
                split_strategy_conf = self.train_conf['split_strategy']
            else:
                split_strategy_conf = self.valid_conf['split_strategy']
            yield IterableSplittingDataset(split_strategy.create(**split_strategy_conf))

            yield IterableAugmentations(*self.build_augmentations(part))

    def build_augmentations(self, part):
        if part == 'train':
            if 'max_seq_len' in self.train_conf:
                yield SeqLenLimit(self.train_conf['max_seq_len'])
            yield DropoutTrx(self.train_conf['trx_dropout'])

        if part == 'valid':
            if 'max_seq_len' in self.valid_conf:
                yield SeqLenLimit(self.valid_conf['max_seq_len'])

    def setup_map(self):
        self.train_dataset = list(tqdm(iter(self.train_dataset)))
        logger.info(f'Loaded {len(self.train_dataset)} for train')
        self.valid_dataset = list(tqdm(iter(self.valid_dataset)))
        logger.info(f'Loaded {len(self.valid_dataset)} for valid')

        self.train_dataset = MapSplittingDataset(
            base_dataset=self.train_dataset,
            splitter=split_strategy.create(**self.train_conf['split_strategy']),
            a_chain=augmentation_chain(*self.build_augmentations('train')),
        )
        self.valid_dataset = MapSplittingDataset(
            base_dataset=self.valid_dataset,
            splitter=split_strategy.create(**self.valid_conf['split_strategy']),
            a_chain=augmentation_chain(*self.build_augmentations('valid')),
        )

    def train_dataloader(self):
        collate_fn = padded_collate
        if self._type != 'iterable':
            collate_fn = nested_list_to_flat_with_collate(collate_fn)

        return DataLoader(
            dataset=self.train_dataset,
            collate_fn=collate_fn,
            shuffle=False if self._type == 'iterable' else True,
            num_workers=self.train_conf['num_workers'],
            batch_size=self.train_conf['batch_size'],
        )

    def val_dataloader(self):
        collate_fn = padded_collate
        if self._type != 'iterable':
            collate_fn = nested_list_to_flat_with_collate(collate_fn)

        return DataLoader(
            dataset=self.valid_dataset,
            collate_fn=collate_fn,
            num_workers=self.valid_conf['num_workers'],
            batch_size=self.valid_conf['batch_size'],
        )