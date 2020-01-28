import argparse
from glob import glob
import logging
import os
import itertools

import pandas as pd


logger = logging.getLogger(__name__)


def block_iterator(iterator, size):
    bucket = list()
    for e in iterator:
        bucket.append(e)
        if len(bucket) >= size:
            yield bucket
            bucket = list()
    if bucket:
        yield bucket

def cycle_block_iterator(iterator, size):
    return block_iterator(itertools.cycle(iterator), size)

class ListSubset:
    def __init__(self, delegate, idx_to_take):
        self.delegate = delegate
        self.idx_to_take = idx_to_take

    def __len__(self):
        return len(self.idx_to_take)

    def __getitem__(self, idx):
        return self.delegate[self.idx_to_take[idx]]

    def __iter__(self):
        for i in self.idx_to_take:
            yield self.delegate[i]


def init_logger(name, level=logging.INFO, filename=None):
    logger = logging.getLogger(name)
    logger.setLevel(level)

    stream_handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s %(levelname)-7s %(name)-20s   : %(message)s")
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if filename is not None:
        file_handler = logging.FileHandler(filename, mode='w')
        formatter = logging.Formatter("%(asctime)s %(levelname)-7s %(name)-20s   : %(message)s")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)


def get_conf(args=None):
    import sys
    import os
    from pyhocon import ConfigFactory

    p = argparse.ArgumentParser()
    p.add_argument('-c', '--conf', nargs='+')
    args, overrides = p.parse_known_args(args)

    logger.info(f'args: {args}, overrides: {overrides}')

    init_conf = f"script_path={os.path.dirname(os.path.abspath(sys.argv[0]))}"
    file_conf = ConfigFactory.parse_string(init_conf)

    if args is not None and args.conf is not None:
        for name in args.conf:
            logger.info(f'Load config from "{name}"')
            file_conf = ConfigFactory.parse_file(name, resolve=False).with_fallback(file_conf, resolve=False)

    overrides = ','.join(overrides)
    over_conf = ConfigFactory.parse_string(overrides)
    conf = over_conf.with_fallback(file_conf)

    return conf


def get_data_files(params):
    path_wc = params['path_wc']

    if 'data_path' in params:
        path_wc = os.path.join(params['data_path'], path_wc)

    files = glob(path_wc)
    logger.info(f'Found {len(files)} files in "{path_wc}"')

    max_files = params.get('max_files', None)

    if max_files is not None:
        if type(max_files) is int:
            files = files[:max_files]
            logger.info(f'First {len(files)} files are available')
        elif type(max_files) is float:
            max_files = int(max_files * len(files))
            files = files[:max_files]
            logger.info(f'First {len(files)} files are available')
        else:
            raise AttributeError(f'Wrong type of `dataset.max_files`: {type(max_files)}')
    else:
        logger.info(f'All {len(files)} files are available')
    return sorted(files)


def plot_arrays(a, b, title=None):
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib import patches

    def plot_a(x, top_s, offset, **params):
        plt.plot((x, x), (top_s * -0.05, top_s * offset), alpha=0.6, **params)
        plt.text(x, top_s * offset * 1.25, str(x))

    x_min = min(a + b)
    x_max = max(a + b)

    x_len = x_max - x_min

    x_min -= x_len * 0.05
    x_max += x_len * 0.05

    fig = plt.figure(figsize=(12, 3))
    ax = fig.add_subplot(1, 1, 1)
    ax.spines['bottom'].set_position('center')
    ax.spines['top'].set_color(None)
    ax.spines['left'].set_color(None)
    ax.spines['right'].set_color(None)
    ax.yaxis.set_visible(False)

    plt.xlim(x_min, x_max)
    plt.ylim(-0.3, 0.3)
    for x in a:
        plot_a(x, 1, offset=0.2, color='darkblue', linestyle='-', linewidth=4)
    plot_a(np.mean(a).round(5), 1, offset=0.14, color='darkblue', linestyle=':', linewidth=2)
    for x in b:
        plot_a(x, -1, offset=0.2, color='darkgreen', linestyle='-', linewidth=4)
    plot_a(np.mean(b).round(5), -1, offset=0.14, color='darkgreen', linestyle=':', linewidth=2)

    if len(a) >= 3:
        _mean = np.mean(a)
        _std = np.std(a)
        rect = patches.Rectangle((_mean - 2 * _std, -0.025), _std * 4, 0.12, color='darkblue', alpha=0.2)
        ax.add_patch(rect)
    if len(b) >= 3:
        _mean = np.mean(b)
        _std = np.std(b)
        rect = patches.Rectangle((_mean - 2 * _std, 0.025), _std * 4, -0.12, color='darkgreen', alpha=0.2)
        ax.add_patch(rect)

    if title:
        plt.title(title)


def group_stat_results(df, group_col_name, col_agg_metric=None, col_list_metrics=None):
    def values(x):
        return '[' + ' '.join([f'{i:.3f}' for i in x]) + ']'

    def t_interval(x, p=0.95):
        import scipy.stats

        n = len(x)
        s = x.std(ddof=1)

        return scipy.stats.t.interval(p, n - 1, loc=x.mean(), scale=s / ((n - 1) ** 0.5))

    def t_int_l(x, p=0.95):
        return t_interval(x, p)[0]

    def t_int_h(x, p=0.95):
        return t_interval(x, p)[1]

    metric_aggregates = []
    metric_names = []
    if col_agg_metric is not None:
        metric_aggregates.extend([
            df.groupby(group_col_name)[m_col].agg(['mean', t_int_l, t_int_h, 'std', values])
            for m_col in col_agg_metric
        ])
        metric_names.extend(col_agg_metric)
    if col_list_metrics is not None:
        metric_aggregates.extend([
            df.groupby(group_col_name)[m_col].agg([values])
            for m_col in col_list_metrics
        ])
        metric_names.extend(col_list_metrics)

    df_results = pd.concat(metric_aggregates, axis=1, keys=metric_names).sort_index()
    return df_results
