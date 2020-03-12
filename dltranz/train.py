import logging
import warnings

import torch
from ignite.contrib.handlers import ProgressBar, LRScheduler, create_lr_scheduler_with_warmup
from ignite.contrib.handlers.param_scheduler import ParamScheduler
from ignite.metrics import RunningAverage
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore', module='tensorboard.compat.tensorflow_stub.dtypes')
from torch.utils.tensorboard import SummaryWriter

from dltranz.seq_encoder import PaddedBatch

from ignite.engine import Engine, Events, create_supervised_trainer, create_supervised_evaluator
import ignite
from bisect import bisect_right

logger = logging.getLogger(__name__)


def batch_to_device(batch, device, non_blocking):
    x, y = batch
    if not isinstance(x, dict):
        new_x = {k: v.to(device=device, non_blocking=non_blocking) if isinstance(v, torch.Tensor) else v for k, v in
                 x.payload.items()}
        new_y = y.to(device=device, non_blocking=non_blocking)
        return PaddedBatch(new_x, x.seq_lens), new_y
    else:
        batches = {}
        for key, sx in x.items():
            new_x = {k: v.to(device=device, non_blocking=non_blocking) if isinstance(v, torch.Tensor) else v for k, v in
                     sx.payload.items()}
            batches[key] = PaddedBatch(new_x, sx.seq_lens)
        new_y = y.to(device=device, non_blocking=non_blocking)
        return batches, new_y


def get_optimizer(model, params):
    """Returns optimizer

    :param model: model with his `model.named_parameters()`
    :param params: dict with options:
        ['train.lr']: `lr` for Adam optimizer
        ['train.weight_decay']: `weight_decay` for Adam optimizer
        ['train.optimiser_params']: (optional) list of tuples (par_name, options),
            each tuple define new parameter group.
            `par_name` is end of parameter name from `model.named_parameters()` for this parameter group
            'options' is dict with options for this parameter group
    :return:
    """
    optimiser_params = params.get('train.optimiser_params', None)
    if optimiser_params is None:
        parameters = model.parameters()
    else:
        parameters = []
        for par_name, options in optimiser_params.items():
            options = options.copy()
            options['params'] = [v for k, v in model.named_parameters() if k.startswith(par_name)]
            parameters.append(options)
        default_options = {
            'params': [v for k, v in model.named_parameters() if all(
                (not k.startswith(par_name) for par_name, options in optimiser_params.items())
            )]}
        parameters.append(default_options)
    optimizer = torch.optim.Adam(parameters, lr=params['train.lr'], weight_decay=params['train.weight_decay'])
    return optimizer


class SchedulerWrapper:
    def __init__(self, optimizer):
        self.optimizer = optimizer

    def __call__(self, *args, **kwargs):
        self.optimizer.step()


class MultiGammaScheduler(torch.optim.lr_scheduler.MultiStepLR):

    def __init__(self, optimizer, milestones, gammas, gamma=0.1, last_epoch=-1):
        self.gammas = gammas
        super(MultiGammaScheduler, self).__init__(optimizer, milestones, gamma, last_epoch)

    def get_lr(self):
        idx = bisect_right(self.milestones, self.last_epoch)
        gammas = self.gammas[:idx]
        gamma = np.prod(gammas)
        return [base_lr * gamma for base_lr in self.base_lrs]


def get_lr_scheduler(optimizer, params):
    if 'scheduler' in params:
        # TODO: check the this code branch
        if params['scheduler.current'] != '':
            scheduler_type = params['scheduler.current']

            scheduler_params = params[f'scheduler.{scheduler_type}']

            if scheduler_type == 'MultiGammaScheduler':
                scheduler = MultiGammaScheduler(optimizer,
                                                milestones=scheduler_params['milestones'],
                                                gammas=scheduler_params['gammas'],
                                                gamma=scheduler_params['gamma'],
                                                last_epoch=scheduler_params['last_epoch'])

            logger.info('MultiGammaScheduler used')
    else:
        lr_step_size = params['lr_scheduler']['step_size']
        lr_step_gamma = params['lr_scheduler']['step_gamma']
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_step_size, gamma=lr_step_gamma)
        logger.info('StepLR lr_scheduler used')

    if 'warmup' in params['lr_scheduler']:
        wrapper = LRScheduler
        # optimiser param groups are not supported with LRScheduler
    else:
        wrapper = SchedulerWrapper
        # create_lr_scheduler_with_warmup don't works with SchedulerWrapper
    scheduler = wrapper(scheduler)

    if 'warmup' in params['lr_scheduler']:
        scheduler = create_lr_scheduler_with_warmup(scheduler, **params['lr_scheduler.warmup'])
        logger.info('LR warmup used')

    return scheduler


class MlflowHandler:
    def __init__(self, logger):
        self.logger = logger

    def __call__(self, train_engine, valid_engine, optimizer):
        def global_state_transform(*args, **kwargs):
            return train_engine.state.iteration

        self.logger.attach(
            train_engine,
            log_handler=ignite.contrib.handlers.mlflow_logger.OutputHandler(
                tag='train',
                metric_names='all'
            ),
            event_name=Events.ITERATION_STARTED
        )

        self.logger.attach(
            valid_engine,
            log_handler=ignite.contrib.handlers.mlflow_logger.OutputHandler(
                tag='validation',
                metric_names='all',
                global_step_transform=global_state_transform
            ),
            event_name=Events.EPOCH_COMPLETED
        )

        self.logger.attach(
            train_engine,
            log_handler=ignite.contrib.handlers.mlflow_logger.OptimizerParamsHandler(optimizer),
            event_name=Events.ITERATION_STARTED
        )


class TensorboardHandler:
    def __init__(self, log_dir):
        self.logger = SummaryWriter(log_dir)

    def __call__(self, train_engine, valid_engine, optimizer):
        @train_engine.on(Events.ITERATION_COMPLETED)
        def log_training_loss(engine):
            self.logger.add_scalar('train/loss', engine.state.metrics['loss'], engine.state.iteration)


class PrepareEpoch:
    def __init__(self, train_loader):
        self.train_loader = train_loader

    def __call__(self, *args, **kwargs):
        if hasattr(self.train_loader, 'prepare_epoch'):
            self.train_loader.prepare_epoch()


def kld_f(embeddings):
    d = embeddings.size()[1] // 2
    mu, logvar = embeddings[:, :d], embeddings[:, d:]
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return kld


def fit_model(model, train_loader, valid_loader, loss, optimizer, scheduler, params, valid_metrics, train_handlers):
    device = torch.device(params.get('device', 'cpu'))
    model.to(device)

    trainer = create_supervised_trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss,
        device=device,
        prepare_batch=batch_to_device,
        output_transform=lambda x, y, y_pred, loss:
        (
            loss.item(),
            x[next(iter(x.keys()))].seq_lens if isinstance(x, dict) else x.seq_lens,
            kld_f(y_pred).item(),
        ),
    )

    RunningAverage(output_transform=lambda x: x[0]).attach(trainer, 'loss')
    RunningAverage(output_transform=lambda x: x[1].float().mean()).attach(trainer, 'seq_len')
    RunningAverage(output_transform=lambda x: x[2]).attach(trainer, 'loss_kld')
    RunningAverage(output_transform=lambda x: x[0] - x[2]).attach(trainer, 'loss_ml')
    pbar = ProgressBar(persist=True, bar_format="")
    pbar.attach(trainer, ['loss', 'seq_len', 'loss_kld', 'loss_ml'])

    validation_evaluator = create_supervised_evaluator(
        model=model,
        device=device,
        prepare_batch=batch_to_device,
        metrics=valid_metrics
    )

    pbar = ProgressBar(persist=True, bar_format="")
    pbar.attach(validation_evaluator)

    # valid_metric_name = valid_metric.__class__.__name__
    # valid_metric.attach(validation_evaluator, valid_metric_name)

    trainer.add_event_handler(Events.EPOCH_STARTED, PrepareEpoch(train_loader))

    trainer.add_event_handler(Events.EPOCH_COMPLETED, scheduler)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(engine):
        validation_evaluator.run(valid_loader)
        metrics = validation_evaluator.state.metrics
        msgs = []
        for metric in valid_metrics:
            msgs.append(f'{metric}: {metrics[metric]:.3f}')
        pbar.log_message(
            f'Epoch: {engine.state.epoch},  {", ".join(msgs)}')

    for handler in train_handlers:
        handler(trainer, validation_evaluator, optimizer)

    trainer.run(train_loader, max_epochs=params['train.n_epoch'])

    return validation_evaluator.state.metrics


def score_model(model, valid_loader, params):
    device = torch.device(params.get('device', 'cpu'))
    model.to(device)

    pred = []
    true = []

    def process_valid(_, batch):
        x, y = batch_to_device(batch, device, False)

        model.eval()
        with torch.no_grad():
            outputs = model(x)
            pred.append(outputs.cpu().numpy())
            true.append(y.cpu().numpy())

        return outputs, y

    validation_evaluator = Engine(process_valid)

    pbar = ProgressBar(persist=True, bar_format="")
    pbar.attach(validation_evaluator)

    validation_evaluator.run(valid_loader)

    true = np.concatenate(true)
    pred = np.concatenate(pred)

    if params['norm_scores']:
        pred = pred / np.max(pred)
        pred = pred - np.min(pred)

    return true, pred


def block_iterator(iterator, size):
    bucket = list()
    for e in iterator:
        bucket.append(e)
        if len(bucket) >= size:
            yield bucket
            bucket = list()
    if bucket:
        yield bucket


def predict_proba_path(model, path_wc, create_loader, files_per_batch=100):
    params = model.params

    from glob import glob
    import sparkpickle

    data_files = [path for path in glob(path_wc)]

    scores = []
    for fl in block_iterator(data_files, files_per_batch):
        score_data = list()

        for path in fl:
            with open(path, 'rb') as f:
                score_data.extend(dict(e) for e in sparkpickle.load(f))

        loader = create_loader(score_data, params)
        if len(loader) == 0:  # no valid samples in block
            continue

        pred = score_model(model, loader, params)
        scores.append(pred)

    return pd.concat(scores)
