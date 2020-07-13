"""
The `fit` function in this file implements a slightly modified version
of the Keras `model.fit()` API.
"""
import torch
from torch.optim import Optimizer
from torch.nn import Module
from torch.utils.data import DataLoader
from typing import Callable, List, Union
from numpy import mean as nmean
import numpy as np
from torch.nn.functional import log_softmax, nll_loss
from collections import *

from few_shot.callbacks import DefaultCallback, ProgressBarLogger, CallbackList, Callback
from few_shot.metrics import NAMED_METRICS


def gradient_step(model: Module, optimiser: Optimizer, loss_fn: Callable, x: torch.Tensor, y: torch.Tensor, **kwargs):
    """Takes a single gradient step.

    # Arguments
        model: Model to be fitted
        optimiser: Optimiser to calculate gradient step from loss
        loss_fn: Loss function to calculate between predictions and outputs
        x: Input samples
        y: Input targets
    """
    model.train()
    optimiser.zero_grad()
    y_pred = model(x)
    loss = loss_fn(y_pred, y)
    loss.backward()
    optimiser.step()

    return loss, y_pred


def logmeanexp_preds(output):
    output = torch.stack(output, dim=0)
    n_models = len(output)
    output = log_softmax(output, dim=-1)
    output = torch.logsumexp(output, dim=0) - np.log(n_models)  # [k*n, n]
    return output


def batch_metrics(model: Module, y_pred: torch.Tensor, y: torch.Tensor, metrics: List[Union[str, Callable]],
                  batch_logs: dict, prefix: str = None):
    """Calculates metrics for the current training batch

    # Arguments
        model: Model being fit
        y_pred: predictions for a particular batch
        y: labels for a particular batch
        batch_logs: Dictionary of logs for the current batch
    """
    if isinstance(model, list):
        for m in model:
            m.eval()
    else:
        model.eval()
    for m in metrics:
        if isinstance(m, str):
            key = m if prefix is None else f'{prefix}_' + m
            batch_logs[key] = NAMED_METRICS[m](y, y_pred)
        else:
            # Assume metric is a callable function
            batch_logs = m(y, y_pred)

    return batch_logs


def fit(model: Union[Module, List[Module]], optimiser: Optimizer, loss_fn: Callable, epochs: int, dataloader: DataLoader,
        prepare_batch: Callable, metrics: List[Union[str, Callable]] = None, callbacks: List[Callback] = None,
        verbose: bool = True, fit_function: Callable = gradient_step, n_models: int = 1, fit_function_kwargs: dict = {}):
    """Function to abstract away training loop.

    The benefit of this function is that allows training scripts to be much more readable and allows for easy re-use of
    common training functionality provided they are written as a subclass of voicemap.Callback (following the
    Keras API).

    # Arguments
        model: Model to be fitted.
        optimiser: Optimiser to calculate gradient step from loss
        loss_fn: Loss function to calculate between predictions and outputs
        epochs: Number of epochs of fitting to be performed
        dataloader: `torch.DataLoader` instance to fit the model to
        prepare_batch: Callable to perform any desired preprocessing
        metrics: Optional list of metrics to evaluate the model with
        callbacks: Additional functionality to incorporate into training such as logging metrics to csv, model
            checkpointing, learning rate scheduling etc... See voicemap.callbacks for more.
        verbose: All print output is muted if this argument is `False`
        fit_function: Function for calculating gradients. Leave as default for simple supervised training on labelled
            batches. For more complex training procedures (meta-learning etc...) you will need to write your own
            fit_function
        fit_function_kwargs: Keyword arguments to pass to `fit_function`
    """
    # Determine number of samples:
    num_batches = len(dataloader)
    batch_size = dataloader.batch_size

    fit_function_kwargs_logs = dict(fit_function_kwargs)

    fit_function_kwargs_logs['train'] = False
    fit_function_kwargs_logs['pred_fn'] = logmeanexp_preds

    callbacks = CallbackList([DefaultCallback(), ] + (callbacks or []) + [ProgressBarLogger(), ])
    callbacks.set_model(model)
    callbacks.set_params({
        'num_batches': num_batches,
        'batch_size': batch_size,
        'verbose': verbose,
        'metrics': (metrics or []),
        'prepare_batch': prepare_batch,
        'loss_fn': loss_fn,
        'optimiser': optimiser,
        'n_models': n_models
    })

    if verbose:
        print('Begin training...')

    callbacks.on_train_begin()

    for epoch in range(1, epochs+1):
        callbacks.on_epoch_begin(epoch)

        epoch_logs = {}
        for batch_index, batch in enumerate(dataloader):
            batch_logs = dict(batch=batch_index, size=(batch_size or 1))

            callbacks.on_batch_begin(batch_index, batch_logs)

            x, y = prepare_batch(batch)
            loss, y_pred, *base_logs = fit_function(model, optimiser, loss_fn, x, y, **fit_function_kwargs)
            batch_logs['loss'] = loss.item()

            # Loops through all metrics
            batch_logs = batch_metrics(model, y_pred, y, metrics, batch_logs)

            models_preds = base_logs[1]  # [n_models, n_tasks, n_objects, n_classes]
            task_preds = defaultdict(list)
            for model_pred in models_preds:
                for i, task in enumerate(model_pred):
                    task_preds[i].append(task)

            # task_preds : {task_idx : [model_1_pred, model_2_pred, ....] }
            logprobs_pred = []
            logprobs_loss = []
            for task_idx, task_pred in task_preds.items():
                y_pred = logmeanexp_preds(task_pred)
                logprobs_pred.append(y_pred)

            y_pred_logprobs = torch.stack(logprobs_pred)

            with torch.no_grad():
                loss_logprobs = loss_fn(y_pred_logprobs, y).item()

            batch_logs['logprobs_loss'] = loss_logprobs
            batch_logs['logprobs_nll'] = nll_loss(y_pred_logprobs, y, reduction="mean").item()
            batch_logs = batch_metrics(model, y_pred_logprobs, y, metrics, batch_logs, 'logprobs')

            if len(base_logs) > 0:
                models_losses = base_logs[0]
                models_preds = base_logs[1]

                for i, (loss, y_pred) in enumerate(zip(models_losses, models_preds)):
                    batch_logs[f'loss_{i}'] = nmean(loss)
                    batch_logs[f'categorical_accuracy_{i}'] = NAMED_METRICS['categorical_accuracy'](y, torch.cat(y_pred))

            callbacks.on_batch_end(batch_index, batch_logs)

        # Run on epoch end
        callbacks.on_epoch_end(epoch, epoch_logs)

    # Run on train end
    if verbose:
        print('Finished.')

    callbacks.on_train_end()


def to_numpy(preds):  # [n_models, n_tasks, n_objects, n_classes]
    a = []
    for model_pred in preds:
        # model_preds []
        task_pred = torch.stack(model_pred)
        pred = task_pred.cpu().detach().numpy()
        a.append(pred)
    return np.array(a)


def save_res(model: Union[Module, List[Module]], optimiser: Optimizer, loss_fn: Callable, epochs: int,
               dataloader: DataLoader,
               prepare_batch: Callable, metrics: List[Union[str, Callable]] = None, callbacks: List[Callback] = None,
               verbose: bool = True, fit_function: Callable = gradient_step, fit_function_kwargs: dict = {}, name=None):
    """Function to abstract away training loop.

    The benefit of this function is that allows training scripts to be much more readable and allows for easy re-use of
    common training functionality provided they are written as a subclass of voicemap.Callback (following the
    Keras API).

    # Arguments
        model: Model to be fitted.
        optimiser: Optimiser to calculate gradient step from loss
        loss_fn: Loss function to calculate between predictions and outputs
        epochs: Number of epochs of fitting to be performed
        dataloader: `torch.DataLoader` instance to fit the model to
        prepare_batch: Callable to perform any desired preprocessing
        metrics: Optional list of metrics to evaluate the model with
        callbacks: Additional functionality to incorporate into training such as logging metrics to csv, model
            checkpointing, learning rate scheduling etc... See voicemap.callbacks for more.
        verbose: All print output is muted if this argument is `False`
        fit_function: Function for calculating gradients. Leave as default for simple supervised training on labelled
            batches. For more complex training procedures (meta-learning etc...) you will need to write your own
            fit_function
        fit_function_kwargs: Keyword arguments to pass to `fit_function`
    """
    # Determine number of samples:
    assert name is not None
    num_batches = len(dataloader)
    batch_size = dataloader.batch_size
    fit_function_kwargs['train'] = False

    callbacks = CallbackList([DefaultCallback(), ] + (callbacks or []) + [ProgressBarLogger(), ])
    callbacks.set_model(model)
    callbacks.set_params({
        'num_batches': num_batches,
        'batch_size': batch_size,
        'verbose': verbose,
        'metrics': (metrics or []),
        'prepare_batch': prepare_batch,
        'loss_fn': loss_fn,
        'optimiser': optimiser
    })

    if verbose:
        print('Begin training...')

    callbacks.on_train_begin()

    for epoch in range(1, epochs + 1):
        callbacks.on_epoch_begin(epoch)

        epoch_logs = {}
        batch_preds = []
        for batch_index, batch in enumerate(dataloader):
            batch_logs = dict(batch=batch_index, size=(batch_size or 1))

            callbacks.on_batch_begin(batch_index, batch_logs)

            x, y = prepare_batch(batch)
            loss, y_pred, *base_logs = fit_function(model, optimiser, loss_fn, x, y, **fit_function_kwargs)

            models_preds = base_logs[1]  # [n_models, n_tasks, n_object, n_classes]
            batch_preds.append(to_numpy(models_preds))

            callbacks.on_batch_end(batch_index, batch_logs)

        # Run on epoch end

        epoch_logs['batches_train'] = batch_preds
        epoch_logs['name'] = name
        callbacks.on_epoch_end(epoch, epoch_logs)

    # Run on train end
    if verbose:
        print('Finished.')

    callbacks.on_train_end()
