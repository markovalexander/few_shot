import torch
from collections import OrderedDict
from torch.optim import Optimizer
from torch.nn import Module
from typing import Dict, List, Callable, Union
import torch.nn.functional as F
import threading
import sys
import traceback
from collections import defaultdict
import numpy as np
from few_shot.functions import MixtureLoss


class KeyErrorMessage(str):
    r"""str subclass that returns itself in repr"""

    def __repr__(self):
        return self


class ExceptionWrapper(object):
    r"""Wraps an exception plus traceback to communicate across threads"""

    def __init__(self, exc_info=None, where="in background"):
        # It is important that we don't store exc_info, see
        # NOTE [ Python Traceback Reference Cycle Problem ]
        if exc_info is None:
            exc_info = sys.exc_info()
        self.exc_type = exc_info[0]
        self.exc_msg = "".join(traceback.format_exception(*exc_info))
        self.where = where

    def reraise(self):
        r"""Reraises the wrapped exception in the current thread"""
        # Format a message such as: "Caught ValueError in DataLoader worker
        # process 2. Original Traceback:", followed by the traceback.
        msg = "Caught {} {}.\nOriginal {}".format(
            self.exc_type.__name__, self.where, self.exc_msg)
        if self.exc_type == KeyError:
            # KeyError calls repr() on its argument (usually a dict key). This
            # makes stack traces unreadable. It will not be changed in Python
            # (https://bugs.python.org/issue2651), so we work around it.
            msg = KeyErrorMessage(msg)
        raise self.exc_type(msg)


from few_shot.core import create_nshot_task_label


def replace_grad(parameter_gradients, parameter_name):
    def replace_grad_(module):
        return parameter_gradients[parameter_name]

    return replace_grad_

#
# def pred_fn(output, mode='mean'):
#     output = torch.stack(output, dim=0)
#     if mode == 'mean':
#         output = F.log_softmax(output, dim=-1)
#         output = torch.mean(output, dim=0)
#         return output
#     elif mode == 'logprobs':
#         n_models = len(output)
#         output = F.log_softmax(output, dim=-1)
#         output = torch.logsumexp(output, dim=0) - np.log(n_models)  # [k*n, n]
#         return output
#     else:
#         raise ValueError("invalid 'pred_mode' argument")


def get_grads(ensemble):
    grads = {}
    for i, m in enumerate(ensemble):
        grads[i] = {idx: p.grad for idx, p in enumerate(m.parameters())}
    return grads


def gather_predictions(predictions, device_idx):
    # predicitons {gpu_idx : task_predictions}
    # task_prediction = [task, model; n_objects, n_classes]

    task_preds = defaultdict(list)

    for gpu_idx, pred in predictions.items():
        # pred[task, models; n_objects x n_classes]

        for task_idx, task in enumerate(pred):
            # task[models, n_objects, n_classes]
            for model_pred in task:
                if gpu_idx != device_idx:
                    append_pred = model_pred.clone().detach().to(device_idx)
                else:
                    append_pred = model_pred

                task_preds[task_idx].append(append_pred)

    return list(task_preds.values())


def copy_grads(models, replicas, models2replicas, device):
    # print('in copy grads')
    for i, model in enumerate(models):
        device_idx, replica_idx = models2replicas[i]
        # print('i: {}, device_idx: {}, replica_idx: {}'.format(i, device_idx, replica_idx))
        replica = replicas[device_idx][replica_idx]
        for p, r_p in zip(model.parameters(), replica.parameters()):
            p.grad = r_p.grad.to(device)


def meta_gradient_ens_step_mgpu_2order(models: List[Module],
                                        optimisers: List[Optimizer],
                                        loss_fn: Callable,
                                        x: torch.Tensor,
                                        y: torch.Tensor,
                                        n_shot: int,
                                        k_way: int,
                                        q_queries: int,
                                        order: int,
                                        pred_fn: Callable,
                                        inner_train_steps: int,
                                        inner_lr: float,
                                        train: bool,
                                        model_params: List,
                                        device: Union[str, torch.device],
                                        update_weights: bool = True):
    data_shape = x.shape[2:]
    create_graph = (True if order == 2 else False) and train

    devices = list(range(torch.cuda.device_count()))
    # inputs = scatter(x, devices)
    model_class = models[0].__class__
    models_to_replicas = {}
    model_replicas = [[] for _ in range(len(devices))]
    for i, model in enumerate(models):
        device_idx = i % len(devices)
        replica = model_class(*model_params).to(devices[device_idx],
                                                dtype=torch.double)
        replica.load_state_dict(model.state_dict())
        model_replicas[device_idx].append(replica)
        models_to_replicas[i] = (
        device_idx, len(model_replicas[device_idx]) - 1)
        if len(model_replicas[device_idx]) > 3 and train:
            print(
                'Probably will trigger memory error: {} GPU has too many models'.format(
                    device_idx))

    lock = threading.Lock()
    barrier = threading.Barrier(len(devices))
    predictions = {}

    meta_batch_losses = {}
    task_predictions_mgpu = {}

    models_losses = {}
    models_predictions = {}
    support_losses = {}
    # grads = {}

    def _worker(i, models, x, device):

        losses_mgpu = []
        preds_mgpu = []
        task_predictions = []
        support_loss = []

        try:
            with torch.cuda.device(device):
                for meta_batch in x:
                    x_task_train = meta_batch[:n_shot * k_way]
                    x_task_val = meta_batch[n_shot * k_way:]

                    preds = []
                    for model in models:
                        fast_weights = OrderedDict(model.named_parameters())

                        for inner_batch in range(inner_train_steps):
                            y = create_nshot_task_label(k_way, n_shot).to(
                                device)
                            logits = model.functional_forward(x_task_train,
                                                              fast_weights)
                            loss = F.cross_entropy(logits, y)
                            gradients = torch.autograd.grad(loss,
                                                            fast_weights.values(),
                                                            create_graph=create_graph)

                            fast_weights = OrderedDict(
                                (name, param - inner_lr * grad)
                                for ((name, param), grad) in
                                zip(fast_weights.items(), gradients)
                            )

                        support_loss.append(loss.item())

                        y = create_nshot_task_label(k_way, q_queries).to(device)
                        model_logits = model.functional_forward(x_task_val,
                                                                fast_weights)
                        preds.append(model_logits)

                    task_predictions.append(preds)

                with lock:
                    predictions[i] = task_predictions

                barrier.wait()

                with lock:
                    task_predictions = gather_predictions(predictions, i)

                for task_pred in task_predictions:
                    y_pred = pred_fn(task_pred)
                    loss = loss_fn(y_pred, y)
                    loss.backward(retain_graph=True)
                    preds_mgpu.append(y_pred)
                    losses_mgpu.append(loss)

                total_models = len(task_predictions[0])

                models_task_losses = []  # [n_models, n_tasks]
                models_task_preds = []  # [n_models, n_tasks, n_classes]
                if i == 0:
                    with torch.no_grad():
                        for model_idx in range(total_models):
                            task_loss = []
                            task_pred = []
                            for task in task_predictions:
                                loss = loss_fn(
                                    F.log_softmax(task[model_idx], dim=-1),
                                    y).item()
                                task_loss.append(loss)
                                task_pred.append(task[model_idx])
                            models_task_losses.append(task_loss)
                            models_task_preds.append(task_pred)

                if order == 2:
                    for model in models:
                        model.train()

                    meta_batch_loss = torch.stack(losses_mgpu).mean()
                    if train:
                        meta_batch_loss.backward()

                    with lock:
                        meta_batch_losses[i] = meta_batch_loss
                        task_predictions_mgpu[i] = torch.cat(preds_mgpu)
                        models_losses[i] = models_task_losses
                        models_predictions[i] = models_task_preds
                        support_losses[i] = support_loss
                elif order == 1:
                    pass
                else:
                    raise ValueError('Order must be either 1 or 2.')

        except Exception:
            with lock:
                meta_batch_losses[i] = ExceptionWrapper(
                    where="in replica {} on device {}".format(i, device))
                task_predictions_mgpu[i] = ExceptionWrapper(
                    where="in replica {} on device {}".format(i, device))

    threads = [threading.Thread(target=_worker,
                                args=(i, models, x.to(device), device))
               for i, (models, device) in
               enumerate(zip(model_replicas, devices))]

    for o in optimisers:
        o.zero_grad()
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    for i in range(len(devices)):
        loss_i = meta_batch_losses[i]
        if isinstance(loss_i, ExceptionWrapper):
            loss_i.reraise()

    if train:
        copy_grads(models, model_replicas, models_to_replicas, device)
        if update_weights:
            for o in optimisers:
                o.step()

    meta_batch_loss = meta_batch_losses[0]
    task_predictions = task_predictions_mgpu[0]

    models_losses = models_losses[0]
    models_predictions = models_predictions[0]

    mean_support_loss = 0
    for k, v in support_losses.items():
        mean_support_loss += np.mean(v)
    mean_support_loss /= len(devices)

    result = {
        "meta_batch_loss" : meta_batch_loss,
        "task_predictions" : task_predictions,
        "models_losses" : models_losses,
        "models_predictions" : models_predictions,
        "mean_support_loss" : mean_support_loss
    }
    return result


# TODO: move this logic inside main function
def meta_gradient_ens_step_mgpu_2order_mixture(models: List[Module],
                                                optimisers: List[Optimizer],
                                                loss_fn: Callable,
                                                x: torch.Tensor,
                                                y: torch.Tensor,
                                                n_shot: int,
                                                k_way: int,
                                                q_queries: int,
                                                order: int,
                                                pred_fn: Callable,
                                                inner_train_steps: int,
                                                inner_lr: float,
                                                train: bool,
                                                model_params: List,
                                                device: Union[str, torch.device]):
    data_shape = x.shape[2:]
    create_graph = (True if order == 2 else False) and train

    devices = list(range(torch.cuda.device_count()))
    # inputs = scatter(x, devices)
    model_class = models[0].__class__
    models_to_replicas = {}
    model_replicas = [[] for _ in range(len(devices))]
    for i, model in enumerate(models):
        device_idx = i % len(devices)
        replica = model_class(*model_params).to(devices[device_idx],
                                                dtype=torch.double)
        replica.load_state_dict(model.state_dict())
        model_replicas[device_idx].append(replica)
        models_to_replicas[i] = (
        device_idx, len(model_replicas[device_idx]) - 1)
        if len(model_replicas[device_idx]) > 3 and train:
            print(
                'Probably will trigger memory error: {} GPU has too many models'.format(
                    device_idx))

    lock = threading.Lock()
    barrier = threading.Barrier(len(devices))
    predictions = {}

    meta_batch_losses = {}
    task_predictions_mgpu = {}

    models_losses = {}
    models_predictions = {}
    support_losses = {}
    # grads = {}

    def _worker(i, models, x, device):

        losses_mgpu = []
        preds_mgpu = []
        task_predictions = []
        support_loss = []

        try:
            with torch.cuda.device(device):
                for meta_batch in x:
                    x_task_train = meta_batch[:n_shot * k_way]
                    x_task_val = meta_batch[n_shot * k_way:]

                    preds = []
                    for model in models:
                        fast_weights = OrderedDict(model.named_parameters())

                        for inner_batch in range(inner_train_steps):
                            y = create_nshot_task_label(k_way, n_shot).to(
                                device)
                            logits = model.functional_forward(x_task_train,
                                                              fast_weights)
                            loss = F.cross_entropy(logits, y)
                            gradients = torch.autograd.grad(loss,
                                                            fast_weights.values(),
                                                            create_graph=create_graph)

                            fast_weights = OrderedDict(
                                (name, param - inner_lr * grad)
                                for ((name, param), grad) in
                                zip(fast_weights.items(), gradients)
                            )

                        support_loss.append(loss.item())

                        y = create_nshot_task_label(k_way, q_queries).to(device)
                        model_logits = model.functional_forward(x_task_val,
                                                                fast_weights)
                        preds.append(model_logits)

                    task_predictions.append(preds)

                with lock:
                    predictions[i] = task_predictions

                barrier.wait()

                with lock:
                    task_predictions = gather_predictions(predictions, i)

                for task_pred in task_predictions:
                    y_pred = pred_fn(task_pred)
                    loss = loss_fn(task_pred, y)
                    loss.backward(retain_graph=True)
                    preds_mgpu.append(y_pred)
                    losses_mgpu.append(loss)

                total_models = len(task_predictions[0])

                models_task_losses = []  # [n_models, n_tasks]
                models_task_preds = []  # [n_models, n_tasks, n_classes]
                if i == 0:
                    with torch.no_grad():
                        for model_idx in range(total_models):
                            task_loss = []
                            task_pred = []
                            for task in task_predictions:
                                loss = F.cross_entropy(
                                    F.log_softmax(task[model_idx], dim=-1),
                                    y).item()
                                task_loss.append(loss)
                                task_pred.append(task[model_idx])
                            models_task_losses.append(task_loss)
                            models_task_preds.append(task_pred)

                if order == 2:
                    for model in models:
                        model.train()

                    meta_batch_loss = torch.stack(losses_mgpu).mean()
                    if train:
                        meta_batch_loss.backward()

                    with lock:
                        meta_batch_losses[i] = meta_batch_loss
                        task_predictions_mgpu[i] = torch.cat(preds_mgpu)
                        models_losses[i] = models_task_losses
                        models_predictions[i] = models_task_preds
                        support_losses[i] = support_loss
                elif order == 1:
                    pass
                else:
                    raise ValueError('Order must be either 1 or 2.')

        except Exception:
            with lock:
                meta_batch_losses[i] = ExceptionWrapper(
                    where="in replica {} on device {}".format(i, device))
                task_predictions_mgpu[i] = ExceptionWrapper(
                    where="in replica {} on device {}".format(i, device))

    threads = [threading.Thread(target=_worker,
                                args=(i, models, x.to(device), device))
               for i, (models, device) in
               enumerate(zip(model_replicas, devices))]

    for o in optimisers:
        o.zero_grad()
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    for i in range(len(devices)):
        loss_i = meta_batch_losses[i]
        if isinstance(loss_i, ExceptionWrapper):
            loss_i.reraise()

    if train:
        copy_grads(models, model_replicas, models_to_replicas, device)
        for o in optimisers:
            o.step()

    meta_batch_loss = meta_batch_losses[0]
    task_predictions = task_predictions_mgpu[0]

    models_losses = models_losses[0]
    models_predictions = models_predictions[0]

    mean_support_loss = 0
    for k, v in support_losses.items():
        mean_support_loss += np.mean(v)
    mean_support_loss /= len(devices)

    result = {
        "meta_batch_loss" : meta_batch_loss,
        "task_predictions" : task_predictions,
        "models_losses" : models_losses,
        "models_predictions" : models_predictions,
        "mean_support_loss" : mean_support_loss
    }
    return result
    #return meta_batch_loss, task_predictions, models_losses, models_predictions, mean_support_loss


def meta_gradient_ens_step_mgpu_1order(models: List[Module],
                                       optimisers: List[Optimizer],
                                       loss_fn: Callable,
                                       x: torch.Tensor,
                                       y: torch.Tensor,
                                       n_shot: int,
                                       k_way: int,
                                       q_queries: int,
                                       order: int,
                                       pred_fn: str,
                                       inner_train_steps: int,
                                       inner_lr: float,
                                       train: bool,
                                       model_params: List,
                                       device: Union[str, torch.device]):
    data_shape = x.shape[2:]
    create_graph = (True if order == 2 else False) and train

    devices = list(range(torch.cuda.device_count()))
    # inputs = scatter(x, devices)
    model_class = models[0].__class__
    models_to_replicas = {}
    model_replicas = [[] for _ in range(len(devices))]
    losses = [loss_fn for _ in range(len(devices))]

    for i, model in enumerate(models):
        device_idx = i % len(devices)
        replica = model_class(*model_params).to(devices[device_idx],
                                                dtype=torch.double)
        replica.load_state_dict(model.state_dict())
        model_replicas[device_idx].append(replica)
        models_to_replicas[i] = (
        device_idx, len(model_replicas[device_idx]) - 1)
        if len(model_replicas[device_idx]) > 3 and train:
            print(
                'Probably will trigger memory error: {} GPU has too many models'.format(
                    device_idx))

    for device, loss in zip(devices, losses):
        if isinstance(loss_fn, MixtureLoss):
            loss_fn = loss_fn.to(device)

    lock = threading.Lock()
    barrier = threading.Barrier(len(devices))
    predictions = {}

    meta_batch_losses = {}
    task_predictions_mgpu = {}

    models_losses = {}
    models_predictions = {}

    def _worker(i, models, x, loss_fn, device):

        for model in models:
            o = torch.optim.SGD(model.parameters(), lr=1)
            o.zero_grad()

        losses_mgpu = []
        preds_mgpu = []
        named_grads = {}
        task_predictions_gathered = []
        try:
            with torch.cuda.device(device):
                for meta_batch in x:
                    task_predictions = []
                    x_task_train = meta_batch[:n_shot * k_way]
                    x_task_val = meta_batch[n_shot * k_way:]

                    preds = []
                    fast_weights_dict = {}
                    for idx, model in enumerate(models):
                        fast_weights = OrderedDict(model.named_parameters())

                        for inner_batch in range(inner_train_steps):
                            y = create_nshot_task_label(k_way, n_shot).to(
                                device)
                            logits = model.functional_forward(x_task_train,
                                                              fast_weights)
                            loss = F.cross_entropy(logits, y)
                            gradients = torch.autograd.grad(loss,
                                                            fast_weights.values(),
                                                            create_graph=create_graph)

                            fast_weights = OrderedDict(
                                (name, param - inner_lr * grad)
                                for ((name, param), grad) in
                                zip(fast_weights.items(), gradients)
                            )

                        y = create_nshot_task_label(k_way, q_queries).to(device)
                        model_logits = model.functional_forward(x_task_val,
                                                                fast_weights)
                        preds.append(model_logits)
                        fast_weights_dict[idx] = fast_weights

                    task_predictions.append(preds)

                    with lock:
                        predictions[i] = task_predictions

                    barrier.wait()

                    with lock:
                        task_predictions = gather_predictions(predictions, i)

                    # TODO: does not work
                    y_pred = pred_fn(task_predictions[0])
                    task_predictions_gathered.append(task_predictions[0])
                    loss = loss_fn(y_pred, y)
                    loss.backward(retain_graph=True)

                    preds_mgpu.append(y_pred)
                    losses_mgpu.append(loss)

                    for idx, model in enumerate(models):
                        gradients = torch.autograd.grad(loss, fast_weights_dict[idx].values(),
                                                        create_graph=create_graph, retain_graph=True)
                        for p, grad in zip(model.parameters(), gradients):
                            p.grad += grad

                total_models = len(task_predictions[0])

                models_task_losses = []  # [n_models, n_tasks]
                models_task_preds = []  # [n_models, n_tasks, n_classes]
                if i == 0:
                    with torch.no_grad():
                        for model_idx in range(total_models):
                            task_loss = []
                            task_pred = []
                            for task in task_predictions_gathered:
                                loss = loss_fn(
                                    F.log_softmax(task[model_idx], dim=-1),
                                    y).item()
                                task_loss.append(loss)
                                task_pred.append(task[model_idx])
                            models_task_losses.append(task_loss)
                            models_task_preds.append(task_pred)

                if order == 2:
                    raise ValueError('Order must be 1')
                elif order == 1:
                    meta_batch_loss = torch.stack(losses_mgpu).mean()
                    with lock:
                        meta_batch_losses[i] = meta_batch_loss
                        task_predictions_mgpu[i] = torch.cat(preds_mgpu)
                        models_losses[i] = models_task_losses
                        models_predictions[i] = models_task_preds

                else:
                    raise ValueError('Order must be either 1 or 2.')

        except Exception:
            with lock:
                meta_batch_losses[i] = ExceptionWrapper(
                    where="in replica {} on device {}".format(i, device))
                task_predictions_mgpu[i] = ExceptionWrapper(
                    where="in replica {} on device {}".format(i, device))

    threads = [threading.Thread(target=_worker,
                                args=(i, models, x.to(device), loss_fn, device))
               for i, (models, loss_fn, device) in
               enumerate(zip(model_replicas, losses, devices))]

    for o in optimisers:
        o.zero_grad()
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    for i in range(len(devices)):
        loss_i = meta_batch_losses[i]
        if isinstance(loss_i, ExceptionWrapper):
            loss_i.reraise()

    if train:
        copy_grads(models, model_replicas, models_to_replicas, device)
        for o in optimisers:
            o.step()

    meta_batch_loss = meta_batch_losses[0]
    task_predictions = task_predictions_mgpu[0]
    models_losses = models_losses[0]
    models_predictions = models_predictions[0]
    return meta_batch_loss, task_predictions, models_losses, models_predictions
