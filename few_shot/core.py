from torch.utils.data import Sampler
from typing import List, Iterable, Callable, Tuple
import numpy as np
import torch

from few_shot.metrics import categorical_accuracy
from few_shot.callbacks import Callback


class NShotTaskSampler(Sampler):
    def __init__(self,
                 dataset: torch.utils.data.Dataset,
                 episodes_per_epoch: int = None,
                 n: int = None,
                 k: int = None,
                 q: int = None,
                 num_tasks: int = 1,
                 fixed_tasks: List[Iterable[int]] = None):
        """PyTorch Sampler subclass that generates batches of n-shot, k-way, q-query tasks.

        Each n-shot task contains a "support set" of `k` sets of `n` samples and a "query set" of `k` sets
        of `q` samples. The support set and the query set are all grouped into one Tensor such that the first n * k
        samples are from the support set while the remaining q * k samples are from the query set.

        The support and query sets are sampled such that they are disjoint i.e. do not contain overlapping samples.

        # Arguments
            dataset: Instance of torch.utils.data.Dataset from which to draw samples
            episodes_per_epoch: Arbitrary number of batches of n-shot tasks to generate in one epoch
            n_shot: int. Number of samples for each class in the n-shot classification tasks.
            k_way: int. Number of classes in the n-shot classification tasks.
            q_queries: int. Number query samples for each class in the n-shot classification tasks.
            num_tasks: Number of n-shot tasks to group into a single batch
            fixed_tasks: If this argument is specified this Sampler will always generate tasks from
                the specified classes
        """
        super(NShotTaskSampler, self).__init__(dataset)
        self.episodes_per_epoch = episodes_per_epoch
        self.dataset = dataset
        if num_tasks < 1:
            raise ValueError('num_tasks must be > 1.')

        self.num_tasks = num_tasks
        # TODO: Raise errors if initialise badly
        self.k = k
        self.n = n
        self.q = q
        self.fixed_tasks = fixed_tasks

        self.i_task = 0

    def __len__(self):
        return self.episodes_per_epoch

    def __iter__(self):
        for _ in range(self.episodes_per_epoch):
            batch = []

            for task in range(self.num_tasks):
                if self.fixed_tasks is None:
                    # Get random classes
                    episode_classes = np.random.choice(self.dataset.df['class_id'].unique(), size=self.k, replace=False)
                else:
                    # Loop through classes in fixed_tasks
                    episode_classes = self.fixed_tasks[self.i_task % len(self.fixed_tasks)]
                    self.i_task += 1

                df = self.dataset.df[self.dataset.df['class_id'].isin(episode_classes)]

                support_k = {k: None for k in episode_classes}
                for k in episode_classes:
                    # Select support examples
                    support = df[df['class_id'] == k].sample(self.n)
                    support_k[k] = support

                    for i, s in support.iterrows():
                        batch.append(s['id'])

                for k in episode_classes:
                    query = df[(df['class_id'] == k) & (~df['id'].isin(support_k[k]['id']))].sample(self.q)
                    for i, q in query.iterrows():
                        batch.append(q['id'])

            yield np.stack(batch)


class AccumulateSNR(Callback):
    def __init__(self, eval_fn, taskloader, prepare_batch, n_batches=20):
        super().__init__()
        self.eval_fn = eval_fn
        self.taskloader = taskloader
        self.prepare_batch = prepare_batch
        self.n_batches = n_batches

    def on_train_begin(self, logs=None):
        self.first_moment = [{k: np.zeros(v.shape) for k, v in model.named_parameters()} for _, model in enumerate(self.model)]
        self.second_moment = [{k: np.zeros(v.shape) for k, v in model.named_parameters()} for _, model in enumerate(self.model)]
        self.count = 0

    def on_epoch_begin(self, epoch, logs=None):
        for fm, sm in zip(self.first_moment, self.second_moment):
            for key in fm.keys():
                fm[key].fill(0)
                sm[key].fill(0)
        self.count = 0

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        seen = 0

        for batch_index, batch in enumerate(self.taskloader):
            if batch_index > self.n_batches:
                break
            x, y = self.prepare_batch(batch)

            #loss, y_pred, *base_logs = self.eval_fn(
            result = self.eval_fn(
                self.model,
                self.optimiser,
                self.loss_fn,
                x,
                y,
                n_shot=self.n_shot,
                k_way=self.k_way,
                q_queries=self.q_queries,
                train=False,
                **self.kwargs
            )

            for idx, model in enumerate(self.model):
                for k, v in model.named_parameters():
                    grad = v.grad.data.cpu().numpy()
                    self.first_moment[idx][k] += grad
                    self.second_moment[idx][k] += grad ** 2
            self.count += 1

        snrs = [self.evaluate_model_snr(fm, sm) for fm, sm in zip(self.first_moment, self.second_moment)]
        for i, snr in enumerate(snrs):
            logs[f'snr_{i}'] = snr

    def evaluate_model_snr(self, first_moment, second_moment, eps=1e-6):
        std = {k: np.sqrt(eps + second_moment[k] / self.count - (first_moment[k] / self.count) ** 2)
               for k in first_moment.keys()}
        mean = {k: first_moment[k] / self.count for k in first_moment.keys()}
        snr = {k: mean[k] / std[k] for k in first_moment.keys()}

        total_snr, n_params = 0, 0
        for v in snr.values():
            total_snr += np.sum(np.abs(v))
            n_params += v.size

        total_snr /= n_params
        return total_snr



class EvaluateFewShot(Callback):
    """Evaluate a network on  an n-shot, k-way classification tasks after every epoch.

    # Arguments
        eval_fn: Callable to perform few-shot classification. Examples include `proto_net_episode`,
            `matching_net_episode` and `meta_gradient_step` (MAML).
        num_tasks: int. Number of n-shot classification tasks to evaluate the model with.
        n_shot: int. Number of samples for each class in the n-shot classification tasks.
        k_way: int. Number of classes in the n-shot classification tasks.
        q_queries: int. Number query samples for each class in the n-shot classification tasks.
        task_loader: Instance of NShotWrapper class
        prepare_batch: function. The preprocessing function to apply to samples from the dataset.
        prefix: str. Prefix to identify dataset.
    """

    def __init__(self,
                 eval_fn: Callable,
                 num_tasks: int,
                 n_shot: int,
                 k_way: int,
                 q_queries: int,
                 taskloader: torch.utils.data.DataLoader,
                 prepare_batch: Callable,
                 loss_fn: Callable,
                 prefix: str = 'val_',
                 **kwargs):
        super(EvaluateFewShot, self).__init__()
        self.eval_fn = eval_fn
        self.num_tasks = num_tasks
        self.n_shot = n_shot
        self.k_way = k_way
        self.q_queries = q_queries
        self.taskloader = taskloader
        self.prepare_batch = prepare_batch
        self.prefix = prefix
        self.kwargs = kwargs
        self.metric_name = f'{self.prefix}{self.n_shot}-shot_{self.k_way}-way_acc'
        self.loss_fn = loss_fn

    def on_train_begin(self, logs=None):
        # self.loss_fn = self.params['loss_fn']
        self.optimiser = self.params['optimiser']

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        seen = 0
        totals = {'loss': 0, self.metric_name: 0}
        if isinstance(self.model, list):
            per_model_stats = {f'loss_{i}': 0 for i in range(len(self.model))}
            per_model_stats.update({self.metric_name + f"_{i}": 0 for i in range(len(self.model))})

        for batch_index, batch in enumerate(self.taskloader):
            x, y = self.prepare_batch(batch)

            #loss, y_pred, *base_logs = self.eval_fn(
            result = self.eval_fn(
                self.model,
                self.optimiser,
                self.loss_fn,
                x,
                y,
                n_shot=self.n_shot,
                k_way=self.k_way,
                q_queries=self.q_queries,
                train=False,
                **self.kwargs
            )

            loss = result['meta_batch_loss']
            y_pred = result['task_predictions']
            models_losses = result['models_losses']
            models_preds = result['models_predictions']

            for i, (loss_i, y_pred_i) in enumerate(zip(models_losses, models_preds)):
                y_pred_i = torch.cat(y_pred_i)
                per_model_stats[f'loss_{i}'] += np.mean(loss_i) * y_pred_i.shape[0]
                per_model_stats[self.metric_name + f"_{i}"] += categorical_accuracy(y, y_pred_i) * y_pred_i.shape[0]

            seen += y_pred.shape[0]

            totals['loss'] += loss.item() * y_pred.shape[0]
            totals[self.metric_name] += categorical_accuracy(y, y_pred) * y_pred.shape[0]

        logs[self.prefix + 'loss'] = totals['loss'] / seen
        logs[self.metric_name] = totals[self.metric_name] / seen
        if isinstance(self.model, list):
            for i in range(len(self.model)):
                logs[self.prefix + 'loss_{}'.format(i)] = per_model_stats['loss_{}'.format(i)] / seen
                logs[self.metric_name + "_{}".format(i)] = per_model_stats[self.metric_name + "_{}".format(i)] / seen


def to_numpy(preds):  # [n_models, n_tasks, n_objects, n_classes]
    a = []
    for model_pred in preds:
        # model_preds []
        task_pred = torch.stack(model_pred)
        pred = task_pred.cpu().detach().numpy()
        a.append(pred)
    return np.array(a)


class SaveFewShot(Callback):
    """Evaluate a network on  an n-shot, k-way classification tasks after every epoch.

    # Arguments
        eval_fn: Callable to perform few-shot classification. Examples include `proto_net_episode`,
            `matching_net_episode` and `meta_gradient_step` (MAML).
        num_tasks: int. Number of n-shot classification tasks to evaluate the model with.
        n_shot: int. Number of samples for each class in the n-shot classification tasks.
        k_way: int. Number of classes in the n-shot classification tasks.
        q_queries: int. Number query samples for each class in the n-shot classification tasks.
        task_loader: Instance of NShotWrapper class
        prepare_batch: function. The preprocessing function to apply to samples from the dataset.
        prefix: str. Prefix to identify dataset.
    """

    def __init__(self,
                 eval_fn: Callable,
                 num_tasks: int,
                 n_shot: int,
                 k_way: int,
                 q_queries: int,
                 taskloader: torch.utils.data.DataLoader,
                 prepare_batch: Callable,
                 prefix: str = 'val_',
                 **kwargs):
        super().__init__()
        self.eval_fn = eval_fn
        self.num_tasks = num_tasks
        self.n_shot = n_shot
        self.k_way = k_way
        self.q_queries = q_queries
        self.taskloader = taskloader
        self.prepare_batch = prepare_batch
        self.prefix = prefix
        self.kwargs = kwargs
        self.metric_name = f'{self.prefix}{self.n_shot}-shot_{self.k_way}-way_acc'

    def on_train_begin(self, logs=None):
        self.loss_fn = self.params['loss_fn']
        self.optimiser = self.params['optimiser']

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        test_preds = []
        for batch_index, batch in enumerate(self.taskloader):
            x, y = self.prepare_batch(batch)

            loss, y_pred, *base_logs = self.eval_fn(
                self.model,
                self.optimiser,
                self.loss_fn,
                x,
                y,
                n_shot=self.n_shot,
                k_way=self.k_way,
                q_queries=self.q_queries,
                train=False,
                **self.kwargs
            )
            models_preds = base_logs[1]
            test_preds.append(to_numpy(models_preds))

        name = logs['name']
        train_batches = logs['batches_train']
        test_batches = test_preds

        np.savez('../predictions/' + name + '_train', *train_batches)
        np.savez('../predictions/' + name +'_test', *test_batches)

'''

def prepare_nshot_task(n: int, k: int, q: int) -> Callable:
    """Typical n-shot task preprocessing.

    # Arguments
        n: Number of samples for each class in the n-shot classification task
        k: Number of classes in the n-shot classification task
        q: Number of query samples for each class in the n-shot classification task

    # Returns
        prepare_nshot_task_: A Callable that processes a few shot tasks with specified n, k and q
    """
    def prepare_nshot_task_(batch: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create 0-k label and move to GPU.

        TODO: Move to arbitrary device
        """
        x, y = batch
        x = x.double().cuda()
        # Create dummy 0-(num_classes - 1) label
        y = create_nshot_task_label(k, q).cuda()
        return x, y

    return prepare_nshot_task_
'''


def create_nshot_task_label(k: int, q: int) -> torch.Tensor:
    """Creates an n-shot task label.

    Label has the structure:
        [0]*q + [1]*q + ... + [k-1]*q

    # Arguments
        k: Number of classes in the n-shot classification task
        q: Number of query samples for each class in the n-shot classification task

    # Returns
        y: Label vector for n-shot task of shape [q * k, ]
    """
    y = torch.arange(0, k, 1 / q).long()
    return y
