from torch.utils.data import DataLoader
from torch import nn
import torch.nn.functional as F
import argparse
import torch
import random

import sys
sys.path.append('..')

from few_shot.datasets import OmniglotDataset, MiniImageNet
from few_shot.core import NShotTaskSampler, create_nshot_task_label, EvaluateFewShot
from few_shot.maml_ens_mgpu import meta_gradient_ens_step_mgpu_2order, \
    meta_gradient_ens_step_mgpu_1order, meta_gradient_ens_step_mgpu_2order_mixture
from few_shot.maml_mean_loss import meta_gradient_ens_step_mgpu_meanloss
from few_shot.models import FewShotClassifier
from few_shot.train import fit
from few_shot.callbacks import *
from few_shot.utils import setup_dirs
from config import PATH
from few_shot.functions import get_pred_fn, logmeanexp_preds, MixturePredsLoss

setup_dirs()
assert torch.cuda.is_available()

device = torch.device('cuda')
torch.backends.cudnn.benchmark = True


##############
# Parameters #
##############
parser = argparse.ArgumentParser()
parser.add_argument('--dataset')
parser.add_argument('--n', default=1, type=int)
parser.add_argument('--k', default=5, type=int)
parser.add_argument('--q', default=1, type=int)  # Number of examples per class to calculate meta gradients with
parser.add_argument('--inner-train-steps', default=1, type=int)
parser.add_argument('--inner-val-steps', default=3, type=int)
parser.add_argument('--inner-lr', default=0.4, type=float)
parser.add_argument('--meta-lr', default=0.001, type=float)
parser.add_argument('--meta-batch-size', default=32, type=int)
parser.add_argument('--order', default=1, type=int,
                    help="1, 2 or 0 (0 for mean losses)")
parser.add_argument('--epochs', default=50, type=int)
parser.add_argument('--epoch-len', default=100, type=int)
parser.add_argument('--eval-batches', default=20, type=int)
parser.add_argument('--n-models', default=3, type=int)
parser.add_argument('--weight', default=0.5, type=float)
parser.add_argument('--train-pred-mode', default='mean', type=str)
parser.add_argument('--test-pred-mode', default='same', type=str)

args = parser.parse_args()

if args.dataset == 'omniglot':
    dataset_class = OmniglotDataset
    fc_layer_size = 64
    num_input_channels = 1
elif args.dataset == 'miniImageNet':
    dataset_class = MiniImageNet
    fc_layer_size = 1600
    num_input_channels = 3
else:
    raise(ValueError('Unsupported dataset'))

param_str = f'{args.dataset}_order={args.order}_n={args.n}_k={args.k}_metabatch={args.meta_batch_size}_' \
            f'train_steps={args.inner_train_steps}_val_steps={args.inner_val_steps}_n_models={args.n_models}_' \
            f'lambda={args.weight}'
print(param_str)


###################
# Create datasets #
###################
background = dataset_class('background')
background_taskloader = DataLoader(
    background,
    batch_sampler=NShotTaskSampler(background, args.epoch_len, n=args.n, k=args.k, q=args.q,
                                   num_tasks=args.meta_batch_size),
    num_workers=8
)
evaluation = dataset_class('evaluation')
evaluation_taskloader = DataLoader(
    evaluation,
    batch_sampler=NShotTaskSampler(evaluation, args.eval_batches, n=args.n, k=args.k, q=args.q,
                                   num_tasks=args.meta_batch_size),
    num_workers=8
)

############
# Training #
############
print(f'Training MAML on {args.dataset}...')

model_params = [num_input_channels, args.k, fc_layer_size]
meta_models = [FewShotClassifier(num_input_channels, args.k, fc_layer_size).to(device, dtype=torch.double)
               for _ in range(args.n_models)]
meta_optimisers = [torch.optim.Adam(meta_model.parameters(), lr=args.meta_lr)
                   for meta_model in meta_models]


loss_fn = MixturePredsLoss(loss_fn=F.cross_entropy, lmbda=args.weight)
fit_fn = meta_gradient_ens_step_mgpu_2order_mixture

train_pred_fn, test_pred_fn = logmeanexp_preds, logmeanexp_preds


def prepare_meta_batch(n, k, q, meta_batch_size):
    def prepare_meta_batch_(batch):
        x, y = batch
        # Reshape to `meta_batch_size` number of tasks. Each task contains
        # n*k support samples to train the fast model on and q*k query samples to
        # evaluate the fast model on and generate meta-gradients
        x = x.reshape(meta_batch_size, n*k + q*k, num_input_channels, x.shape[-2], x.shape[-1])
        # Move to device
        x = x.double().to(device)
        # Create label
        y = create_nshot_task_label(k, q).cuda().repeat(meta_batch_size)
        return x, y

    return prepare_meta_batch_


ReduceLRCallback = [ReduceLROnPlateau(patience=10, factor=0.5, monitor=f'val_loss', index=i)
                    for i in range(len(meta_optimisers))]
ReduceLRCallback = CallbackList(ReduceLRCallback)

hash = ''.join([chr(random.randint(97, 122)) for _ in range(3)])
callbacks = [
    EvaluateFewShot(
        eval_fn=fit_fn,
        num_tasks=args.eval_batches,
        n_shot=args.n,
        k_way=args.k,
        q_queries=args.q,
        taskloader=evaluation_taskloader,
        prepare_batch=prepare_meta_batch(args.n, args.k, args.q, args.meta_batch_size),
        loss_fn=loss_fn,
        # MAML kwargs
        inner_train_steps=args.inner_val_steps,
        inner_lr=args.inner_lr,
        device=device,
        order=args.order,
        model_params=model_params,
        pred_fn=test_pred_fn
    ),
    EvaluateFewShot(
        eval_fn=meta_gradient_ens_step_mgpu_2order,
        num_tasks=args.eval_batches,
        n_shot=args.n,
        k_way=args.k,
        q_queries=args.q,
        taskloader=evaluation_taskloader,
        prepare_batch=prepare_meta_batch(args.n, args.k, args.q, args.meta_batch_size),
        prefix="val_logprobs_",
        loss_fn=F.nll_loss,
        # MAML kwargs
        inner_train_steps=args.inner_val_steps,
        inner_lr=args.inner_lr,
        device=device,
        order=args.order,
        model_params=model_params,
        pred_fn=logmeanexp_preds
    ),
    EnsembleCheckpoint(
        filepath=PATH + f'/models/maml_ens/mgpu-mixture_{param_str}.pth',
        monitor=f'val_{args.n}-shot_{args.k}-way_acc',
        hash=hash
    ),
    ReduceLRCallback,
    CSVLogger(PATH + f'/logs/maml_ens/mgpu-mixture_{param_str}.csv',
              hash=hash),  # TODO: add logprobs and mean loss values from mixture
]


fit(
    meta_models,
    meta_optimisers,
    loss_fn,
    epochs=args.epochs,
    dataloader=background_taskloader,
    prepare_batch=prepare_meta_batch(args.n, args.k, args.q, args.meta_batch_size),
    callbacks=callbacks,
    metrics=['categorical_accuracy'],
    fit_function=fit_fn,
    n_models=args.n_models,
    fit_function_kwargs={'n_shot': args.n, 'k_way': args.k, 'q_queries': args.q,
                         'train': True, 'order': args.order, 'device': device,
                         'inner_train_steps': args.inner_train_steps,
                         'inner_lr': args.inner_lr, 'model_params': model_params,
                         'pred_fn': train_pred_fn},
)
