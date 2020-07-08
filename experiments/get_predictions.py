from torch.utils.data import DataLoader
from torch import nn
import torch.nn.functional as F
import argparse
import torch
import random

import sys
sys.path.append('..')

from few_shot.datasets import OmniglotDataset, MiniImageNet
from few_shot.core import NShotTaskSampler, create_nshot_task_label, SaveFewShot
from few_shot.maml_ens_mgpu import meta_gradient_ens_step_mgpu_2order, \
    meta_gradient_ens_step_mgpu_1order
from few_shot.maml_mean_loss import meta_gradient_ens_step_mgpu_meanloss
from few_shot.models import FewShotClassifier
from few_shot.train import fit, save_res
from few_shot.callbacks import *

assert torch.cuda.is_available()

device = torch.device('cuda')
torch.backends.cudnn.benchmark = True

dataset_class = MiniImageNet
fc_layer_size = 1600
num_input_channels = 3

names = os.listdir('../logs/maml_ens/')

model_names = sorted(os.listdir('maml_ens/'))


def parse_name(filename):
    params = filename.split('_')
    n = params[3][2:]
    k = params[4][2:]
    n_models = params[-3][7:]
    predmode = params[-1][5:-7]
    order = params[2][-1]
    params = {'n': int(n), 'k' : int(k), 'n_models': int(n_models), 'pred_mode': predmode,
             'order': order}
    return params


models = []
for exp in names:
    params = parse_name(exp)
    if params['pred_mode'] != 'logprobs':
        continue
    else:
        exp_models = []
        for name in model_names:
            check_name = exp.split('.')[0] + '.pth' + exp[-3:]
            if name.startswith(check_name):
                exp_models.append(name)
        models.append(exp_models)

###################
# Create datasets #
###################

def run_one(names):
    class args:
        epoch_len = 800
        n = 5
        k = 5
        q = 5
        meta_batch_size = 2
        n_models = len(names)
        eval_batches = 80
        pred_mode = 'logprobs'
        order = 2
        epochs = 1
        inner_train_steps = 5
        inner_val_steps = 10
        inner_lr = 0.01

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

    for i, (model, name) in enumerate(zip(meta_models, names)):
        model.load_state_dict(torch.load(name + f'_{i}.pth'))

    loss_fn = F.nll_loss if args.order > 0 else F.cross_entropy

    if args.order == 2:
        fit_fn = meta_gradient_ens_step_mgpu_2order
    elif args.order == 1:
        fit_fn = meta_gradient_ens_step_mgpu_1order
    else:
        fit_fn = meta_gradient_ens_step_mgpu_meanloss


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

    callbacks = [
        SaveFewShot(
            eval_fn=fit_fn,
            num_tasks=args.eval_batches,
            n_shot=args.n,
            k_way=args.k,
            q_queries=args.q,
            taskloader=evaluation_taskloader,
            prepare_batch=prepare_meta_batch(args.n, args.k, args.q, args.meta_batch_size),
            # MAML kwargs
            inner_train_steps=args.inner_val_steps,
            inner_lr=args.inner_lr,
            device=device,
            order=args.order,
            pred_mode=args.pred_mode,
            model_params=model_params,
            name=names[0]
        )
    ]

    save_res(
        meta_models,
        meta_optimisers,
        loss_fn,
        epochs=args.epochs,
        dataloader=background_taskloader,
        prepare_batch=prepare_meta_batch(args.n, args.k, args.q, args.meta_batch_size),
        callbacks=callbacks,
        metrics=['categorical_accuracy'],
        fit_function=fit_fn,
        fit_function_kwargs={'n_shot': args.n, 'k_way': args.k, 'q_queries': args.q,
                             'train': False, 'pred_mode': args.pred_mode,
                             'order': args.order, 'device': device,
                             'inner_train_steps': args.inner_train_steps,
                             'inner_lr': args.inner_lr, 'model_params': model_params,
                             'name': names[0]},
    )


def main():
    for i, names in enumerate(model_names):
        print(f'Doing {i+1} ensemble of {len(model_names)}...')
        run_one(names)
