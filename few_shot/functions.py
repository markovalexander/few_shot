import torch
from torch.nn.functional import log_softmax
from numpy import log as nplog, exp as npexp, sum as npsum
import torch.nn.functional as F
import numpy as np


def softmax_mean(output):
    output = torch.stack(output, dim=0)
    output = log_softmax(output, dim=-1)
    output = torch.mean(output, dim=0)
    return output


def logmeanexp_preds(output):
    output = torch.stack(output, dim=0)
    n_models = len(output)
    output = log_softmax(output, dim=-1)
    output = torch.logsumexp(output, dim=0) - nplog(n_models)  # [k*n, n]
    return output


def mean_softmax(output):
    output = torch.stack(output, dim=0)
    output = torch.mean(output, dim=0)
    output = log_softmax(output, dim=-1)
    return output


def get_pred_fn(args):
    if args.train_pred_mode == "mean":
        train_pred_fn = softmax_mean
    elif args.train_pred_mode == "logprobs":
        train_pred_fn = logmeanexp_preds
    elif args.train_pred_mode == "ms":
        train_pred_fn = mean_softmax
    else:
        raise ValueError("Train-pred-mode is not supported yet.")

    if args.test_pred_mode == "same":
        test_pred_fn = train_pred_fn
    elif args.test_pred_mode == "mean":
        test_pred_fn = softmax_mean
    elif args.test_pred_mode == "logprobs":
        test_pred_fn = logmeanexp_preds
    elif args.test_pred_mode == "ms":
        test_pred_fn = mean_softmax
    else:
        raise ValueError("Test-pred-mode is not supported yet.")

    return train_pred_fn, test_pred_fn


def invsp(x):
    return nplog(npexp(x) - 1)


def softplus(x):
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)


class MixtureLoss(object):
    def __init__(self, n_way, active_losses):
        super().__init__()
        self.n_way = n_way
        self.active_losses = [0 for _ in range(4)]
        for idx in active_losses:
            self.active_losses[idx] = 1

        self.n_losses = npsum(self.active_losses)

        self.losses = [
            lambda x, y: F.multi_margin_loss(x, y, margin=0.9, p=2),
            F.mse_loss,  # one-hot, softmax
            F.cross_entropy,
            F.multilabel_margin_loss # one-hot, softmax
        ]

        self.onehot = [
            False,
            True,
            False,
            True
        ]
        self.softmax = [
            False,
            True,
            False,
            True
        ]
        # Weights are inputs to softplus
        self.weights = np.ones(len(self.losses)) * invsp(1. / self.n_losses)
        # self.weights = torch.nn.Parameter(torch.DoubleTensor(len(self.losses)))
        # self.weights.data.fill_(invsp(1. / self.n_losses))

    def get_weights(self):
        return softplus(self.weights)

    def one_hot_encoding(self, tensor, n_classes):
        ohe = torch.Tensor(tensor.size(0), n_classes).to(tensor.device, dtype=self.weights.dtype)
        ohe.zero_()
        ohe.scatter_(1, tensor[:, None], 1)
        return ohe

    def __call__(self, y_pred, y_true):
        """

        :param y_pred: must be log probabilities
        :param y_true:
        :return:
        """
        oh = self.one_hot_encoding(y_true, self.n_way)
        loss = 0.0
        weights = self.get_weights()
        for i in range(len(self.losses)):

            w = weights[i] * self.active_losses[i]
            pred = y_pred
            if self.softmax[i]:
                pred = torch.exp(y_pred)
            y = y_true
            if self.onehot[i]:
                y = oh
            loss += w * self.losses[i](pred, y)
        return loss
