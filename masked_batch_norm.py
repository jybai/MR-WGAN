"""
source: https://yangkky.github.io/2020/03/16/masked-batch-norm.html
gist: https://gist.github.com/yangkky/364413426ec798589463a3a88be24219

matching real batchnorm in pytorch: https://github.com/pytorch/pytorch/blob/700109eb630b79fd65cb93becb7f2d14f93bdb5c/aten/src/ATen/native/Normalization.cpp#L238-L260
"""
import numpy as np

import torch
import torch.nn as nn
from torch.nn import init

class MaskedBatchNorm(nn.Module):
    """ A masked version of nn.BatchNorm1d. Only tested for 3D inputs.

        Args:
            num_features: :math:`C` from an expected input of size
                :math:`(N, C, L)`
            eps: a value added to the denominator for numerical stability.
                Default: 1e-5
            momentum: the value used for the running_mean and running_var
                computation. Can be set to ``None`` for cumulative moving average
                (i.e. simple average). Default: 0.1
            affine: a boolean value that when set to ``True``, this module has
                learnable affine parameters. Default: ``True``
            track_running_stats: a boolean value that when set to ``True``, this
                module tracks the running mean and variance, and when set to ``False``,
                this module does not track such statistics and always uses batch
                statistics in both training and eval modes. Default: ``True``

        Shape:
            - Input: :math:`(N, C, L)`
            - input_mask: (N, 1, L) tensor of ones and zeros, where the zeros indicate locations not to use.
            - Output: :math:`(N, C)` or :math:`(N, C, L)` (same shape as input)
    """

    def __init__(self, num_features, eps=1e-5, momentum=0.1,
                 affine=True, track_running_stats=True, manual_update=True):
        super(MaskedBatchNorm, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        if affine:
            self.weight = nn.Parameter(torch.Tensor(num_features))
            self.bias = nn.Parameter(torch.Tensor(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.track_running_stats = track_running_stats
        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features))
            self.register_buffer('running_var', torch.ones(num_features))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_var', None)
        self.manual_update = manual_update
        self.input_cache = None
        self.reset_parameters()

    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.fill_(1)

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            init.ones_(self.weight)
            init.zeros_(self.bias)

    def forward(self, input):
        # Update cache
        if self.track_running_stats and self.training and self.manual_update:
            if self.input_cache is not None:
                raise ValueError('Should clear update running stats with cache before next forward pass when manual_update = True')
            self.input_cache = input
        # Calculate the mean and variance
        B = input.shape[0]
        C = input.shape[1]
        L = np.prod(input.shape[2:]) # if no shape[2] returns 1
        if C != self.num_features:
            raise ValueError('Expected %d channels but input has %d channels' % (self.num_features, C))
        n = B * L
        reduce_dims = [i for i in range(len(input.shape)) if i != 1]
        view_shape = [-1] + [1] * len(input.shape[2:])
        current_mean = input.mean(reduce_dims)
        current_var = input.var(reduce_dims, unbiased=False)
        running_mean = current_mean if self.training else self.running_mean
        running_var = current_var if self.training else self.running_var
        # Calculate running stats
        if self.track_running_stats and self.training:
            running_mean = (1 - self.momentum) * self.running_mean + self.momentum * current_mean
            running_var = (1 - self.momentum) * self.running_var + self.momentum * current_var * n / (n - 1)
            # Commit running stats if manual_update = False
            if not self.manual_update:
                assert(running_mean is not None and running_var is not None)
                self.running_mean = running_mean
                self.running_var = running_var
        # Norm the input
        if self.track_running_stats and not self.training:
            normed = (input - running_mean.view(view_shape)) / (torch.sqrt(running_var.view(view_shape) + self.eps))
        else:
            normed = (input - current_mean.view(view_shape)) / (torch.sqrt(current_var.view(view_shape) + self.eps))
        # Apply affine parameters
        if self.affine:
            normed = normed * self.weight.view(view_shape) + self.bias.view(view_shape)
        # print('fwd', running_mean, running_var)
        return normed

    def update_masked_running_stats(self, input_mask):
        if not (self.manual_update and self.track_running_stats and self.training):
            raise ValueError('Should only call update_running_stats when self.manual_update = True, self.track_running_stats = True, and self.training = True.')
        if self.input_cache is None:
            raise ValueError('Should perform forward pass and cache input before calling update_running_stats when manual_update = True')
        B = self.input_cache.shape[0]
        C = self.input_cache.shape[1]
        L = np.prod(self.input_cache.shape[2:])
        reduce_dims = [i for i in range(len(self.input_cache.shape)) if i != 1]
        view_shape = [-1] + [1] * len(self.input_cache.shape[2:])
        if input_mask.shape != (B,):
            raise ValueError(f"Mask should have shape {(B,)} instead have {input_mask.shape}.")
        if C != self.num_features:
            raise ValueError('Expected %d channels but input has %d channels' % (self.num_features, C))
        # Apply input_mask
        mask_view = [-1] + [1] * len(self.input_cache.shape[1:])
        masked = self.input_cache * input_mask.view(mask_view)
        n = input_mask.sum() * L
        # Sum
        masked_sum = masked.sum(reduce_dims)
        # Divide by sum of mask
        current_mean = masked_sum / n
        current_var = ((masked - current_mean.view(view_shape)) ** 2).sum(reduce_dims) / n
        # Update running stats
        self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * current_mean
        self.running_var = (1 - self.momentum) * self.running_var + self.momentum * current_var * n / (n - 1)
        # Clear cache
        self.input_cache = None

class MaskedBatchNorm1d(nn.Module):
    """ A masked version of nn.BatchNorm1d. Only tested for 3D inputs.

        Args:
            num_features: :math:`C` from an expected input of size
                :math:`(N, C, L)`
            eps: a value added to the denominator for numerical stability.
                Default: 1e-5
            momentum: the value used for the running_mean and running_var
                computation. Can be set to ``None`` for cumulative moving average
                (i.e. simple average). Default: 0.1
            affine: a boolean value that when set to ``True``, this module has
                learnable affine parameters. Default: ``True``
            track_running_stats: a boolean value that when set to ``True``, this
                module tracks the running mean and variance, and when set to ``False``,
                this module does not track such statistics and always uses batch
                statistics in both training and eval modes. Default: ``True``

        Shape:
            - Input: :math:`(N, C, L)`
            - input_mask: (N, 1, L) tensor of ones and zeros, where the zeros indicate locations not to use.
            - Output: :math:`(N, C)` or :math:`(N, C, L)` (same shape as input)
    """

    def __init__(self, num_features, eps=1e-5, momentum=0.1,
                 affine=True, track_running_stats=True, manual_update=True):
        super(MaskedBatchNorm1d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        if affine:
            self.weight = nn.Parameter(torch.Tensor(num_features, 1))
            self.bias = nn.Parameter(torch.Tensor(num_features, 1))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.track_running_stats = track_running_stats
        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features, 1))
            self.register_buffer('running_var', torch.ones(num_features, 1))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_var', None)
        self.manual_update = manual_update
        self.input_cache = None
        self.reset_parameters()

    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.fill_(1)

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            init.ones_(self.weight)
            init.zeros_(self.bias)

    def forward(self, input):
        # expand input size
        squeeze = False
        if len(input.size()) == 2:
            input = torch.unsqueeze(input, -1)
            squeeze = True
        # Update cache
        if self.track_running_stats and self.training and self.manual_update:
            if self.input_cache is not None:
                raise ValueError('Should clear update running stats with cache before next forward pass when manual_update = True')
            self.input_cache = input
        # Calculate the mean and variance
        B, C, L = input.shape
        if C != self.num_features:
            raise ValueError('Expected %d channels but input has %d channels' % (self.num_features, C))
        n = B * L
        current_mean = input.mean([0, 2], keepdim=True)
        # current_mean = input.sum(dim=0, keepdim=True).sum(dim=2, keepdim=True) / n
        current_var = input.var([0, 2], unbiased=False, keepdim=True)
        # current_var = ((input - current_mean) ** 2).sum(dim=0, keepdim=True).sum(dim=2, keepdim=True) / (n - 1)
        running_mean = current_mean if self.training else self.running_mean
        # running_mean = self.running_mean
        running_var = current_var if self.training else self.running_var
        # running_var = self.running_var
        # Calculate running stats
        if self.track_running_stats and self.training:
            running_mean = (1 - self.momentum) * self.running_mean + self.momentum * current_mean
            running_var = (1 - self.momentum) * self.running_var + self.momentum * current_var * n / (n - 1)
            # Commit running stats if manual_update = False
            if not self.manual_update:
                assert(running_mean is not None and running_var is not None)
                self.running_mean = running_mean
                self.running_var = running_var
        # Norm the input
        if self.track_running_stats and not self.training:
            normed = (input - running_mean) / (torch.sqrt(running_var + self.eps))
        else:
            normed = (input - current_mean) / (torch.sqrt(current_var + self.eps))
        # Apply affine parameters
        if self.affine:
            normed = normed * self.weight + self.bias
        # print('fwd', running_mean, running_var)
        if squeeze:
            normed = torch.squeeze(normed, -1)
        return normed

    def update_masked_running_stats(self, input_mask):
        if not (self.manual_update and self.track_running_stats and self.training):
            raise ValueError('Should only call update_running_stats when self.manual_update = True, self.track_running_stats = True, and self.training = True.')
        if self.input_cache is None:
            raise ValueError('Should perform forward pass and cache input before calling update_running_stats when manual_update = True')
        B, C, L = self.input_cache.shape
        if input_mask.shape != (B, 1, L):
            raise ValueError('Mask should have shape (B, 1, L).')
        if C != self.num_features:
            raise ValueError('Expected %d channels but input has %d channels' % (self.num_features, C))
        # Apply input_mask
        masked = self.input_cache * input_mask
        n = input_mask.sum()
        # Sum
        masked_sum = masked.sum(dim=0, keepdim=True).sum(dim=2, keepdim=True)
        # Divide by sum of mask
        current_mean = masked_sum / n
        current_var = ((masked - current_mean) ** 2).sum(dim=0, keepdim=True).sum(dim=2, keepdim=True) / n
        # Update running stats
        self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * current_mean
        self.running_var = (1 - self.momentum) * self.running_var + self.momentum * current_var * n / (n - 1)
        # Clear cache
        self.input_cache = None


def compare_bn(bn1, bn2):
    err = False
    if bn1.track_running_stats and bn2.track_running_stats:
        if not torch.allclose(bn1.running_mean, bn2.running_mean):
            print('Diff in running_mean: {} vs {}'.format(
                bn1.running_mean, bn2.running_mean))
            err = True

        if not torch.allclose(bn1.running_var, bn2.running_var):
            print('Diff in running_var: {} vs {}'.format(
                bn1.running_var, bn2.running_var))
            err = True

    if bn1.affine and bn2.affine:
        if not torch.allclose(bn1.weight, bn2.weight):
            print('Diff in weight: {} vs {}'.format(
                bn1.weight, bn2.weight))
            err = True

        if not torch.allclose(bn1.bias, bn2.bias):
            print('Diff in bias: {} vs {}'.format(
                bn1.bias, bn2.bias))
            err = True

    if not err:
        print('All parameters are equal!')

def main():
    """Test whether MaskedBatchNorm1d behaves the same as BatchNorm1d
    self.training X [self.track_running_stats X self.manual_update]
    """
    runs = 10
    B = 8
    num_features = 8
    L = 32
    # iterate through all cases
    for track_running_stats in [True, False]:
        for manual_update in [True, False]:
            print(f"----------- Setting: track_running_stats = {track_running_stats}, manual_update = {manual_update} ------------")
            criterion = nn.MSELoss()
            # train mbn
            mbn = MaskedBatchNorm(num_features, affine=True, track_running_stats=track_running_stats, 
                                    manual_update=manual_update)
            # train bn
            bn = nn.BatchNorm2d(num_features, affine=True, track_running_stats=track_running_stats)
            for _ in range(runs):
                x = torch.randn(B, num_features, L, L)
                out_mbn = mbn(x)
                out_bn = bn(x)
                # assert(torch.allclose(out_mbn.data, out_bn.data))
                print(torch.allclose(out_mbn.data, out_bn.data), (out_mbn - out_bn).abs().max().data)
                assert(out_mbn.size() == out_bn.size())
                if manual_update and track_running_stats:
                    mbn.update_masked_running_stats(input_mask=torch.ones(B))
                compare_bn(mbn, bn)
            # eval
            mbn.eval()
            bn.eval()
            for _ in range(runs):
                x = torch.randn(B, num_features, L, L)
                out_mbn = mbn(x)
                out_bn = bn(x)
                # assert(torch.allclose(out_mbn.data, out_bn.data))
                print(torch.allclose(out_mbn.data, out_bn.data), (out_mbn - out_bn).abs().max().data)
                assert(out_mbn.size() == out_bn.size())
                compare_bn(mbn, bn)
if __name__ == '__main__':
    main()
