import argparse
import os
import numpy as np
import math
import sys
from datetime import datetime
import time

import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch

from fid.inception import InceptionV3

def get_activation(batch, model, device, dims=2048):
    batch = batch.to(device)

    pred = model(batch)[0]

    # If model output is not scalar, apply global spatial average pooling.
    # This happens if you choose a dimensionality not equal 2048.
    if pred.size(2) != 1 or pred.size(3) != 1:
        pred = F.adaptive_avg_pool2d(pred, output_size=(1, 1))

    return pred

def get_activations(dataloader, model, device, dims=2048,
                    verbose=False, use_numpy=True):
    """Calculates the activations of the pool_3 layer for all images.

    Params:
    -- dataloader  : dataloader (batched)
    -- model       : Instance of inception model
    -- dims        : Dimensionality of features returned by Inception
    -- cuda        : If set to True, use GPU
    -- verbose     : If set to True and parameter out_step is given, the number
                     of calculated batches is reported.
    Returns:
    -- A numpy array of dimension (num images, dims) that contains the
       activations of the given tensor when feeding inception with the
       query tensor.
    """
    model.eval()

    pred_arr = []
    for i, (batch, _) in enumerate(dataloader):
        if verbose:
            print('\rPropagating batch %d/%d' % (i + 1, len(dataloader)),
                  end='', flush=True)
        '''
        # Reshape to (n_images, 3, height, width)
        images = images.transpose((0, 3, 1, 2))
        images /= 255 # to [0, 1]

        batch = torch.from_numpy(images).type(torch.FloatTensor)
        '''
        pred = get_activation(batch, model, device, dims).reshape(-1, dims)
        pred_arr.append(pred)

    if verbose:
        print(' done')

    pred_arr = torch.cat(pred_arr, dim=0)

    return pred_arr if not use_numpy else pred_arr.cpu().data.numpy()

def get_stats(features):
    mu = np.mean(features, axis=0)
    sigma = np.cov(features, rowvar=False)
    return mu, sigma

def prepare_data(batch_size):
    # Configure data loader
    data_dir = "/home/cybai2020/PyTorch-GAN/data/cifar10"
    os.makedirs(data_dir, exist_ok=True)

    dataloader = torch.utils.data.DataLoader(
        datasets.CIFAR10(
            data_dir,
            train=True,
            download=True,
            transform=transforms.Compose(
                [transforms.ToTensor()]
            ),
        ),
        batch_size=batch_size,
        shuffle=False,
    )
    return dataloader

def main():
    cuda = True
    batch_size = 64
    dims = 2048
    prep_dataset_stats_path = 'cifar10-train.npz'

    if not os.path.isfile(prep_dataset_stats_path):
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
        model = InceptionV3([block_idx])
        if cuda:
            model.cuda()

        dataloader = prepare_data(batch_size)
        features = get_activations(dataloader, model, dims, cuda, verbose=False)
        mu, sigma = get_stats(features)

        np.savez(prep_dataset_stats_path, features=features, mu=mu, sigma=sigma)

    else:
        f = np.load(prep_dataset_stats_path)
        features, mu, sigma = f['features'][:], f['mu'][:], f['sigma'][:]
        f.close()

    print(features.shape)
    print(mu.shape)
    print(sigma.shape)

if __name__ == '__main__':
    main()
