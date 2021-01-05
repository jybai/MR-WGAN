import argparse
import os
import numpy as np
import math
import sys
from datetime import datetime
import time
import yaml

import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch

from tensorboardX import SummaryWriter
from torchsummary import summary

from fid.inception import InceptionV3
from fid.fid_score import calculate_frechet_distance
from build_dataset_fid_stats import get_activation

class View(nn.Module):
    """https://github.com/pytorch/vision/issues/720#issuecomment-581142009
    """
    def __init__(self, shape):
        super(View, self).__init__()
        self.shape = shape,  # extra comma
    def forward(self, x):
        return x.view(*self.shape)

class Generator(nn.Module):
    def __init__(self, latent_dim, n_channels):
        super(Generator, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.ConvTranspose2d(in_feat, out_feat, kernel_size=5, stride=2, padding=2, output_padding=1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_feat, momentum=0.8))
            layers.append(nn.ReLU(inplace=True))
            return layers

        self.model = nn.Sequential(
            nn.Linear(latent_dim, 4*4*4*latent_dim),
            nn.BatchNorm1d(4*4*4*latent_dim, momentum=0.8), # not aligned with original
            nn.ReLU(inplace=True),
            View([-1, 4*latent_dim, 4, 4]),
            *block(4*latent_dim, 2*latent_dim),
            *block(2*latent_dim, latent_dim),
            nn.ConvTranspose2d(latent_dim, n_channels, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        return img

class Discriminator(nn.Module):
    def __init__(self, latent_dim, n_channels):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(n_channels, latent_dim, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(latent_dim, 2*latent_dim, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(2*latent_dim, 4*latent_dim, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            nn.Linear(4*4*4*latent_dim, 1),
        )

    def forward(self, img):
        validity = self.model(img)
        return validity

class GeneratorMRSampler():
    def __init__(self, generator, latent_dim, device, real_features, 
                 proj_model=None, bsize=64, normalize=True):
        self.g = generator
        self.latent_dim = latent_dim
        self.device = device
        self.bsize = bsize
        self.normalize = normalize

        self.real = torch.from_numpy(real_features).to(self.device)
        self.real = torch.div(self.real, torch.norm(self.real, dim=1).view(-1, 1))

        if proj_model is None:
            block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
            self.proj_model = InceptionV3([block_idx]).to(self.device)
            self.proj_model.eval()

    def __iter__(self):
        return self

    def gen_features(self, n_instances):
        fake_features = []
        n_now = 0
        while n_now < n_instances:
            fake_features_ = self._gen_features()
            n_now += fake_features_.shape[0]
            if n_now > n_instances:
                fake_features_ = fake_features_[n_now - n_instances:]
            fake_features.append(fake_features_)
        fake_features = torch.cat(fake_features)
        assert(fake_features.shape[0] == n_instances)
        return fake_features

    def compute_memorization_distance(self, fake_features_tensor):
        fake_norms = torch.norm(fake_features_tensor, dim=1).view(-1, 1)
        fake_features_tensor = torch.div(fake_features_tensor, fake_norms)
        d = 1.0 - torch.abs(torch.mm(fake_features_tensor, self.real.T))
        min_d, _ = torch.min(d, dim=1)
        return min_d

    def _gen_features(self, z=None):
        if z is None:
            z = self.__next__()
        fake_imgs = self.g(z)
        fake_features_tensor = get_activation((fake_imgs + 1) / 2, self.proj_model, 
                                              device=self.device).view(-1, 2048)
        return fake_features_tensor

    def __next__(self):
        z = Variable(torch.from_numpy(np.random.normal(0, 1, (self.bsize, self.latent_dim))).float()).to(self.device)
        return z

def compute_gradient_penalty(D, real_samples, fake_samples, device):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = torch.from_numpy(np.random.random((real_samples.size(0), 1, 1, 1))).float().to(device)
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = Variable(torch.empty(real_samples.shape[0], 1).fill_(1.0), requires_grad=False).to(device)
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

# benchmark variables
def benchmark(op_name):
    def decorator(func):
        def wrapped(*args, **kwargs):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            output = func(*args, **kwargs)
            end.record()
            torch.cuda.synchronize()
            print(op_name, start.elapsed_time(end))
            return output
        return wrapped
    return decorator

