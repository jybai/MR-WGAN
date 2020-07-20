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
from build_dataset_fid_stats import get_activation

from masked_batch_norm import MaskedBatchNorm

path_file = 'path.yml'
with open(path_file) as f:
    paths = yaml.load(f, Loader=yaml.FullLoader)

parser = argparse.ArgumentParser()
parser.add_argument('--verbose', action='store_true')
parser.add_argument("--n_epochs", type=int, default=200000, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.00005, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.9, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=128, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--n_critic", type=int, default=5, help="number of training steps for discriminator per iter")
parser.add_argument("--gp", type=float, default=10., help="Loss weight for gradient penalty")
parser.add_argument("--mrt", type=float, default=0, help="Minimum memorization rejection threshold, cosine distance have to be greater than mrt")
parser.add_argument("--mrt_decay", type=float, default=0.01, help="Decay for minimum memorization rejection threshold in case nothing satisfies")
parser.add_argument("--sample_interval", type=int, default=100, help="interval betwen image samples")
opt = parser.parse_args()
print(opt)

img_dir = paths['img_dir']
os.makedirs(img_dir, exist_ok=True)

img_shape = (opt.channels, opt.img_size, opt.img_size)

cuda = True if torch.cuda.is_available() else False

class View(nn.Module):
    """https://github.com/pytorch/vision/issues/720#issuecomment-581142009
    """
    def __init__(self, shape):
        super(View, self).__init__()
        self.shape = shape,  # extra comma
    def forward(self, x):
        return x.view(*self.shape)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.ConvTranspose2d(in_feat, out_feat, kernel_size=5, stride=2, padding=2, output_padding=1)]
            if normalize:
                layers.append(MaskedBatchNorm(out_feat, momentum=0.8))
                # layers.append(nn.BatchNorm2d(out_feat, momentum=0.8))
            layers.append(nn.ReLU(inplace=True))
            return layers

        self.model = nn.Sequential(
            nn.Linear(opt.latent_dim, 4*4*4*opt.latent_dim),
            nn.BatchNorm1d(4*4*4*opt.latent_dim, momentum=0.8), # not aligned with original
            nn.ReLU(inplace=True),
            View([-1, 4*opt.latent_dim, 4, 4]),
            *block(4*opt.latent_dim, 2*opt.latent_dim),
            *block(2*opt.latent_dim, opt.latent_dim),
            nn.ConvTranspose2d(opt.latent_dim, opt.channels, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        # img = img.view(img.shape[0], *img_shape)
        return img

    def update_masked_bn(self, mask):
        for m in self.model.modules():
            if isinstance(m, MaskedBatchNorm):
                m.update_masked_running_stats(mask)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(opt.channels, opt.latent_dim, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(opt.latent_dim, 2*opt.latent_dim, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(2*opt.latent_dim, 4*opt.latent_dim, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            nn.Linear(4*4*4*opt.latent_dim, 1),
        )

    def forward(self, img):
        # img_flat = img.view(img.shape[0], -1)
        # validity = self.model(img_flat)
        validity = self.model(img)
        return validity

# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

if cuda:
    generator.cuda()
    discriminator.cuda()

# will perform one forward pass, mess with bn stats
'''
summary(generator, input_size=(opt.latent_dim,))
summary(discriminator, input_size=(opt.channels, opt.img_size, opt.img_size))
'''

# Configure data loader
data_dir = paths['data_dir']
os.makedirs(data_dir, exist_ok=True)
dataloader = torch.utils.data.DataLoader(
    datasets.CIFAR10(
        data_dir,
        train=True,
        download=True,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize([0.5] * opt.channels, [0.5] * opt.channels)]
        ),
    ),
    batch_size=opt.batch_size,
    shuffle=True,
    drop_last=True,
)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


def compute_gradient_penalty(D, real_samples, fake_samples, mask):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = Tensor(np.random.random((real_samples.size(0), 1, 1, 1)))
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = Variable(Tensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False)
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
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2 * mask).sum() / mask.sum()
    return gradient_penalty

# load prep train features
prep_dataset_path = paths['prep_dataset_path']
f = np.load(prep_dataset_path)
real_features = torch.from_numpy(f['features'][:]).type(torch.FloatTensor).cuda() # [50000, 2048]
f.close()
real_features = torch.div(real_features, torch.norm(real_features, dim=1).view(-1, 1))

block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
inception_v3 = InceptionV3([block_idx])
if cuda:
    inception_v3.cuda()
inception_v3.eval()

def compute_memorization_rejection_mask(fake_imgs, t):
    fake_features = get_activation(fake_imgs, inception_v3, cuda=cuda).view(-1, 2048)
    fake_features = torch.div(fake_features, torch.norm(fake_features, dim=1).view(-1, 1))
    d = 1.0 - torch.abs(torch.mm(fake_features, real_features.T))
    min_d, _ = torch.min(d, dim=1)
    return (min_d > t).double()

# prepare save directory
timestamp = datetime.fromtimestamp(time.time()).strftime('%Y%m%d%H%M')
log_dir = os.path.join(img_dir, timestamp)
os.makedirs(log_dir)

# prepare writer
writer = SummaryWriter(logdir=log_dir)

# ----------
#  Training
# ----------

batches_done = 0
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
start_mr = torch.cuda.Event(enable_timing=True)
end_mr = torch.cuda.Event(enable_timing=True)
for epoch in range(opt.n_epochs):
    for i, (imgs, _) in enumerate(dataloader):
        if opt.verbose:
            if i > 0:
                end.record()
                torch.cuda.synchronize()
                print(f"Batch {i}", start.elapsed_time(end), start_mr.elapsed_time(end_mr))
            start.record()
        # Configure input
        real_imgs = Variable(imgs.type(Tensor))

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Sample noise as generator input
        z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))

        # Generate a batch of images
        if opt.verbose:
            start_mr.record()
        fake_imgs = generator(z)
        with torch.no_grad():
            mr_mask = compute_memorization_rejection_mask(fake_imgs, t=opt.mrt)
        generator.update_masked_bn(mr_mask)
        if opt.verbose:
            end_mr.record()

        # Real images
        real_validity = discriminator(real_imgs)
        # Fake images
        fake_validity = discriminator(fake_imgs)
        # Gradient penalty
        gradient_penalty = compute_gradient_penalty(discriminator, real_imgs.data, fake_imgs.data, mr_mask)
        # Adversarial loss
        d_loss = -(real_validity.squeeze() * mr_mask).sum() / mr_mask.sum() + (fake_validity.squeeze() * mr_mask).sum() / mr_mask.sum() + opt.gp * gradient_penalty

        d_loss.backward()
        optimizer_D.step()

        optimizer_G.zero_grad()

        # Train the generator every n_critic steps
        if i % opt.n_critic == 0:

            # -----------------
            #  Train Generator
            # -----------------

            # Generate a batch of images
            fake_imgs = generator(z)
            with torch.no_grad():
                mr_mask = compute_memorization_rejection_mask(fake_imgs, t=opt.mrt)
            generator.update_masked_bn(mr_mask)
            mrr = 1 - torch.mean(mr_mask)

            # Loss measures generator's ability to fool the discriminator
            # Train on fake images
            fake_validity = discriminator(fake_imgs)
            g_loss = -(fake_validity.squeeze() * mr_mask).sum() / mr_mask.sum()

            g_loss.backward()
            optimizer_G.step()

            print(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] [mrr: %f]"
                % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item(), mrr.item())
            )
            batches_done += opt.n_critic

    writer.add_scalar('loss/d_loss', d_loss.item(), epoch)
    writer.add_scalar('loss/g_loss', g_loss.item(), epoch)
    writer.add_scalar('stats/mrr', mrr.item(), epoch)

    if epoch % opt.sample_interval == 0:
        img_fname = os.path.join(log_dir, f"{epoch}.png")
        accepted_fake_imgs = torch.masked_select(fake_imgs, mr_mask.bool().view(-1, 1, 1, 1)).view(-1, 3, 32, 32)
        save_image(accepted_fake_imgs.data[:25], img_fname, nrow=5, normalize=True)

        sampled_images = make_grid(accepted_fake_imgs.data[:25], nrow=5, normalize=True)
        writer.add_image('sampled_images', sampled_images, epoch)

export_json = os.path.join(log_dir, "all_scalars.json")
writer.export_scalars_to_json(export_json)
writer.close()
