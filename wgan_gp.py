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
from build_dataset_fid_stats import get_activation, get_activations, get_stats

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
parser.add_argument("--num_samples", type=int, default=10000, help="number of samples")
parser.add_argument("--sample_interval", type=int, default=100, help="interval betwen image samples")
parser.add_argument("--metric_interval", type=int, default=100, help="interval betwen metrics evaluations")
parser.add_argument("--save_model_interval", type=int, default=1000, help="interval betwen model saves")
parser.add_argument('--mr', action='store_true')
parser.add_argument("--mrt", type=float, default=0, help="Minimum memorization rejection threshold, cosine distance have to be greater than mrt")
parser.add_argument("--mrt_decay", type=float, default=0.02, help="Decay for minimum memorization rejection threshold in case nothing satisfies")
parser.add_argument("--load_gen_path", default=None, help="path to load the generator state_dict")
parser.add_argument("--load_dis_path", default=None, help="path to load the discriminator state_dict")
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
                layers.append(nn.BatchNorm2d(out_feat, momentum=0.8))
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

if opt.load_gen_path is not None:
    generator.load_state_dict(torch.load(opt.load_gen_path))
    print(f"Loading {opt.load_gen_path} as generator...")
if opt.load_dis_path is not None:
    discriminator.load_state_dict(torch.load(opt.load_dis_path))
    print(f"Loading {opt.load_dis_path} as discriminator...")

if cuda:
    generator.cuda()
    discriminator.cuda()

summary(generator, input_size=(opt.latent_dim,))
summary(discriminator, input_size=(opt.channels, opt.img_size, opt.img_size))

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


def compute_gradient_penalty(D, real_samples, fake_samples):
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
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

def normalize_rows(x):
    x = x[np.sum(x, axis=1) != 0]
    x = np.nan_to_num(x/np.linalg.norm(x, ord=2, axis=1, keepdims=True))
    return x

# load prep train features
prep_dataset_path = paths['prep_dataset_path']
f = np.load(prep_dataset_path)
real_features = normalize_rows(f['features'][:])
real_features_tensor = torch.from_numpy(real_features).type(torch.FloatTensor).cuda() # [50000, 2048]
real_mu = f['mu'][:]
real_sigma = f['sigma'][:]
f.close()

block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
inception_v3 = InceptionV3([block_idx])
inception_v3.eval()
if cuda:
    inception_v3.cuda()


def compute_memorization_distance(fake_features, use_numpy=False):
    if use_numpy:
        fake_features = normalize_rows(fake_features)
        d = 1.0 - np.abs(np.matmul(fake_features, real_features.T))
        min_d = np.min(d, axis=1)
    else:
        fake_features = torch.div(fake_features, torch.norm(fake_features, dim=1).view(-1, 1))
        d = 1.0 - torch.abs(torch.mm(fake_features, real_features_tensor.T))
        min_d, _ = torch.min(d, dim=1)
    return min_d

'''
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
'''

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

def compute_memorization_rejection_mask(fake_imgs, t):
    # assume fake_imgs is alrealdy normalized to [0, 1]
    # start.record()
    fake_features = get_activation(fake_imgs, inception_v3, cuda=cuda).view(-1, 2048)
    # end.record()
    # torch.cuda.synchronize()
    # print(f"pretrained forward time: {start.elapsed_time(end)}")
    # start.record()
    min_d = compute_memorization_distance(fake_features)
    # end.record()
    # torch.cuda.synchronize()
    # print(f"compute cosine distance: {start.elapsed_time(end)}")
    return min_d > t

# prepare save directory
timestamp = datetime.fromtimestamp(time.time()).strftime('%Y%m%d%H%M')
log_dir = os.path.join(img_dir, timestamp)
os.makedirs(log_dir)

# prepare writer
writer = SummaryWriter(logdir=log_dir)

# define testing generator
def generator_sampler(generator, num_instances, bsize, latent_dim, normalize=True, mr=False):
    with torch.no_grad():
        for i in range(0, num_instances, bsize):
            if mr:
                z = Variable(torch.empty(0, latent_dim))
                fake_imgs = torch.empty(0, 3, 32, 32) 
                if cuda:
                    z = z.cuda()
                    fake_imgs = fake_imgs.cuda()
                mrt = opt.mrt
                while fake_imgs.size()[0] < bsize:
                    z_ = Variable(Tensor(np.random.normal(0, 1, (bsize, latent_dim))))
                    fake_imgs_ = generator(z_)
                    mr_mask = compute_memorization_rejection_mask((fake_imgs_ + 1) / 2, t=mrt)
                    z_ = torch.masked_select(z_, mr_mask.view(-1, 1)).view(-1, latent_dim)
                    fake_imgs_ = torch.masked_select(fake_imgs_, mr_mask.view(-1, 1, 1, 1)).view(-1, 3, 32, 32)
                    z = torch.cat((z, z_))
                    fake_imgs = torch.cat((fake_imgs, fake_imgs_))

                    if fake_imgs_.size()[0] == 0:
                        mrt = max(mrt - opt.mrt_decay, 0)
                z = z[:bsize, :]
                fake_imgs = fake_imgs[:bsize, :, :, :]
            else:
                z = Variable(Tensor(np.random.normal(0, 1, (bsize, latent_dim))))
                fake_imgs = generator(z)
            if i + bsize > num_instances:
                fake_imgs = fake_imgs[i + bsize - num_instances:]
            if normalize:
                fake_imgs += 1
                fake_imgs /= 2
            yield fake_imgs, 0
# ----------
#  Training
# ----------

batches_done = 0
for epoch in range(opt.n_epochs):

    for i, (imgs, _) in enumerate(dataloader):

        # Configure input
        real_imgs = Variable(imgs.type(Tensor))

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Sample noise as generator input

        # Generate a batch of images
        if opt.mr:
            mr_all, mr_acc = 0, 0
            z = Variable(torch.empty(0, opt.latent_dim))
            fake_imgs = torch.empty(0, 3, 32, 32) 
            if cuda:
                z = z.cuda()
                fake_imgs = fake_imgs.cuda()
            mrt = opt.mrt
            while fake_imgs.size()[0] < real_imgs.size()[0]:
                z_ = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))
                fake_imgs_ = generator(z_)
                mr_mask = compute_memorization_rejection_mask((fake_imgs_ + 1) / 2, t=mrt) # remember to normalize from [-1, 1] to [0, 1]
                # start.record()
                z_ = torch.masked_select(z_, mr_mask.view(-1, 1)).view(-1, opt.latent_dim)
                fake_imgs_ = torch.masked_select(fake_imgs_, mr_mask.view(-1, 1, 1, 1)).view(-1, 3, 32, 32)
                # end.record()
                # torch.cuda.synchronize()
                # print(f"masked_select: {start.elapsed_time(end)}")
                z = torch.cat((z, z_))
                fake_imgs = torch.cat((fake_imgs, fake_imgs_))

                if fake_imgs_.size()[0] == 0:
                    mrt = max(mrt - opt.mrt_decay, 0)

                mr_all += opt.batch_size
                mr_acc += torch.sum(mr_mask.double())

            z = z[:real_imgs.size()[0], :]
            fake_imgs = fake_imgs[:real_imgs.size()[0], :, :, :]
            mrr = 1 - mr_acc / mr_all
        else:
            z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))
            fake_imgs = generator(z)

        # Real images
        real_validity = discriminator(real_imgs)
        # Fake images
        fake_validity = discriminator(fake_imgs)
        # Gradient penalty
        gradient_penalty = compute_gradient_penalty(discriminator, real_imgs.data, fake_imgs.data)
        # Adversarial loss
        d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + opt.gp * gradient_penalty

        d_loss.backward()
        optimizer_D.step()

        optimizer_G.zero_grad()

        # Train the generator every n_critic steps
        if i % opt.n_critic == 0:

            # -----------------
            #  Train Generator
            # -----------------

            # Generate a batch of images
            fake_imgs = generator(z) # do not have to perform memorization rejection b/c generator same

            # Loss measures generator's ability to fool the discriminator
            # Train on fake images
            fake_validity = discriminator(fake_imgs)
            g_loss = -torch.mean(fake_validity)

            g_loss.backward()
            optimizer_G.step()

            if opt.verbose:
                if opt.mr:
                    print(
                            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] [mrr: %f] [mrt: %f]"
                        % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item(), mrr.item(), mrt)
                    )
                else:
                    print(
                            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                        % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
                    )
            batches_done += opt.n_critic

    writer.add_scalar('loss/d_loss', d_loss.item(), epoch)
    writer.add_scalar('loss/g_loss', g_loss.item(), epoch)
    if opt.mr:
        writer.add_scalar('stats/mrr', mrr.item(), epoch)
        writer.add_scalar('stats/mrt', mrt, epoch)

    if epoch % opt.sample_interval == 0:
        img_fname = os.path.join(log_dir, f"{epoch}.png")
        save_image(fake_imgs.data[:36], img_fname, nrow=6, normalize=True)

        sampled_images = make_grid(fake_imgs.data[:36], nrow=6, normalize=True)
        writer.add_image('sampled_images', sampled_images, epoch)

    if epoch % opt.metric_interval == 0:
        generator.eval()
        fake_features = get_activations(generator_sampler(generator, opt.num_samples, opt.batch_size, opt.latent_dim, normalize=True, mr=opt.mr), inception_v3, cuda=cuda)
        generator.train()
        # fake_features = get_activation((fake_imgs + 1) / 2, inception_v3, cuda=cuda).view(-1, 2048) # scale [-1, 1] to [0, 1]
        mmd = np.mean(compute_memorization_distance(fake_features, use_numpy=True))
        # mmd = compute_memorization_distance(fake_features).mean()
        fake_mu, fake_sigma = get_stats(fake_features)
        # fake_mu, fake_sigma = get_stats(fake_features.cpu().data.numpy())
        fid = calculate_frechet_distance(fake_mu, fake_sigma, real_mu, real_sigma) # numpy input
        writer.add_scalar('metric/fid', fid, epoch)
        writer.add_scalar('metric/mmd', mmd, epoch)
        print("[Epoch %d/%d] [D loss: %f] [G loss: %f] [fid: %f] [mmd: %f]" % (epoch, opt.n_epochs, d_loss.item(), g_loss.item(), fid, mmd))

    if epoch % opt.save_model_interval == 0 and epoch > 0:
        torch.save(generator.state_dict(), os.path.join(log_dir, f"generator_epoch{epoch}.pth"))
        torch.save(discriminator.state_dict(), os.path.join(log_dir, f"discriminator_epoch{epoch}.pth"))

# final save
torch.save(generator.state_dict(), os.path.join(log_dir, "generator.pth"))
torch.save(discriminator.state_dict(), os.path.join(log_dir, "discriminator.pth"))

export_json = os.path.join(log_dir, "all_scalars.json")
writer.export_scalars_to_json(export_json)
writer.close()
