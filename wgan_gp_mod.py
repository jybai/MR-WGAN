import argparse
import os
import numpy as np
import math
import sys
from datetime import datetime
import time
import yaml
import GPUtil

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

def mask_gpu(gpu_index=None):
    gpus = GPUtil.getGPUs()

    if gpu_index is None:
        mem_frees = [gpu.memoryFree for gpu in gpus]
        gpu_index = mem_frees.index(max(mem_frees))

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpus[gpu_index].id)

def seed_everything(seed=1126):
    os.environ['PYTHONHASHSEED']=str(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)

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
    def __init__(self, generator, mr, mrt, mrt_decay, latent_dim, device, 
                 real_features, proj_model=None, bsize=64, normalize=True):
        self.g = generator
        self.mr = mr
        self.mrt = mrt
        self.final_mrt = mrt
        self.mrt_decay = mrt_decay
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

        if self.mr:
            self.n_total_samples = 0
            self.n_accepted_samples = 0

    def reset_running_stats(self):
        if self.mr:
            self.n_total_samples = 0
            self.n_accepted_samples = 0
            # self.final_mrt = self.mrt

    def __iter__(self):
        return self

    def get_mrr(self):
        if self.n_total_samples == 0:
            return 0
        else:
            return 1 - self.n_accepted_samples / self.n_total_samples

    def get_mrt(self):
        return self.final_mrt

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
        with torch.no_grad():
            if self.mr:
                n_total_samples, n_accepted_samples = 0, 0
                z = []
                self.final_mrt = self.mrt

                while n_accepted_samples < self.bsize:
                    z_ = Variable(torch.from_numpy(np.random.normal(0, 1, (self.bsize, self.latent_dim))).float()).to(self.device)
                    fake_features_tensor = self._gen_features(z_)

                    min_d = self.compute_memorization_distance(fake_features_tensor)
                    mr_mask = min_d > self.final_mrt
                    z_ = torch.masked_select(z_, mr_mask.view(-1, 1)).view(-1, self.latent_dim)

                    z.append(z_)
                    n_total_samples += self.bsize
                    n_accepted_samples += z_.shape[0]

                    if z_.shape[0] == 0:
                        self.final_mrt = max(self.final_mrt - self.mrt_decay, 0)

                z = Variable(torch.cat(z)).to(self.device)
                z = z[:self.bsize]

                self.n_total_samples += n_total_samples
                self.n_accepted_samples += n_accepted_samples
            else:
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

def main():

    seed_everything()
    mask_gpu()

    parser = argparse.ArgumentParser()
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument("--n_epochs", type=int, default=2500, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.00005, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.9, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--latent_dim", type=int, default=128, help="dimensionality of the latent space")
    parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
    parser.add_argument("--n_channels", type=int, default=3, help="number of image channels")
    parser.add_argument("--n_critic", type=int, default=5, help="number of training steps for discriminator per iter")
    parser.add_argument("--gp", type=float, default=10., help="Loss weight for gradient penalty")
    parser.add_argument("--num_samples", type=int, default=10000, help="number of samples")
    parser.add_argument("--sample_interval", type=int, default=100, help="interval betwen image samples")
    parser.add_argument("--metric_interval", type=int, default=10, help="interval betwen metrics evaluations")
    parser.add_argument("--save_model_interval", type=int, default=100, help="interval betwen model saves")
    parser.add_argument("--print_interval", type=int, default=1, help="interval betwen printing training stats")
    parser.add_argument('--mr', action='store_true')
    parser.add_argument("--mrt", type=float, default=0, help="Minimum memorization rejection threshold, cosine distance have to be greater than mrt")
    parser.add_argument("--mrt_decay", type=float, default=0.02, help="Decay for minimum memorization rejection threshold in case nothing satisfies")
    parser.add_argument("--load_gen_path", default=None, help="path to load the generator state_dict")
    parser.add_argument("--load_dis_path", default=None, help="path to load the discriminator state_dict")
    parser.add_argument("--path_path", default='path.yml', help="path to configuration for training related file paths and dirs")
    opt = parser.parse_args()

    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with open(opt.path_path) as f:
        opt.paths = yaml.load(f, Loader=yaml.FullLoader)
    print(opt)

    # load prep train features
    with np.load(opt.paths['prep_dataset_path']) as f:
        real_features = f['features'][:]
        real_mu = f['mu'][:]
        real_sigma = f['sigma'][:]

    # Initialize generator and discriminator
    generator = Generator(latent_dim=opt.latent_dim, n_channels=opt.n_channels).to(device=opt.device)
    discriminator = Discriminator(latent_dim=opt.latent_dim, n_channels=opt.n_channels).to(device=opt.device)
    gs = GeneratorMRSampler(generator, opt.mr, opt.mrt, opt.mrt_decay, opt.latent_dim, opt.device, 
                            real_features, bsize=opt.batch_size)

    if opt.load_gen_path is not None:
        generator.load_state_dict(torch.load(opt.load_gen_path))
        print(f"Loading {opt.load_gen_path} as generator...")
    if opt.load_dis_path is not None:
        discriminator.load_state_dict(torch.load(opt.load_dis_path))
        print(f"Loading {opt.load_dis_path} as discriminator...")

    if opt.verbose:
        summary(generator, input_size=(opt.latent_dim,))
        summary(discriminator, input_size=(opt.n_channels, opt.img_size, opt.img_size))

    # Configure data loader
    data_dir = opt.paths['data_dir']
    os.makedirs(data_dir, exist_ok=True)
    dataloader = torch.utils.data.DataLoader(
        datasets.CIFAR10(
            data_dir,
            train=True,
            download=True,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize([0.5] * opt.n_channels, [0.5] * opt.n_channels)]
            ),
        ),
        batch_size=opt.batch_size,
        shuffle=True,
        drop_last=True,
    )

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    if not opt.test:
        # prepare save directory
        timestamp = datetime.fromtimestamp(time.time()).strftime('%Y%m%d%H%M')
        log_dir = os.path.join(opt.paths['img_dir'], timestamp)
        os.makedirs(log_dir)
        # prepare writer
        writer = SummaryWriter(logdir=log_dir)

    for epoch in range(opt.n_epochs):

        generator.train()
        gs.reset_running_stats()

        for i, (imgs, _) in enumerate(dataloader):

            optimizer_D.zero_grad()

            real_imgs = Variable(imgs).to(opt.device)
            z = next(gs)[:real_imgs.shape[0]]
            fake_imgs = generator(z)

            real_validity = discriminator(real_imgs)
            fake_validity = discriminator(fake_imgs)
            gradient_penalty = compute_gradient_penalty(discriminator, real_imgs.data, fake_imgs.data, device=opt.device)
            d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + opt.gp * gradient_penalty

            d_loss.backward()
            optimizer_D.step()

            optimizer_G.zero_grad()

            if i % opt.n_critic == 0:

                fake_imgs = generator(z)
                fake_validity = discriminator(fake_imgs)
                g_loss = -torch.mean(fake_validity)

                g_loss.backward()
                optimizer_G.step()

        generator.eval()

        if not opt.test:
            writer.add_scalar('loss/d_loss', d_loss.item(), epoch)
            writer.add_scalar('loss/g_loss', g_loss.item(), epoch)

            if opt.mr:
                writer.add_scalar('stats/mrr', gs.get_mrr(), epoch)
                writer.add_scalar('stats/mrt', gs.get_mrt(), epoch)

        if epoch % opt.print_interval == 0:
            print_str = f"{datetime.now().strftime('%Y/%m/%d %H:%M:%S')} [Epoch {epoch}/{opt.n_epochs}] [D loss: {d_loss.item()}] [G loss: {g_loss.item()}]"
            if opt.mr:
                print_str += f" [mrr: {gs.get_mrr()}] [mrt: {gs.get_mrt()}]"
            print(print_str)

        if epoch % opt.sample_interval == 0 and not opt.test:
            img_fname = os.path.join(log_dir, f"{epoch}.png")
            save_image(fake_imgs.data[:36], img_fname, nrow=6, normalize=True)
            # sampled_images = make_grid(fake_imgs.data[:36], nrow=6, normalize=True)
            # writer.add_image('sampled_images', sampled_images, epoch)

        if epoch % opt.metric_interval == 0:

            with torch.no_grad():
                fake_features_tensor = gs.gen_features(n_instances=10000).detach()

            mds = gs.compute_memorization_distance(fake_features_tensor).cpu().data.numpy()
            mmd = np.mean(mds)
            fake_mu, fake_sigma = get_stats(fake_features_tensor.cpu().data.numpy())
            fid = calculate_frechet_distance(fake_mu, fake_sigma, real_mu, real_sigma) # numpy input

            if not opt.test:
                writer.add_scalar('metric/fid', fid, epoch)
                writer.add_scalar('metric/mmd', mmd, epoch)

            print(f"{datetime.now().strftime('%Y/%m/%d %H:%M:%S')} [fid: {fid}] [mmd: {mmd}]")

        if epoch % opt.save_model_interval == 0 and epoch > 0 and not opt.test:
            torch.save(generator.state_dict(), os.path.join(log_dir, f"generator_epoch{epoch}.pth"))
            torch.save(discriminator.state_dict(), os.path.join(log_dir, f"discriminator_epoch{epoch}.pth"))

    if not opt.test:
        # final save
        torch.save(generator.state_dict(), os.path.join(log_dir, "generator.pth"))
        torch.save(discriminator.state_dict(), os.path.join(log_dir, "discriminator.pth"))

        export_json = os.path.join(log_dir, "all_scalars.json")
        writer.export_scalars_to_json(export_json)
        writer.close()

if __name__ == '__main__':
    main()
