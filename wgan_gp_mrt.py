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
from build_dataset_fid_stats import get_stats
from utils import mask_gpu, seed_everything
from wgan_gp import Generator, Discriminator, GeneratorMRSampler, compute_gradient_penalty

class GeneratorMRTSampler(GeneratorMRSampler):
    def __init__(self, generator, mrt, mrt_decay, latent_dim, device, 
                 real_features, proj_model=None, bsize=64, normalize=True):
        self.mrt = mrt
        self.mrt_decay = mrt_decay

        super(GeneratorMRTSampler, self).__init__(
                generator=generator, latent_dim=latent_dim, device=device, 
                real_features=real_features, proj_model=proj_model, 
                bsize=bsize, normalize=normalize)

        self.final_mrt = mrt
        self.n_total_samples = 0
        self.n_accepted_samples = 0

    def reset_running_stats(self):
        self.n_total_samples = 0
        self.n_accepted_samples = 0

    def get_mrr(self):
        if self.n_total_samples == 0:
            return 0
        else:
            return 1 - self.n_accepted_samples / self.n_total_samples

    def get_mrt(self):
        return self.final_mrt

    def __next__(self):
        if self.mrt == 0:
            z = super(GeneratorMRTSampler, self).__next__()

            self.n_total_samples += self.bsize
            self.n_accepted_samples += self.bsize
        else:
            with torch.no_grad():
                n_total_samples, n_accepted_samples = 0, 0
                z = []
                self.final_mrt = self.mrt # reset mrt back

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
        return z

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
    parser.add_argument("--mrt", type=float, default=0, help="Minimum memorization rejection threshold, cosine distance have to be greater than mrt")
    parser.add_argument("--mrt_decay", type=float, default=0.02, help="Decay for minimum memorization rejection threshold in case nothing satisfies")
    parser.add_argument("--epoch_start", type=int, default=0, help="the index of the beginning of training, usually set when loaded from pretrain model")
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
    gs = GeneratorMRTSampler(generator, opt.mrt, opt.mrt_decay, opt.latent_dim, opt.device, 
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
        drop_last=True, # to avoid sample images not full (if last batch less than 36)
    )

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    if not opt.test:
        # prepare save directory
        timestamp = datetime.fromtimestamp(time.time()).strftime('%Y%m%d%H%M')
        log_dir = os.path.join(opt.paths['img_dir'], timestamp)
        os.makedirs(log_dir)
        # dump options config
        with open(os.path.join(log_dir, 'opt.yml'), 'w') as f:
            yaml.dump(opt, f, default_flow_style=False)
        # prepare writer
        writer = SummaryWriter(logdir=log_dir)

    for epoch in range(opt.epoch_start + 1, opt.n_epochs + 1):

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

            writer.add_scalar('stats/mrr', gs.get_mrr(), epoch)
            writer.add_scalar('stats/mrt', gs.get_mrt(), epoch)

        if epoch % opt.print_interval == 0 or opt.test:
            print_str = f"{datetime.now().strftime('%Y/%m/%d %H:%M:%S')} [Epoch {epoch}/{opt.n_epochs}] [D loss: {d_loss.item()}] [G loss: {g_loss.item()}] [mrr: {gs.get_mrr()}] [mrt: {gs.get_mrt()}]"
            print(print_str)

        if epoch % opt.sample_interval == 0 and not opt.test:
            img_fname = os.path.join(log_dir, f"{epoch}.png")
            save_image(fake_imgs.data[:36], img_fname, nrow=6, normalize=True)

        if epoch % opt.metric_interval == 0 or opt.test:

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

        if epoch % opt.save_model_interval == 0 and epoch > 1 and not opt.test:
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
