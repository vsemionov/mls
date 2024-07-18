#!/usr/bin/env python
# ref: https://arxiv.org/abs/1511.06434 https://github.com/pytorch/examples/tree/main/dcgan https://github.com/aladdinpersson/Machine-Learning-Collection

# usage: <script> [dataset [root [epochs [checkpoint]]]]


import os
import sys
import math
import functools
from datetime import datetime
from pathlib import Path
import multiprocessing

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.utils.tensorboard as tb
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.utils as vutils
from tqdm import tqdm


class Generator(nn.Module):
    def __init__(self, n_channels: int, latent_dim: int = 100, image_size: int = 64, base_filters: int = 128):
        super().__init__()
        n_blocks = int(math.log2(image_size / 4)) - 1
        max_filters = base_filters * 2 ** n_blocks
        block_filters = reversed([base_filters * 2 ** block for block in range(n_blocks)])
        self.layers = nn.Sequential(
            self._block(latent_dim, max_filters, 1, 0, activation=False),
            *[self._block(filters * 2, filters) for filters in block_filters],
            self._block(base_filters, n_channels, normalize=False, activation=False),
            nn.Tanh()
        )

    @staticmethod
    def _block(in_filters, out_filters, stride=2, padding=1, normalize=True, activation=True):
        return nn.Sequential(
            nn.ConvTranspose2d(in_filters, out_filters, 4, stride=stride, padding=padding, bias=(not normalize)),
            *([nn.BatchNorm2d(out_filters)] if normalize else []),
            *([nn.ReLU(inplace=True)] if activation else [])
        )

    def forward(self, x):
        return self.layers(x.view(*x.size(), 1, 1))

    @property
    def n_channels(self):
        return self.layers[-2][0].out_channels

    @property
    def latent_dim(self):
        return self.layers[0][0].in_channels

    @property
    def image_size(self):
        return 4 * 2 ** (len(self.layers) - 2)

    @property
    def base_filters(self):
        return self.layers[-2][0].in_channels


class Discriminator(nn.Module):
    def __init__(self, n_channels: int, image_size: int = 64, base_filters: int = 128):
        super().__init__()
        n_blocks = int(math.log2(image_size / 4)) - 1
        max_filters = base_filters * 2 ** n_blocks
        block_filters = [base_filters * 2 ** block for block in range(n_blocks)]
        self.layers = nn.Sequential(
            self._block(n_channels, base_filters),
            *[self._block(filters, filters * 2) for filters in block_filters],
            self._block(max_filters, 1, 1, 0, normalize=False, activation=False)
        )

    @staticmethod
    def _block(in_filters, out_filters, stride=2, padding=1, normalize=True, activation=True):
        return nn.Sequential(
            nn.Conv2d(in_filters, out_filters, 4, stride=stride, padding=padding, bias=(not normalize)),
            *([nn.BatchNorm2d(out_filters)] if normalize else []),
            *([nn.LeakyReLU(0.2, inplace=True)] if activation else [])
        )

    def forward(self, x):
        return self.layers(x).view(-1)

    @property
    def n_channels(self):
        return self.layers[0][0].in_channels

    @property
    def image_size(self):
        return 4 * 2 ** (len(self.layers) - 1)

    @property
    def base_filters(self):
        return self.layers[0][0].out_channels


def init_weights(m):
    if isinstance(m, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(m.weight, 0.0, 0.02)
    if isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
        torch.nn.init.normal_(m.weight, 1.0, 0.02)
        torch.nn.init.zeros_(m.bias)


@torch.no_grad()
def test(generator, discriminator, n_channels, device, verbose=0):
    generator.eval()
    discriminator.eval()
    assert generator.n_channels == discriminator.n_channels == n_channels
    assert generator.image_size == discriminator.image_size
    batch_size = 2
    image_size = generator.image_size
    z = torch.randn(batch_size, generator.latent_dim, device=device)
    x = generator(z)
    assert x.size() == (batch_size, n_channels, image_size, image_size)
    y = discriminator(x)
    assert y.size() == (batch_size,)
    if verbose >= 1:
        if verbose >= 2:
            print(generator)
            print(discriminator)
        print('Image size:', f'{image_size}x{image_size}')
        print('Channels:', n_channels)
        print('Latent size:', generator.latent_dim)
        print('Generator filters:', generator.base_filters)
        print('Discriminator filters:', generator.base_filters)
        print('Generator parameters:', sum(p.numel() for p in generator.parameters()))
        print('Discriminator parameters:', sum(p.numel() for p in discriminator.parameters()))


multiprocessing.set_start_method('fork')

dataset_config = {
    'mnist': (functools.partial(datasets.MNIST, train=True), 1),
    'celeba': (functools.partial(datasets.CelebA, split='train'), 3)
}
dataset_name = sys.argv[1] if len(sys.argv) > 1 else 'mnist'
dataset_class, n_channels = dataset_config[dataset_name]

generator = Generator(n_channels)
discriminator = Discriminator(n_channels)

has_cuda = torch.cuda.is_available()
device = torch.device('cuda' if has_cuda else 'cpu')
generator.to(device)
discriminator.to(device)

root_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(__file__).parent / '..' / '..'

data_dir = str(root_dir / 'data')
transform = transforms.Compose([
    transforms.Resize(generator.image_size),
    transforms.CenterCrop(generator.image_size),
    transforms.ToTensor(),
    transforms.Normalize([0.5] * n_channels, [0.5] * n_channels)
])
dataset = dataset_class(data_dir, download=True, transform=transform)

dataloader = data.DataLoader(dataset, batch_size=128, shuffle=True, drop_last=True,
    num_workers=os.cpu_count(), persistent_workers=True, pin_memory=True)

criterion = nn.BCEWithLogitsLoss()

gen_optimizer = optim.Adam(generator.parameters(), lr=2e-4, betas=(0.5, 0.999))
disc_optimizer = optim.Adam(discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))

scaler = torch.cuda.amp.GradScaler(enabled=has_cuda)

epochs = int(sys.argv[3]) if len(sys.argv) > 3 else 25

if len(sys.argv) > 4:
    checkpoint = torch.load(sys.argv[4])
    start_epoch = checkpoint['epoch'] + 1
    generator.load_state_dict(checkpoint['generator'])
    discriminator.load_state_dict(checkpoint['discriminator'])
    gen_optimizer.load_state_dict(checkpoint['gen_optimizer'])
    disc_optimizer.load_state_dict(checkpoint['disc_optimizer'])
    fixed_noise = checkpoint['noise']
else:
    start_epoch = 0
    generator.apply(init_weights)
    discriminator.apply(init_weights)
    fixed_noise = torch.randn(64, generator.latent_dim, device=device)

test(generator, discriminator, n_channels, device, verbose=1)

timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')

log_dir = str(root_dir / 'logs' / Path(__file__).stem / dataset_name / timestamp)
writer = tb.SummaryWriter(log_dir=log_dir)

generator.train()
discriminator.train()

labels_real = torch.tensor(1.0, device=device)
labels_fake = torch.tensor(0.0, device=device)

step = start_epoch * len(dataloader)
gen_losses = []
disc_losses = []
mean_gen_loss = 0
mean_disc_loss = 0
get_postfix = lambda: dict(gen_loss=f'{mean_gen_loss:.2e}', disc_loss=f'{mean_disc_loss:.2e}')
ckpt_dir = root_dir / 'ckpt' / Path(__file__).stem / dataset_name / timestamp
ckpt_dir.mkdir(parents=True, exist_ok=True)
for epoch in range(start_epoch, epochs):
    loop = tqdm(dataloader, desc=f'Epoch {epoch + 1}/{epochs}', postfix=get_postfix(), file=sys.stdout)
    for batch_idx, (real, _) in enumerate(loop):
        real = real.to(device)
        batch_len = real.size(0)

        targets_real = labels_real.expand(batch_len)
        targets_fake = labels_fake.expand(batch_len)

        z = torch.randn(batch_len, generator.latent_dim, device=device)

        discriminator.requires_grad_(True)
        with torch.cuda.amp.autocast(enabled=has_cuda):
            fake = generator(z)
            preds_real = discriminator(real)
            preds_fake = discriminator(fake.detach())
            disc_loss_real = criterion(preds_real, targets_real)
            disc_loss_fake = criterion(preds_fake, targets_fake)
            disc_loss = (disc_loss_real + disc_loss_fake) * 0.5
        disc_optimizer.zero_grad()
        scaler.scale(disc_loss).backward()
        scaler.step(disc_optimizer)
        scaler.update()

        discriminator.requires_grad_(False)
        with torch.cuda.amp.autocast(enabled=has_cuda):
            preds_fake = discriminator(fake)
            gen_loss = criterion(preds_fake, targets_real)
        gen_optimizer.zero_grad()
        scaler.scale(gen_loss).backward()
        scaler.step(gen_optimizer)
        scaler.update()

        gen_losses.append(gen_loss.item())
        disc_losses.append(disc_loss.item())

        loop.set_postfix(**get_postfix())

        if (step % 100 == 0 or epoch == epochs - 1 and batch_idx == len(dataloader) - 1) and gen_losses:
            mean_gen_loss = sum(gen_losses) / len(gen_losses)
            mean_disc_loss = sum(disc_losses) / len(disc_losses)

            generator.eval()
            with torch.no_grad():
                images = generator(fixed_noise)
            generator.train()

            grid_fake = vutils.make_grid(images, normalize=True, value_range=(-1, 1))
            grid_real = vutils.make_grid(real[:len(images)], normalize=True, value_range=(-1, 1))

            writer.add_scalar('Loss/generator', mean_gen_loss, step)
            writer.add_scalar('Loss/discriminator', mean_disc_loss, step)
            writer.add_image('Generated images', grid_fake, step)
            writer.add_image('Real images', grid_real, step)

            gen_losses = []
            disc_losses = []

        step += 1

    if (epoch + 1) % 5 == 0 or (epoch + 1) == epochs:
        checkpoint = {
            'epoch': epoch,
            'generator': generator.state_dict(),
            'discriminator': discriminator.state_dict(),
            'gen_optimizer': gen_optimizer.state_dict(),
            'disc_optimizer': disc_optimizer.state_dict(),
            'noise': fixed_noise
        }
        ckpt_path = ckpt_dir / f'epoch-{epoch + 1}.pt'
        torch.save(checkpoint, ckpt_path)
