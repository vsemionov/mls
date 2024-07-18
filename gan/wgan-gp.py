#!/usr/bin/env python
# ref: https://arxiv.org/abs/1701.07875 https://arxiv.org/abs/1704.00028 https://github.com/eriklindernoren/PyTorch-GAN https://github.com/aladdinpersson/Machine-Learning-Collection

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


class Reshape(nn.Module):
    def __init__(self, *shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(-1, *self.shape)


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


class Critic(nn.Module):
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
            *([nn.InstanceNorm2d(out_filters, affine=True)] if normalize else []),
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


def gradient_penalty(critic, real, fake, device):
    alpha = torch.rand(real.size(0), 1, 1, 1, device=device)
    interpolated = real * alpha + fake * (1 - alpha)
    interpolated.requires_grad_(True)

    scores = critic(interpolated)

    gradient = torch.autograd.grad(
        inputs=interpolated,
        outputs=scores,
        grad_outputs=torch.tensor(1.0, device=device).expand_as(scores),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    gradient = gradient.view(gradient.size(0), -1)
    norm = gradient.norm(2, dim=1)
    penalty = torch.mean((norm - 1) ** 2)
    return penalty


@torch.no_grad()
def test(generator, critic, n_channels, device, verbose=0):
    generator.eval()
    critic.eval()
    assert generator.n_channels == critic.n_channels == n_channels
    assert generator.image_size == critic.image_size
    batch_size = 2
    image_size = generator.image_size
    z = torch.randn(batch_size, generator.latent_dim, device=device)
    x = generator(z)
    assert x.size() == (batch_size, n_channels, image_size, image_size)
    y = critic(x)
    assert y.size() == (batch_size,)
    if verbose >= 1:
        if verbose >= 2:
            print(generator)
            print(critic)
        print('Image size:', f'{image_size}x{image_size}')
        print('Channels:', n_channels)
        print('Latent size:', generator.latent_dim)
        print('Generator filters:', generator.base_filters)
        print('Critic filters:', generator.base_filters)
        print('Generator parameters:', sum(p.numel() for p in generator.parameters()))
        print('Critic parameters:', sum(p.numel() for p in critic.parameters()))


multiprocessing.set_start_method('fork')

dataset_config = {
    'mnist': (functools.partial(datasets.MNIST, train=True), 1),
    'celeba': (functools.partial(datasets.CelebA, split='train'), 3)
}
dataset_name = sys.argv[1] if len(sys.argv) > 1 else 'mnist'
dataset_class, n_channels = dataset_config[dataset_name]

generator = Generator(n_channels)
critic = Critic(n_channels)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
generator.to(device)
critic.to(device)

root_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(__file__).parent / '..' / '..'

data_dir = str(root_dir / 'data')
transform = transforms.Compose([
    transforms.Resize(generator.image_size),
    transforms.CenterCrop(generator.image_size),
    transforms.ToTensor(),
    transforms.Normalize([0.5] * n_channels, [0.5] * n_channels)
])
dataset = dataset_class(data_dir, download=True, transform=transform)

dataloader = data.DataLoader(dataset, batch_size=64, shuffle=True, drop_last=True,
    num_workers=os.cpu_count(), persistent_workers=True, pin_memory=True)

gen_optimizer = optim.Adam(generator.parameters(), lr=1e-4, betas=(0.0, 0.9))
crit_optimizer = optim.Adam(critic.parameters(), lr=1e-4, betas=(0.0, 0.9))

epochs = int(sys.argv[3]) if len(sys.argv) > 3 else 25

if len(sys.argv) > 4:
    checkpoint = torch.load(sys.argv[4])
    start_epoch = checkpoint['epoch'] + 1
    generator.load_state_dict(checkpoint['generator'])
    critic.load_state_dict(checkpoint['critic'])
    gen_optimizer.load_state_dict(checkpoint['gen_optimizer'])
    crit_optimizer.load_state_dict(checkpoint['crit_optimizer'])
    fixed_noise = checkpoint['noise']
else:
    start_epoch = 0
    generator.apply(init_weights)
    critic.apply(init_weights)
    fixed_noise = torch.randn(64, generator.latent_dim, device=device)

test(generator, critic, n_channels, device, verbose=1)

timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')

log_dir = str(root_dir / 'logs' / Path(__file__).stem / dataset_name / timestamp)
writer = tb.SummaryWriter(log_dir=log_dir)

generator.train()
critic.train()

step = start_epoch * len(dataloader)
gen_losses = []
crit_losses = []
real_scores = []
fake_scores = []
gps = []
mean_gen_loss = 0
mean_crit_loss = 0
mean_real_score = 0
mean_fake_score = 0
mean_gp = 0
get_postfix = lambda: dict(gen_loss=f'{mean_gen_loss:.2e}', crit_loss=f'{mean_crit_loss:.2e}')
ckpt_dir = root_dir / 'ckpt' / Path(__file__).stem / dataset_name / timestamp
ckpt_dir.mkdir(parents=True, exist_ok=True)
for epoch in range(start_epoch, epochs):
    loop = tqdm(dataloader, desc=f'Epoch {epoch + 1}/{epochs}', postfix=get_postfix(), file=sys.stdout)
    for batch_idx, (real, _) in enumerate(loop):
        real = real.to(device)
        batch_len = real.size(0)

        z = torch.randn(batch_len, generator.latent_dim, device=device)
        fake = generator(z)

        preds_real = critic(real)
        preds_fake = critic(fake.detach())
        real_score = preds_real.mean()
        fake_score = preds_fake.mean()
        gp = gradient_penalty(critic, real, fake.detach(), device)
        crit_loss = fake_score - real_score + gp * 10
        crit_optimizer.zero_grad()
        crit_loss.backward()
        crit_optimizer.step()

        crit_losses.append(crit_loss.item())
        real_scores.append(real_score.item())
        fake_scores.append(fake_score.item())
        gps.append(gp.item())

        if (step + 1) % 5 == 0:
            preds_fake = critic(fake)
            gen_loss = -preds_fake.mean()
            gen_optimizer.zero_grad()
            gen_loss.backward()
            gen_optimizer.step()

            gen_losses.append(gen_loss.item())

        loop.set_postfix(**get_postfix())

        if (step % 100 == 0 or epoch == epochs - 1 and batch_idx == len(dataloader) - 1) and gen_losses:
            mean_gen_loss = sum(gen_losses) / len(gen_losses)
            mean_crit_loss = sum(crit_losses) / len(crit_losses)
            mean_real_score = sum(real_scores) / len(real_scores)
            mean_fake_score = sum(fake_scores) / len(fake_scores)
            mean_gp = sum(gps) / len(gps)

            with torch.no_grad():
                images = generator(fixed_noise)

            grid_fake = vutils.make_grid(images, normalize=True, value_range=(-1, 1))
            grid_real = vutils.make_grid(real[:len(images)], normalize=True, value_range=(-1, 1))

            writer.add_scalar('Loss/generator', mean_gen_loss, step)
            writer.add_scalar('Loss/critic', mean_crit_loss, step)
            writer.add_scalar('Score/gp (unscaled)', mean_gp, step)
            writer.add_scalars('Score/value', {'real': mean_real_score, 'fake': mean_fake_score}, step)
            writer.add_image('Generated images', grid_fake, step)
            writer.add_image('Real images', grid_real, step)

            gen_losses = []
            crit_losses = []
            real_scores = []
            fake_scores = []
            gps = []

        step += 1

    if (epoch + 1) % 5 == 0 or (epoch + 1) == epochs:
        checkpoint = {
            'epoch': epoch,
            'generator': generator.state_dict(),
            'critic': critic.state_dict(),
            'gen_optimizer': gen_optimizer.state_dict(),
            'crit_optimizer': crit_optimizer.state_dict(),
            'noise': fixed_noise
        }
        ckpt_path = ckpt_dir / f'epoch-{epoch + 1}.pt'
        torch.save(checkpoint, ckpt_path)
