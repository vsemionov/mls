#!/usr/bin/env python
# ref: https://arxiv.org/abs/1406.2661 https://github.com/aladdinpersson/Machine-Learning-Collection

import os
import sys
from datetime import datetime
from pathlib import Path
import multiprocessing

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.utils.tensorboard as tb
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from tqdm import tqdm


class Generator(nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: list[int], image_size: (int,) * 3):
        super().__init__()
        self.latent_dim = latent_dim
        self.image_size = image_size
        self.layers = nn.ModuleList()
        output_dim = image_size[0] * image_size[1] * image_size[2]
        layer_in_dims = [latent_dim] + hidden_dim
        layer_out_dims = hidden_dim + [output_dim]
        layer_dims = list(zip(layer_in_dims, layer_out_dims))
        for in_dim, out_dim in layer_dims[:-1]:
            self.layers.append(nn.Linear(in_dim, out_dim))
            self.layers.append(nn.LeakyReLU(0.1))
        self.layers.append(nn.Linear(*layer_dims[-1]))
        self.layers.append(nn.Tanh())

    def forward(self, x):
        B = x.size(0)
        for layer in self.layers:
            x = layer(x)
        x = x.view(B, *self.image_size)
        return x


class Discriminator(nn.Module):
    def __init__(self, image_size: (int,) * 3, hidden_dim: list[int]):
        super().__init__()
        self.image_size = image_size
        self.layers = nn.ModuleList()
        input_dim = image_size[0] * image_size[1] * image_size[2]
        layer_in_dims = [input_dim] + hidden_dim
        layer_out_dims = hidden_dim + [1]
        layer_dims = list(zip(layer_in_dims, layer_out_dims))
        for in_dim, out_dim in layer_dims[:-1]:
            self.layers.append(nn.Linear(in_dim, out_dim))
            self.layers.append(nn.LeakyReLU(0.1))
        self.layers.append(nn.Linear(*layer_dims[-1]))
        self.layers.append(nn.Sigmoid())

    def forward(self, x):
        B = x.size(0)
        x = x.view(B, -1)
        for layer in self.layers:
            x = layer(x)
        return x


multiprocessing.set_start_method('fork')

root_dir = Path(__file__).parent / '..' / '..'
data_dir = str(root_dir / 'data')
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=(0.5,), std=(0.5,))])
dataset = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
dataloader = data.DataLoader(dataset, batch_size=32, shuffle=True,
    num_workers=os.cpu_count(), persistent_workers=True, pin_memory=True)

log_dir = str(root_dir / 'logs' / Path(__file__).stem / datetime.now().strftime('%Y%m%d-%H%M%S'))
writer = tb.SummaryWriter(log_dir=log_dir)

generator = Generator(latent_dim=64, hidden_dim=[256], image_size=(1, 28, 28))
discriminator = Discriminator(image_size=(1, 28, 28), hidden_dim=[128])

gen_criterion = nn.BCELoss()
disc_criterion = nn.BCELoss()
gen_optimizer = optim.Adam(generator.parameters(), lr=3e-4)
disc_optimizer = optim.Adam(discriminator.parameters(), lr=3e-4)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
generator.to(device)
discriminator.to(device)

targets_real = torch.ones(dataloader.batch_size, 1, device=device)
targets_fake = torch.zeros(dataloader.batch_size, 1, device=device)
test_noise = torch.randn(64, generator.latent_dim, device=device)

epochs = 50
for epoch in range(epochs):
    gen_losses = []
    disc_losses = []

    generator.train().requires_grad_(True)

    for real, _ in tqdm(dataloader, file=sys.stdout):
        real = real.to(device)

        z = torch.randn(real.size(0), generator.latent_dim, device=device)
        fake = generator(z)

        discriminator.train().requires_grad_(True)
        preds_real = discriminator(real)
        preds_fake = discriminator(fake.detach())
        disc_loss_real = disc_criterion(preds_real, targets_real)
        disc_loss_fake = disc_criterion(preds_fake, targets_fake)
        disc_loss = (disc_loss_real + disc_loss_fake) * 0.5
        disc_optimizer.zero_grad()
        disc_loss.backward()
        disc_optimizer.step()

        discriminator.eval().requires_grad_(False)
        preds_fake = discriminator(fake)
        gen_loss = gen_criterion(preds_fake, targets_real)
        gen_optimizer.zero_grad()
        gen_loss.backward()
        gen_optimizer.step()

        gen_losses.append(gen_loss.item())
        disc_losses.append(disc_loss.item())

    disc_loss = sum(disc_losses) / len(disc_losses)
    gen_loss = sum(gen_losses) / len(gen_losses)

    generator.eval().requires_grad_(False)
    images = generator(test_noise)
    grid = torchvision.utils.make_grid(images, normalize=True, value_range=(-1, 1))

    writer.add_scalar('Loss/generator', gen_loss, epoch + 1)
    writer.add_scalar('Loss/discriminator', disc_loss, epoch + 1)
    writer.add_image('Generated images', grid, epoch + 1)

    print(f'Epoch: {epoch + 1}/{epochs}, disc_loss: {disc_loss:.2e}, gen_loss: {gen_loss:.2e}')
