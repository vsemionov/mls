#!/usr/bin/env python
# ref: https://arxiv.org/abs/1611.07004 https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix https://github.com/aladdinpersson/Machine-Learning-Collection

# dataset: http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/$FILE.tar.gz
# where $FILE is one of cityscapes, night2day, edges2handbags, edges2shoes, facades, maps

# usage: <script> [dataset [root [epochs [checkpoint]]]]


import os
import sys
import math
import multiprocessing
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.utils.tensorboard as tb
import torchvision.utils as vutils
import albumentations as A
import albumentations.pytorch as ap
import numpy as np
from PIL import Image
from tqdm import tqdm


DEFAULT_DATASET = 'maps'
DEFAULT_ROOT_DIR = Path(__file__).parent / '..' / '..'
DEFAULT_EPOCHS = 200

NORM_AFFINE = False

LAMBDA_L1 = 100

BATCH_SIZE = 1
VAL_BATCH_SIZE = 64
LEARNING_RATE = 2e-4
BETAS = (0.5, 0.999)

ENABLE_AMP = True
LOG_EVERY = 16
SAVE_EVERY = 5
SAVE_IMAGES_EVERY = 1
SAVE_IMAGES = 16


class Dataset(data.Dataset):
    def __init__(self, root_dir, split, in_channels, out_channels, augment=False):
        self.augment = augment
        self.paths = sorted((Path(root_dir) / split).iterdir())
        transforms = self.transforms
        if augment:
            transforms += self.augmentations
        self.transform = A.Compose(transforms, additional_targets={'target': 'image'})
        self.input_transform = None
        norm_tensor = lambda n_channels: A.Compose([
            A.Normalize([0.5] * n_channels, [0.5] * n_channels),
            ap.ToTensorV2()
        ])
        self.transform_x = norm_tensor(in_channels)
        self.transform_y = norm_tensor(out_channels)

    @property
    def transforms(self):
        return [A.Resize(256, 256)]

    @property
    def augmentations(self):
        return []

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path)
        a = np.array(img)
        w = a.shape[1] // 2
        x, y = a[:, :w], a[:, w:]
        xy = self.transform(image=x, target=y)
        x, y = xy['image'], xy['target']
        if self.input_transform:
            x = self.input_transform(image=x)['image']
        x, y = self.transform_x(image=x)['image'], self.transform_y(image=y)['image']
        return x, y


class MapsDataset(Dataset):
    def __init__(self, root_dir, split, augment=False):
        super().__init__(Path(root_dir) / 'maps', split, 3, 3, augment=augment)
        if augment:
            self.input_transform = A.ColorJitter(brightness=(0.9, 1.1), contrast=(0.9, 1.1), saturation=(0.9, 1.1),
                hue=(-0.5, 0.5), always_apply=True)

    @property
    def transforms(self):
        fit = A.RandomCrop(256, 256) if self.augment else A.CenterCrop(256, 256)
        return [fit]

    @property
    def augmentations(self):
        return [A.Flip()]


class Generator(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.encoder = nn.ModuleList([
            self._down(in_channels, 64, normalize=False),
            self._down(64, 128),
            self._down(128, 256),
            self._down(256, 512),
            self._down(512, 512),
            self._down(512, 512),
            self._down(512, 512),
            self._down(512, 512, normalize=False)
        ])
        self.decoder = nn.ModuleList([
            self._up(512, 512),
            self._up(1024, 512, dropout=True),
            self._up(1024, 512, dropout=True),
            self._up(1024, 512, dropout=True),
            self._up(1024, 256),
            self._up(512, 128),
            self._up(256, 64)
        ])
        self.head = nn.Sequential(
            nn.ConvTranspose2d(128, out_channels, 4, 2, 1),
            nn.Tanh()
        )

    @staticmethod
    def _down(in_channels, out_channels, normalize=True):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, stride=2, padding=1, bias=(not normalize)),
            *([nn.InstanceNorm2d(out_channels, affine=NORM_AFFINE)] if normalize else []),
            nn.LeakyReLU(0.2, inplace=True)
        )

    @staticmethod
    def _up(in_channels, out_channels, dropout=False):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels, affine=NORM_AFFINE),
            nn.ReLU(inplace=True),
            *([nn.Dropout(0.5)] if dropout else [])
        )

    def forward(self, x):
        skips = []
        x = self.encoder[0](x)
        for block in self.encoder[1:]:
            skips.append(x)
            x = block(x)
        for block in self.decoder:
            x = block(x)
            skip = skips.pop()
            x = torch.cat((x, skip), dim=1)
        assert not skips
        return self.head(x)


class Discriminator(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.layers = nn.Sequential(
            self._block(in_channels + out_channels, 64, normalize=False),
            self._block(64, 128),
            self._block(128, 256),
            self._block(256, 512, stride=1),
            nn.Conv2d(512, 1, 4, 1, 1)
        )

    @staticmethod
    def _block(in_channels, out_channels, stride=2, normalize=True):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, stride=stride, padding=1, bias=(not normalize)),
            *([nn.InstanceNorm2d(out_channels, affine=NORM_AFFINE)] if normalize else []),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x, y):
        xy = torch.cat([x, y], dim=1)
        return self.layers(xy).squeeze(1)


def init_weights(m):
    if isinstance(m, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(m.weight, 0.0, 0.02)
    if isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
        if m.weight is not None:
            torch.nn.init.normal_(m.weight, 1.0, 0.02)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)


@torch.no_grad()
def test(generator, discriminator, in_channels, out_channels, device, verbose=0):
    generator.eval()
    discriminator.eval()
    batch_size = 2
    x = torch.randn(batch_size, in_channels, 256, 256, device=device)
    y = generator(x)
    assert y.size() == (batch_size, out_channels, 256, 256)
    y = discriminator(x, y)
    assert y.size() == (batch_size, 30, 30)
    if verbose >= 1:
        if verbose >= 2:
            print(generator)
            print(discriminator)
        print('Image size:', '256x256')
        print('In channels:', in_channels)
        print('Out channels:', out_channels)
        print('Generator parameters:', sum(p.numel() for p in generator.parameters()))
        print('Discriminator parameters:', sum(p.numel() for p in discriminator.parameters()))


@torch.no_grad()
def evaluate(generator, train_dataloader, val_dataloader, l1_criterion, epoch, epochs, train_dataset_static,
        log_train_indices, log_val_indices, writer, device, enable_amp, initial):
    eval_losses = []
    loop = tqdm(val_dataloader, desc=f'Validating', file=sys.stdout)
    for batch_idx, (x, real) in enumerate(loop):
        x = x.to(device)
        real = real.to(device)
        with torch.cuda.amp.autocast(enabled=enable_amp):
            fake = generator(x)
            gen_loss = l1_criterion(fake, real)
        eval_losses.append(gen_loss.item())
    eval_loss = sum(eval_losses) / len(eval_losses)
    print('L1 loss:', f'{eval_loss:.2e}')
    step = (epoch + 1) * len(train_dataloader)
    writer.add_scalar('Epoch', epoch + 1, step)
    writer.add_scalar('Loss/validation', eval_loss, step)
    if (epoch + 1) % SAVE_IMAGES_EVERY == 0 or epoch == epochs - 1:
        for dataset, indices, name in [(train_dataset_static, log_train_indices, 'Train'),
                                       (val_dataloader.dataset, log_val_indices, 'Validation')]:
            items = [dataset[i] for i in indices]
            x = torch.stack([item[0].to(device) for item in items])
            fake = generator(x)
            real = torch.stack([item[1].to(device) for item in items])
            grid_kwargs = dict(nrow=math.ceil(SAVE_IMAGES / math.sqrt(SAVE_IMAGES)), normalize=True,
                value_range=(-1, 1))
            grid_fake = vutils.make_grid(fake, **grid_kwargs)
            writer.add_image(f'{name} images/generated', grid_fake, step)
            if initial:
                grid_real = vutils.make_grid(real, **grid_kwargs)
                grid_input = vutils.make_grid(x, **grid_kwargs)
                writer.add_image(f'{name} images/target', grid_real, step)
                writer.add_image(f'{name} images/input', grid_input, step)
        initial = False
    return initial


def main():
    dataset_config = {
        'maps': (MapsDataset, 3, 3)
    }
    dataset_name = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DATASET
    dataset_class, in_channels, out_channels = dataset_config[dataset_name]

    generator = Generator(in_channels, out_channels)
    discriminator = Discriminator(in_channels, out_channels)

    has_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if has_cuda else 'cpu')
    generator.to(device)
    discriminator.to(device)

    root_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_ROOT_DIR

    data_dir = str(root_dir / 'data')
    train_dataset = dataset_class(data_dir, 'train', augment=True)
    train_dataset_static = dataset_class(data_dir, 'train', augment=False)
    val_dataset = dataset_class(data_dir, 'val', augment=False)

    train_dataloader = data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True,
        num_workers=os.cpu_count(), persistent_workers=True, pin_memory=True)
    val_dataloader = data.DataLoader(val_dataset, batch_size=VAL_BATCH_SIZE, shuffle=False, drop_last=False,
        num_workers=os.cpu_count(), persistent_workers=True, pin_memory=True)

    gan_criterion = nn.BCEWithLogitsLoss()
    l1_criterion = nn.L1Loss()

    gen_optimizer = optim.Adam(generator.parameters(), lr=LEARNING_RATE, betas=BETAS)
    disc_optimizer = optim.Adam(discriminator.parameters(), lr=LEARNING_RATE, betas=BETAS)

    enable_amp = ENABLE_AMP and has_cuda
    scaler = torch.cuda.amp.GradScaler(enabled=enable_amp)

    epochs = int(sys.argv[3]) if len(sys.argv) > 3 else DEFAULT_EPOCHS

    if len(sys.argv) > 4:
        checkpoint = torch.load(sys.argv[4])
        start_epoch = checkpoint['epoch'] + 1
        generator.load_state_dict(checkpoint['generator'])
        discriminator.load_state_dict(checkpoint['discriminator'])
        gen_optimizer.load_state_dict(checkpoint['gen_optimizer'])
        disc_optimizer.load_state_dict(checkpoint['disc_optimizer'])
        log_train_indices = checkpoint['log_train_indices']
        log_val_indices = checkpoint['log_val_indices']
    else:
        start_epoch = 0
        generator.apply(init_weights)
        discriminator.apply(init_weights)
        log_train_indices = torch.randint(0, len(train_dataset_static), (SAVE_IMAGES,))
        log_val_indices = torch.randint(0, len(val_dataset), (SAVE_IMAGES,))

    test(generator, discriminator, in_channels, out_channels, device, verbose=1)

    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')

    log_dir = str(root_dir / 'logs' / Path(__file__).stem / dataset_name / timestamp)
    writer = tb.SummaryWriter(log_dir=log_dir)

    generator.train()
    discriminator.train()

    label_real = torch.tensor(1.0, device=device)
    label_fake = torch.tensor(0.0, device=device)

    step = start_epoch * len(train_dataloader)
    gen_losses = []
    disc_losses = []
    mean_gen_loss = 0
    mean_disc_loss = 0
    get_postfix = lambda: dict(gen_loss=f'{mean_gen_loss:.2e}', disc_loss=f'{mean_disc_loss:.2e}')
    ckpt_dir = root_dir / 'ckpt' / Path(__file__).stem / dataset_name / timestamp
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    initial_eval = True
    for epoch in range(start_epoch, epochs):
        loop = tqdm(train_dataloader, desc=f'Epoch {epoch + 1}/{epochs}', postfix=get_postfix(), file=sys.stdout)
        for batch_idx, (x, real) in enumerate(loop):
            x = x.to(device)
            real = real.to(device)
            batch_len = x.size(0)

            targets_real = label_real.expand(batch_len, 30, 30)
            targets_fake = label_fake.expand(batch_len, 30, 30)

            discriminator.requires_grad_(True)
            with torch.cuda.amp.autocast(enabled=enable_amp):
                fake = generator(x)
                preds_real = discriminator(x, real)
                preds_fake = discriminator(x, fake.detach())
                disc_loss_real = gan_criterion(preds_real, targets_real)
                disc_loss_fake = gan_criterion(preds_fake, targets_fake)
                disc_loss = (disc_loss_real + disc_loss_fake) * 0.5
            disc_optimizer.zero_grad()
            scaler.scale(disc_loss).backward()
            scaler.step(disc_optimizer)
            scaler.update()

            discriminator.requires_grad_(False)
            with torch.cuda.amp.autocast(enabled=enable_amp):
                preds_fake = discriminator(x, fake)
                gen_loss_gan = gan_criterion(preds_fake, targets_real)
                gen_loss_l1 = l1_criterion(fake, real)
                gen_loss = gen_loss_gan + gen_loss_l1 * LAMBDA_L1
            gen_optimizer.zero_grad()
            scaler.scale(gen_loss).backward()
            scaler.step(gen_optimizer)
            scaler.update()

            gen_losses.append(gen_loss.item())
            disc_losses.append(disc_loss.item())

            loop.set_postfix(**get_postfix())

            if (step % LOG_EVERY == 0 or epoch == epochs - 1 and batch_idx == len(train_dataloader) - 1) and gen_losses:
                mean_gen_loss = sum(gen_losses) / len(gen_losses)
                mean_disc_loss = sum(disc_losses) / len(disc_losses)

                writer.add_scalar('Loss/generator', mean_gen_loss, step)
                writer.add_scalar('Loss/discriminator', mean_disc_loss, step)

                gen_losses = []
                disc_losses = []

            step += 1

        initial_eval = evaluate(generator, train_dataloader, val_dataloader, l1_criterion, epoch, epochs,
            train_dataset_static, log_train_indices, log_val_indices, writer, device, enable_amp, initial_eval)

        if (epoch + 1) % SAVE_EVERY == 0 or (epoch + 1) == epochs:
            checkpoint = {
                'epoch': epoch,
                'generator': generator.state_dict(),
                'discriminator': discriminator.state_dict(),
                'gen_optimizer': gen_optimizer.state_dict(),
                'disc_optimizer': disc_optimizer.state_dict(),
                'log_train_indices': log_train_indices,
                'log_val_indices': log_val_indices
            }
            ckpt_path = ckpt_dir / f'epoch-{epoch + 1}.pt'
            torch.save(checkpoint, ckpt_path)


if __name__ == '__main__':
    if sys.platform == 'darwin':
        multiprocessing.set_start_method('fork')
    main()
