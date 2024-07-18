#!/usr/bin/env python
# ref: https://arxiv.org/abs/1703.10593 https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix https://github.com/aladdinpersson/Machine-Learning-Collection

# dataset: http://efrosgans.eecs.berkeley.edu/cyclegan/datasets/$FILE.zip
# where $FILE is one of apple2orange, summer2winter_yosemite, horse2zebra, monet2photo, cezanne2photo, ukiyoe2photo,
#   vangogh2photo, maps, cityscapes, facades, iphone2dslr_flower, ae_photos

# usage: <script> [dataset [root [epochs [checkpoint]]]]


import os
import sys
import math
import random
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
from tqdm import tqdm, trange


DEFAULT_DATASET = 'horse2zebra'
DEFAULT_ROOT_DIR = Path(__file__).parent / '..' / '..'
DEFAULT_EPOCHS = 200

NORM_AFFINE = False

LAMBDA_CYCLE = 10
LAMBDA_ID = 0

BATCH_SIZE = 1
VAL_BATCH_SIZE = 16
LEARNING_RATE = 2e-4
BETAS = (0.5, 0.999)

ENABLE_AMP = True
LOG_EVERY = 16
SAVE_EVERY = 5
SAVE_IMAGES_EVERY = 1
SAVE_IMAGES = 16


class Dataset(data.Dataset):
    def __init__(self, root_dir, n_channels, mode=None, augment=False):
        self.mode = mode
        self.augment = augment
        self.paths = sorted(Path(root_dir).iterdir())
        transforms = self.transforms
        if augment:
            transforms += self.augmentations
        transforms += [
            A.Normalize([0.5] * n_channels, [0.5] * n_channels),
            ap.ToTensorV2()
        ]
        self.transform = A.Compose(transforms)

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
        if self.mode:
            img = img.convert(self.mode)
        a = np.array(img)
        t = self.transform(image=a)['image']
        return t


class Horse2ZebraDataset(Dataset):
    def __init__(self, root_dir, split, suffix, augment=False):
        super().__init__(Path(root_dir) / 'horse2zebra' / (split + suffix), 3, mode='RGB', augment=augment)

    @property
    def transforms(self):
        return []

    @property
    def augmentations(self):
        return [A.HorizontalFlip()]


class HorseDataset(Horse2ZebraDataset):
    def __init__(self, root_dir, split, augment=False):
        super().__init__(root_dir, split, 'A', augment=augment)


class ZebraDataset(Horse2ZebraDataset):
    def __init__(self, root_dir, split, augment=False):
        super().__init__(root_dir, split, 'B', augment=augment)


class Generator(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.tail = self._conv(in_channels, 64)
        self.down = nn.Sequential(self._down(64, 128), self._down(128, 256))
        self.blocks = nn.ModuleList([self._residual(256) for _ in range(9)])
        self.up = nn.Sequential(self._up(256, 128), self._up(128, 64))
        self.head = self._conv(64, out_channels, head=True)

    @staticmethod
    def _conv(in_channels, out_channels, head=False):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 7, stride=1, padding=3, padding_mode='reflect', bias=head),
            *([nn.InstanceNorm2d(out_channels, affine=NORM_AFFINE)] if not head else []),
            nn.ReLU(inplace=True) if not head else nn.Tanh()
        )

    @staticmethod
    def _down(in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels, affine=NORM_AFFINE),
            nn.ReLU(inplace=True)
        )

    @staticmethod
    def _up(in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 3, stride=2, padding=1, output_padding=1, bias=False),
            nn.InstanceNorm2d(out_channels, affine=NORM_AFFINE),
            nn.ReLU(inplace=True)
        )

    @staticmethod
    def _residual(n_channels):
        return nn.Sequential(
            nn.Conv2d(n_channels, n_channels, 3, stride=1, padding=1, padding_mode='reflect', bias=False),
            nn.InstanceNorm2d(n_channels, affine=NORM_AFFINE),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_channels, n_channels, 3, stride=1, padding=1, padding_mode='reflect', bias=False),
            nn.InstanceNorm2d(n_channels, affine=NORM_AFFINE)
        )

    def forward(self, x):
        x = self.tail(x)
        x = self.down(x)
        for block in self.blocks:
            x = x + block(x)
        x = self.up(x)
        x = self.head(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, n_channels: int):
        super().__init__()
        self.layers = nn.Sequential(
            self._block(n_channels, 64, normalize=False),
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

    def forward(self, x):
        return self.layers(x).squeeze(1)


# https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/cyclegan/utils.py
class ReplayBuffer:
    def __init__(self, max_size=50, replace_prob=0.5):
        self.max_size = max_size
        self.replace_prob = replace_prob
        self.data = []

    def sample(self, data):
        to_return = []
        for element in data:
            element = element.unsqueeze(0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0, 1) < self.replace_prob:
                    i = random.randint(0, self.max_size - 1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return torch.cat(to_return)


class LinearLR:
    def __init__(self, epochs, decay_start_epoch):
        self.epochs = epochs
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1 - max(0, epoch - self.decay_start_epoch) / (self.epochs - self.decay_start_epoch)


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
    d = discriminator(y)
    assert d.size() == (batch_size, 30, 30)
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
def evaluate(generatorA, generatorB, train_datasetA_static, val_dataloaderA, l1_loss, epoch, epochs, num_batches,
        log_train_indices, log_val_indices, writer, device, enable_amp, initial):
    eval_losses = []
    loop = tqdm(val_dataloaderA, desc=f'Validating', file=sys.stdout)
    for batch_idx, a in enumerate(loop):
        a = a.to(device)
        with torch.cuda.amp.autocast(enabled=enable_amp):
            fakeB = generatorB(a)
            cycledA = generatorA(fakeB)
            gen_loss = l1_loss(cycledA, a)
        eval_losses.append(gen_loss.item())
    eval_loss = sum(eval_losses) / len(eval_losses)
    print('Cycle loss:', f'{eval_loss:.2e}')
    step = (epoch + 1) * num_batches
    writer.add_scalar('Epoch', epoch + 1, step)
    writer.add_scalar('Loss/validation', eval_loss, step)
    if (epoch + 1) % SAVE_IMAGES_EVERY == 0 or epoch == epochs - 1:
        for dataset, indices, name in [(train_datasetA_static, log_train_indices, 'Train'),
                                       (val_dataloaderA.dataset, log_val_indices, 'Validation')]:
            items = [dataset[i] for i in indices]
            a = torch.stack(items).to(device)
            fakeB = generatorB(a)
            cycledA = generatorA(fakeB)
            grid_kwargs = dict(nrow=math.ceil(SAVE_IMAGES / math.sqrt(SAVE_IMAGES)), normalize=True,
                value_range=(-1, 1))
            grid_fake = vutils.make_grid(fakeB, **grid_kwargs)
            grid_cycled = vutils.make_grid(cycledA, **grid_kwargs)
            writer.add_image(f'{name} images/generated', grid_fake, step)
            writer.add_image(f'{name} images/reconstructed', grid_cycled, step)
            if initial:
                grid_input = vutils.make_grid(a, **grid_kwargs)
                writer.add_image(f'{name} images/input', grid_input, step)
        initial = False
    return initial


def main():
    dataset_config = {
        'horse2zebra': (HorseDataset, ZebraDataset, 3, 3)
    }
    dataset_name = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DATASET
    datasetA_class, datasetB_class, A_channels, B_channels = dataset_config[dataset_name]

    if LAMBDA_ID:
        assert A_channels == B_channels

    generatorA = Generator(B_channels, A_channels)
    generatorB = Generator(A_channels, B_channels)
    discriminatorA = Discriminator(A_channels)
    discriminatorB = Discriminator(B_channels)

    has_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if has_cuda else 'cpu')
    generatorA.to(device)
    generatorB.to(device)
    discriminatorA.to(device)
    discriminatorB.to(device)

    root_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_ROOT_DIR

    data_dir = str(root_dir / 'data')
    train_datasetA = datasetA_class(data_dir, 'train', augment=True)
    train_datasetB = datasetB_class(data_dir, 'train', augment=True)
    train_datasetA_static = datasetA_class(data_dir, 'train', augment=False)
    val_datasetA = datasetA_class(data_dir, 'test', augment=False)

    train_dataloaderA = data.DataLoader(train_datasetA, batch_size=BATCH_SIZE, shuffle=True, drop_last=True,
        num_workers=os.cpu_count(), persistent_workers=True, pin_memory=True)
    train_dataloaderB = data.DataLoader(train_datasetB, batch_size=BATCH_SIZE, shuffle=True, drop_last=True,
        num_workers=os.cpu_count(), persistent_workers=True, pin_memory=True)
    val_dataloaderA = data.DataLoader(val_datasetA, batch_size=VAL_BATCH_SIZE, shuffle=False, drop_last=False,
        num_workers=os.cpu_count(), persistent_workers=True, pin_memory=True)

    mse_loss = nn.MSELoss()
    l1_loss = nn.L1Loss()

    gen_optimizer = optim.Adam(list(generatorA.parameters()) + list(generatorB.parameters()), lr=LEARNING_RATE,
        betas=BETAS)
    disc_optimizer = optim.Adam(list(discriminatorA.parameters()) + list(discriminatorB.parameters()), lr=LEARNING_RATE,
        betas=BETAS)

    enable_amp = ENABLE_AMP and has_cuda
    scaler = torch.cuda.amp.GradScaler(enabled=enable_amp)

    epochs = int(sys.argv[3]) if len(sys.argv) > 3 else DEFAULT_EPOCHS

    if len(sys.argv) > 4:
        checkpoint = torch.load(sys.argv[4])
        start_epoch = checkpoint['epoch'] + 1
        generatorA.load_state_dict(checkpoint['generatorA'])
        generatorB.load_state_dict(checkpoint['generatorB'])
        discriminatorA.load_state_dict(checkpoint['discriminatorA'])
        discriminatorB.load_state_dict(checkpoint['discriminatorB'])
        gen_optimizer.load_state_dict(checkpoint['gen_optimizer'])
        disc_optimizer.load_state_dict(checkpoint['disc_optimizer'])
        log_train_indices = checkpoint['log_train_indices']
        log_val_indices = checkpoint['log_val_indices']
    else:
        start_epoch = 0
        generatorA.apply(init_weights)
        generatorB.apply(init_weights)
        discriminatorA.apply(init_weights)
        discriminatorB.apply(init_weights)
        log_train_indices = torch.randint(0, len(train_datasetA_static), (SAVE_IMAGES,))
        log_val_indices = torch.randint(0, len(val_datasetA), (SAVE_IMAGES,))

    gen_lr_scheduler = optim.lr_scheduler.LambdaLR(gen_optimizer, LinearLR(epochs, epochs // 2).step,
        last_epoch=(start_epoch - 1))
    disc_lr_scheduler = optim.lr_scheduler.LambdaLR(disc_optimizer, LinearLR(epochs, epochs // 2).step,
        last_epoch=(start_epoch - 1))

    print('Domain A:')
    test(generatorA, discriminatorA, B_channels, A_channels, device, verbose=1)
    print('Domain B:')
    test(generatorB, discriminatorB, A_channels, B_channels, device, verbose=1)

    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')

    log_dir = str(root_dir / 'logs' / Path(__file__).stem / dataset_name / timestamp)
    writer = tb.SummaryWriter(log_dir=log_dir)

    generatorA.train()
    generatorB.train()
    discriminatorA.train()
    discriminatorB.train()

    label_real = torch.tensor(1.0, device=device)
    label_fake = torch.tensor(0.0, device=device)

    fakeA_buffer = ReplayBuffer(max_size=50)
    fakeB_buffer = ReplayBuffer(max_size=50)

    num_batches = min(len(train_dataloaderA), len(train_dataloaderB))
    step = start_epoch * num_batches
    gen_losses = []
    disc_losses = []
    mean_gen_loss = 0
    mean_disc_loss = 0
    get_postfix = lambda: dict(gen_loss=f'{mean_gen_loss:.2e}', disc_loss=f'{mean_disc_loss:.2e}')
    ckpt_dir = root_dir / 'ckpt' / Path(__file__).stem / dataset_name / timestamp
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    initial_eval = True
    for epoch in range(start_epoch, epochs):
        loop = trange(num_batches, desc=f'Epoch {epoch + 1}/{epochs}', postfix=get_postfix(), file=sys.stdout)
        train_iterA = iter(train_dataloaderA)
        train_iterB = iter(train_dataloaderB)
        for batch_idx in loop:
            a = next(train_iterA)
            b = next(train_iterB)
            a = a.to(device)
            b = b.to(device)
            batch_len = a.size(0)
            assert b.size(0) == batch_len

            targets_real = label_real.expand(batch_len, 30, 30)
            targets_fake = label_fake.expand(batch_len, 30, 30)

            discriminatorA.requires_grad_(False)
            discriminatorB.requires_grad_(False)
            with torch.cuda.amp.autocast(enabled=enable_amp):
                fakeA = generatorA(b)
                fakeB = generatorB(a)
                cycledA = generatorA(fakeB)
                cycledB = generatorB(fakeA)
                preds_fakeA = discriminatorA(fakeA)
                preds_fakeB = discriminatorB(fakeB)
                gen_loss_ganA = mse_loss(preds_fakeA, targets_real)
                gen_loss_ganB = mse_loss(preds_fakeB, targets_real)
                gen_loss_cycleA = l1_loss(cycledA, a)
                gen_loss_cycleB = l1_loss(cycledB, b)
                gen_loss = gen_loss_ganA + gen_loss_ganB + (gen_loss_cycleA + gen_loss_cycleB) * LAMBDA_CYCLE
                if LAMBDA_ID:
                    selfA = generatorA(a)
                    selfB = generatorB(b)
                    gen_loss_idA = l1_loss(selfA, a)
                    gen_loss_idB = l1_loss(selfB, b)
                    gen_loss = gen_loss + (gen_loss_idA + gen_loss_idB) * LAMBDA_ID
            gen_optimizer.zero_grad()
            scaler.scale(gen_loss).backward()
            scaler.step(gen_optimizer)
            scaler.update()

            discriminatorA.requires_grad_(True)
            discriminatorB.requires_grad_(True)
            with torch.cuda.amp.autocast(enabled=enable_amp):
                fakeA = fakeA_buffer.sample(fakeA.detach())
                fakeB = fakeB_buffer.sample(fakeA.detach())
                preds_realA = discriminatorA(a)
                preds_realB = discriminatorB(b)
                preds_fakeA = discriminatorA(fakeA)
                preds_fakeB = discriminatorB(fakeB)
                disc_loss_realA = mse_loss(preds_realA, targets_real)
                disc_loss_realB = mse_loss(preds_realB, targets_real)
                disc_loss_fakeA = mse_loss(preds_fakeA, targets_fake)
                disc_loss_fakeB = mse_loss(preds_fakeB, targets_fake)
                disc_loss = (disc_loss_realA + disc_loss_realB + disc_loss_fakeA + disc_loss_fakeB) * 0.5
            disc_optimizer.zero_grad()
            scaler.scale(disc_loss).backward()
            scaler.step(disc_optimizer)
            scaler.update()

            gen_losses.append(gen_loss.item())
            disc_losses.append(disc_loss.item())

            loop.set_postfix(**get_postfix())

            if (step % LOG_EVERY == 0 or epoch == epochs - 1 and batch_idx == num_batches - 1) and gen_losses:
                mean_gen_loss = sum(gen_losses) / len(gen_losses)
                mean_disc_loss = sum(disc_losses) / len(disc_losses)

                writer.add_scalar('Loss/generators', mean_gen_loss, step)
                writer.add_scalar('Loss/discriminators', mean_disc_loss, step)

                gen_losses = []
                disc_losses = []

            step += 1

        initial_eval = evaluate(generatorA, generatorB, train_datasetA_static, val_dataloaderA, l1_loss, epoch, epochs,
            num_batches, log_train_indices, log_val_indices, writer, device, enable_amp, initial_eval)

        if (epoch + 1) % SAVE_EVERY == 0 or (epoch + 1) == epochs:
            checkpoint = {
                'epoch': epoch,
                'generatorA': generatorA.state_dict(),
                'generatorB': generatorB.state_dict(),
                'discriminatorA': discriminatorA.state_dict(),
                'discriminatorB': discriminatorB.state_dict(),
                'gen_optimizer': gen_optimizer.state_dict(),
                'disc_optimizer': disc_optimizer.state_dict(),
                'log_train_indices': log_train_indices,
                'log_val_indices': log_val_indices
            }
            ckpt_path = ckpt_dir / f'epoch-{epoch + 1}.pt'
            torch.save(checkpoint, ckpt_path)

        gen_lr_scheduler.step()
        disc_lr_scheduler.step()


if __name__ == '__main__':
    if sys.platform == 'darwin':
        multiprocessing.set_start_method('fork')
    main()
