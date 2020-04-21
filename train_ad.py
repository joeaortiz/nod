import configargparse
import torch
import util
import datetime
import os
import pickle

import numpy as np
import matplotlib.pyplot as plt
import logging

from torch.utils import data
import torchvision
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import torch.nn as nn

import dataio
from custom_modules import autodecoder
import data_util
import util

""" Parameters"""
img_size = [128, 128]
embedding_dim = 128
n_slots = 3

epochs = 10000
learning_rate = 5e-4
batch_size = 4

scene_dir = '/mnt/sda/clevr-dataset-gen-1/datasets/manyviews/train/000000/'
num_views = 4

events_dir = 'logs/ad/events'

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

device = torch.device('cuda')

dataset = dataio.PosedImagesDataset(data_dir=scene_dir,
                                    num_views=num_views)

loader = data.DataLoader(dataset, batch_size=batch_size,
                         shuffle=True, num_workers=4)

model = autodecoder.StructuredAutoDecoder(img_size=img_size,
                                          embedding_dim=embedding_dim,
                                          n_slots=n_slots,
                                          action_dim=12,
                                          residual=True).to(device)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=learning_rate)

l2_loss = nn.MSELoss(reduction="mean")

util.cond_mkdir(events_dir)
writer = SummaryWriter(events_dir)


print("Initialising random weights")
model.apply(util.weights_init)

print("Number of views of scene ", len(dataset))

print("\nStarting training...")

step = 0
for epoch in range(1, epochs + 1):
    model.train()
    train_loss = 0.

    for batch_idx, data_batch in enumerate(loader):
        images = data_batch['image'].to(device)
        optimizer.zero_grad()

        out = model(data_batch['pose'].to(device))
        masks, masked_comps, recs = model.compose_image(out)

        loss = l2_loss(recs, images)

        # Backprop and optimise
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        step += 1

        writer.add_scalar("loss", loss, step)
        if not step % 5:
            model.write_updates(writer, recs, images,
                                masks, masked_comps, step)

        if not step % 10:
            print(f'Step: {step} Average loss: {loss.item() / batch_size:.6f}')

    avg_loss = train_loss / len(loader.dataset)
    print('====> Epoch: {} Average loss: {:.6f}'.format(
        epoch, avg_loss))

