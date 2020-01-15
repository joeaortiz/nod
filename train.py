import argparse
import torch
import util
import datetime
import os
import pickle

import numpy as np
import logging

from torch.utils import data
import torchvision
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import torch.nn as nn

import dataio
import model
import data_util
import util

parser = argparse.ArgumentParser()
parser.add_argument('--batch-size', type=int, default=10,
                    help='Batch size.')
parser.add_argument('--epochs', type=int, default=50,
                    help='Number of training epochs.')
parser.add_argument('--learning-rate', type=float, default=5e-4,
                    help='Learning rate.')

parser.add_argument('--encoder', type=str, default='cswm',
                    help='Object inference model / encoder.')
parser.add_argument('--decoder', type=str, default='broadcast',
                    help='Decoder of latent to image space.')

parser.add_argument('--hidden-dim', type=int, default=512,
                    help='Number of hidden units in object extractor and object encoder.')
parser.add_argument('--embedding-dim', type=int, default=256,
                    help='Dimensionality of embedding.')
parser.add_argument('--action-dim', type=int, default=12,
                    help='Dimensionality of action space.')
parser.add_argument('--num-slots', type=int, default=3,
                    help='Number of object slots in model.')

parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disable CUDA training.')

parser.add_argument('--seed', type=int, default=42,
                    help='Random seed (default: 42).')
parser.add_argument('--log-interval', type=int, default=5,
                    help='How many batches to wait before logging'
                         'training status.')

parser.add_argument('--train_dir', type=str,
                    default='/mnt/sda/clevr-dataset-gen-1/manyviews/scenes/edgar',
                    help='Path to training dataset.')
parser.add_argument('--num_pairs_per_instance', type=int, default=20,
                    help='Number of pairs of views per scene.')
parser.add_argument('--name', type=str, default='test',
                    help='Experiment name.')
parser.add_argument('--save-folder', type=str,
                    default='logs',
                    help='Path to checkpoints.')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

device = torch.device('cuda' if args.cuda else 'cpu')

dataset = dataio.TwoViewsDataset(data_dir=args.train_dir,
                                 num_pairs_per_instance=args.num_pairs_per_instance)
train_loader = data.DataLoader(dataset, batch_size=args.batch_size,
                               shuffle=True, num_workers=4)

print(f'Size of dataset {len(dataset)}')

# obs = train_loader.__iter__().next()
# data_util.show_batch_pairs(obs)
# input_shape = obs['image1'].size()[1:]
input_shape = torch.Size([3, 128, 128])

model = model.NodModel(
    embedding_dim=args.embedding_dim,
    input_dims=input_shape,
    hidden_dim=args.hidden_dim,
    num_slots=args.num_slots,
    encoder=args.encoder,
    decoder=args.decoder)
model.to(device)

# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv2d(3, 6, 5)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(6, 16, 5)
#         self.fc1 = nn.Linear(16 * 29 * 29, 120)
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, 10)
#
#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = x.view(-1, 16 * 29 * 29)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x
# model = Net().to(device)

model.apply(util.weights_init)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=args.learning_rate)

now = datetime.datetime.now()
timestamp = now.isoformat()

if args.name == 'none':
    exp_name = timestamp
else:
    exp_name = args.name

exp_counter = 0
save_folder = '{}/{}/'.format(args.save_folder, exp_name)

ckpt_dir = os.path.join(save_folder, 'checkpoints')
events_dir = os.path.join(save_folder, 'events')

util.cond_mkdir(save_folder)
util.cond_mkdir(ckpt_dir)
util.cond_mkdir(events_dir)

# Save command-line parameters log directory.
with open(os.path.join(save_folder, "params.txt"), "w") as out_file:
    out_file.write('\n'.join(["%s: %s" % (key, value) for key, value in vars(args).items()]))

# Save text summary of model into log directory.
with open(os.path.join(save_folder, "model.txt"), "w") as out_file:
    out_file.write(str(model))

writer = SummaryWriter(events_dir)

# writer.add_graph(model)


# Train model.
print('Starting model training...')
print("\n" + "#" * 10)
print("Training for %d epochs with batch size %d" % (args.epochs, args.batch_size))
print("#" * 10 + "\n")
step = 0
best_loss = 1e9

criterion = nn.MSELoss(reduction="mean")

for epoch in range(1, args.epochs + 1):
    model.train()
    train_loss = 0.

    for batch_idx, data_batch in enumerate(train_loader):
        img1, img2 = data_batch['image1'].to(device), data_batch['image2'].to(device)
        imgs = torch.cat((img1, img2), dim=0)

        optimizer.zero_grad()

        recs = model(imgs)

        # Decode embedding
        # flat_state = state.reshape(-1, state.size(2))
        # comps, masks = model.decoder(flat_state)

        # comps = comps.view(-1, self.num_slots, comps.size(1), comps.size(2), comps.size(3))
        # masks = masks.view(-1, self.num_slots, masks.size(1), masks.size(2))
        # scaled_masks = F.softmax(masks, dim=1)
        #
        # masked_comps = torch.mul(scaled_masks.unsqueeze(2), comps)
        # recs = masked_comps.sum(dim=1)

        # l2_loss = model.get_pixel_loss(recs, imgs)

        # total_loss = l2_loss
        # target = torch.zeros_like(recs).to(device)
        # total_loss = criterion(recs, target)
        #
        # total_loss.backward()
        # optimizer.step()
        #
        # train_loss += total_loss.item()
        total_loss = 0.

        if batch_idx % args.log_interval == 0:
            print(f" Epoch {epoch:03d}   Iter {step:07d}  Loss {total_loss:06f} ")

        # writer.add_scalar("total_loss", total_loss, step)
        # model.write_updates(writer, recs, data_batch, step)
        #
        step += 1

    avg_loss = train_loss / len(train_loader.dataset)
    print('====> Epoch: {} Average loss: {:.6f}'.format(
        epoch, avg_loss))

    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(model.state_dict(),
                   os.path.join(ckpt_dir, 'epoch_%04d_iter_%06d.pth' % (epoch, step)))


