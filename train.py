import configargparse
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
import nod
import data_util
import util

parser = configargparse.ArgumentParser()
parser.add('-c', '--config_filepath', required=False, is_config_file=True, help='Path to config file.')

parser.add_argument('--batch_size', type=int, default=2,
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

parser.add_argument('--identity_action', action='store_true', default=False,
                    help='Should we use the transition model conditioned on identity relative '
                         'pose to condition same view rendering?')
parser.add_argument('--residual', action='store_true', default=False,
                    help='Should we use residual connections in the transition model?')

parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disable CUDA training.')

parser.add_argument('--seed', type=int, default=42,
                    help='Random seed (default: 42).')
parser.add_argument('--log-interval', type=int, default=5,
                    help='How many batches to wait before logging'
                         'training status.')

parser.add_argument('--train_dir', type=str,
                    default='/mnt/sda/clevr-dataset-gen-1/manyviews/scenes/train',
                    help='Path to training dataset.')
parser.add_argument('--train_pairs_per_scene', type=int, default=20,
                    help='Number of pairs of views per scene.')
parser.add_argument('--num_train_scenes', type=int, default=-1,
                    help='Number of different scene instances to use. -1 is use all scenes. ')

parser.add_argument('--val_dir', type=str,
                    default='/mnt/sda/clevr-dataset-gen-1/manyviews/scenes/val',
                    help='Path to validation dataset.')
parser.add_argument('--val_pairs_per_scene', type=int, default=20,
                    help='Number of pairs of views per scene.')
parser.add_argument('--num_val_scenes', type=int, default=-1,
                    help='Number of different scene instances to use. -1 is use all scenes. ')

parser.add_argument('--steps_til_val', type=int, default=200,
               help='Number of iterations until validation set is run.')
parser.add_argument('--no_validation', action='store_true', default=False,
               help='If no validation set should be used.')

parser.add_argument('--name', type=str, default='test',
                    help='Experiment name.')
parser.add_argument('--save-folder', type=str,
                    default='logs',
                    help='Path to checkpoints.')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

device = torch.device('cuda' if args.cuda else 'cpu')

train_dataset = dataio.TwoViewsDataset(data_dir=args.train_dir,
                                       num_pairs_per_scene=args.train_pairs_per_scene,
                                       num_scenes=args.num_train_scenes)
train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size,
                               shuffle=True, num_workers=4)
if not args.no_validation:
    val_batch_size = 10
    val_dataset = dataio.TwoViewsDataset(data_dir=args.val_dir,
                                         num_pairs_per_scene=args.val_pairs_per_scene,
                                         num_scenes=args.num_val_scenes)
    val_loader = data.DataLoader(val_dataset, batch_size=val_batch_size,
                                 shuffle=True, num_workers=4)


print(f'Size of dataset {len(train_dataset)}')

# obs = train_loader.__iter__().next()
# data_util.show_batch_pairs(obs)
# input_shape = obs['image1'].size()[1:]
input_shape = torch.Size([3, 128, 128])

model = nod.NodModel(
    embedding_dim=args.embedding_dim,
    input_dims=input_shape,
    hidden_dim=args.hidden_dim,
    num_slots=args.num_slots,
    encoder=args.encoder,
    decoder=args.decoder,
    identity_action=args.identity_action,
    residual=args.residual)
model.to(device)

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
print("Training for %d epochs with batch size %d and %d steps per batch"
      % (args.epochs, args.batch_size, np.ceil(len(train_dataset) / args.batch_size)))
print("#" * 10 + "\n")
step = 0
best_loss = 1e9

l2_loss = nn.MSELoss(reduction="mean")

for epoch in range(1, args.epochs + 1):
    model.train()
    train_loss = 0.

    for batch_idx, data_batch in enumerate(train_loader):
        img1, img2 = data_batch['image1'].to(device), data_batch['image2'].to(device)
        batch_size = img1.shape[0]
        imgs = torch.cat((img1, img2), dim=0)
        action1, action2 = data_batch['transf21'].to(device), data_batch['transf12'].to(device)
        actions = torch.cat((action1, action2), dim=0)

        optimizer.zero_grad()

        out = model(imgs, actions)
        masks, masked_comps, recs = model.compose_image(out)

        rec_views = recs[:batch_size*2]
        novel_views = recs[batch_size*2:]

        same_view_loss = l2_loss(rec_views, imgs)
        novel_view_loss = l2_loss(novel_views, imgs)

        total_loss = same_view_loss + novel_view_loss

        # Backprop and optimise
        total_loss.backward()
        optimizer.step()

        train_loss += total_loss.item()

        if batch_idx % args.log_interval == 0:
            print(f" Epoch {epoch:03d}   Iter {step:07d} | " +
                  f"Same View Loss {same_view_loss.item():06f} | " +
                  f"Novel View Loss {novel_view_loss.item():06f} | " +
                  f"Total Loss {total_loss.item():06f}  ")

        writer.add_scalar("same_view_loss", same_view_loss, step)
        writer.add_scalar("novel_view_loss", novel_view_loss, step)
        writer.add_scalar("total_loss", total_loss, step)

        if not step % 100:
            model.write_updates(writer, recs, imgs,
                                masks, masked_comps, step)

        if step % args.steps_til_val == 0 and not args.no_validation:
            print("Running validation set...")

            model.eval()
            with torch.no_grad():
                same_view_losses = []
                diff_view_losses = []
                total_losses = []
                for val_batch in val_loader:
                    img1, img2 = val_batch['image1'].to(device), val_batch['image2'].to(device)
                    val_batch_size = img1.shape[0]
                    imgs = torch.cat((img1, img2), dim=0)
                    action1, action2 = val_batch['transf21'].to(device), val_batch['transf12'].to(device)
                    actions = torch.cat((action1, action2), dim=0)

                    out = model(imgs, actions)
                    masks, masked_comps, recs = model.compose_image(out)

                    rec_views = recs[:val_batch_size * 2]
                    novel_views = recs[val_batch_size * 2:]

                    same_view_loss = l2_loss(rec_views, imgs)
                    novel_view_loss = l2_loss(novel_views, imgs)

                    total_loss = same_view_loss + novel_view_loss
                    same_view_losses.append(same_view_loss.item())
                    diff_view_losses.append(novel_view_loss.item())
                    total_losses.append(total_loss.item())

                model.write_updates(writer, recs, imgs,
                                    masks, masked_comps, step, prefix='val_')

                writer.add_scalar("val_same_view_loss", np.mean(same_view_losses), step)
                writer.add_scalar("val_diff_view_loss", np.mean(diff_view_losses), step)
                writer.add_scalar("val_total_loss", np.mean(total_losses), step)
            model.train()

        step += 1

    avg_loss = train_loss / len(train_loader.dataset)
    print('====> Epoch: {} Average loss: {:.6f}'.format(
        epoch, avg_loss))

    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(model.state_dict(),
                   os.path.join(ckpt_dir, 'epoch_%04d_iter_%06d.pth' % (epoch, step)))


