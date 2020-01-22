import configargparse
import os, time, datetime, yaml, shutil

import torch
import numpy as np

from torch.utils.data import DataLoader
from torch.utils import data
import torch.nn as nn
import torchvision

import dataio
import nod
import data_util
import util
import matplotlib.pyplot as plt


parser = configargparse.ArgumentParser()
parser.add('-c', '--config_filepath', required=False, is_config_file=True, help='Path to config file.')

# Specify test data
parser.add_argument('--test_dir', type=str,
                    default='/mnt/sda/clevr-dataset-gen-1/manyviews/scenes/test',
                    help='Path to training dataset.')
parser.add_argument('--test_pairs_per_scene', type=int, default=20,
                    help='Number of pairs of views per scene.')
parser.add_argument('--num_test_scenes', type=int, default=-1,
                    help='Number of different scene instances to use. -1 is use all scenes. ')

parser.add_argument('--train_log_dir', type=str, default='./logs', required=True,
                    help='Path to training log directory.')
parser.add_argument('--checkpoint_path', default=None,
                    help='Path to trained model.')
parser.add_argument('--results_dir', default=None,
                    help='Path to save evaluation results too.')

parser.add_argument('--batch_size', type=int, default=10, help='Batch size.')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disable CUDA.')

parser.add_argument('--save_out_first_n', type=int, default=50,
                    help='Only saves images of first n instances.')
parser.add_argument('--circle_source_img_path', type=str, default=None,
                    help='Path(s) to source image(s) from which circle of views around objects are generated. ')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

device = torch.device('cuda' if args.cuda else 'cpu')

def test():

    test_dataset = dataio.TwoViewsDataset(data_dir=args.test_dir,
                                          num_pairs_per_scene=args.test_pairs_per_scene,
                                          num_scenes=args.num_test_scenes)
    test_loader = data.DataLoader(test_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  num_workers=4)

    print(f'Size of test dataset {len(test_dataset)}')

    obs = test_loader.__iter__().next()
    data_util.show_batch_pairs(obs)
    input_shape = obs['image1'].size()[1:]

    # Load training params
    with open(args.train_log_dir + '/params.txt', 'r') as f:
        train_params = yaml.safe_load(f)

    model = nod.NodModel(
        embedding_dim=train_params['embedding_dim'],
        input_dims=input_shape,
        hidden_dim=train_params['hidden_dim'],
        num_slots=train_params['num_slots'],
        encoder=train_params['encoder'],
        decoder=train_params['decoder'])

    print("Loading model from %s" % args.checkpoint_path)
    util.custom_load(model, path=args.checkpoint_path)

    model.to(device)
    model.eval()

    gt_comparison_dir = os.path.join(args.results_dir, 'gt_comparisons')
    sv_comps_dir = os.path.join(args.results_dir, 'components_same_view')
    dv_comps_dir = os.path.join(args.results_dir, 'components_diff_view')
    util.cond_mkdir(args.results_dir)
    util.cond_mkdir(gt_comparison_dir)
    util.cond_mkdir(sv_comps_dir)
    util.cond_mkdir(dv_comps_dir)

    # Save command-line parameters to log directory.
    with open(os.path.join(args.results_dir, "params.txt"), "w") as out_file:
        out_file.write('\n'.join(["%s: %s" % (key, value) for key, value in vars(args).items()]))

    l2_loss = nn.MSELoss(reduction="mean")

    print('Beginning evaluation...')
    with torch.no_grad():
        same_view_losses = []
        diff_view_losses = []
        total_losses = []
        for batch_idx, data_batch in enumerate(test_loader):
            img1, img2 = data_batch['image1'].to(device), data_batch['image2'].to(device)
            batch_size = img1.shape[0]
            imgs = torch.cat((img1, img2), dim=0)
            w, h = imgs.size(-2), imgs.size(-1)
            images_gt = imgs.reshape(-1, 2, 3, w, h)

            action1, action2 = data_batch['transf21'].to(device), data_batch['transf12'].to(device)
            actions = torch.cat((action1, action2), dim=0)

            out = model(imgs, actions)
            masks, masked_comps, recs = model.compose_image(out)

            rec_views = recs[:batch_size * 2]
            novel_views = recs[batch_size * 2:]

            same_view_loss = l2_loss(rec_views, imgs)
            novel_view_loss = l2_loss(novel_views, imgs)
            total_loss = same_view_loss + novel_view_loss
            same_view_losses.append(same_view_loss.item())
            diff_view_losses.append(novel_view_loss.item())
            total_losses.append(total_loss.item())
            print(f"Number input images {batch_idx * args.batch_size}  |  Running l2 loss: {np.mean(total_losses)}")

            if batch_idx * args.batch_size < args.save_out_first_n:

                rec_views = rec_views.reshape(args.batch_size, 2, 3, w, h)
                novel_views = novel_views.reshape(args.batch_size, 2, 3, w, h)
                same_view_masked_comps = masked_comps[:args.batch_size * 2].reshape(
                    args.batch_size, 2, model.num_slots, 3, w, h)
                diff_view_masked_comps = masked_comps[args.batch_size * 2:].reshape(
                    args.batch_size, 2, model.num_slots, 3, w, h)
                same_view_masks = masks[args.batch_size * 2:].reshape(args.batch_size, 2, model.num_slots, w, h)
                diff_view_masks = masks[args.batch_size * 2:].reshape(args.batch_size, 2, model.num_slots, w, h)
                # Expand to have 3 channels so can concat with rgb images
                same_view_masks = same_view_masks.unsqueeze(3).repeat(1, 1, 1, 3, 1, 1)
                diff_view_masks = diff_view_masks.unsqueeze(3).repeat(1, 1, 1, 3, 1, 1)
                # Shift to be in range [-1, 1] like rgb
                same_view_masks = same_view_masks*2 - 1
                diff_view_masks = diff_view_masks*2 - 1

                for i in range(args.batch_size):
                    gt = images_gt[i]
                    same_view_rec = rec_views[i]
                    diff_view_rec = novel_views[i]

                    # Save ground truth reconstruction comparison
                    gt_vs_rec_vs_nv = torch.cat((gt,
                                                same_view_rec,
                                                diff_view_rec), dim=0)
                    gt_comparison_imgs = torchvision.utils.make_grid(gt_vs_rec_vs_nv,
                                                                     nrow=2,
                                                                     scale_each=False,
                                                                     normalize=True,
                                                                     range=(-1, 1)).cpu().detach().numpy()
                    plt.imsave(os.path.join(gt_comparison_dir, f'{i + batch_idx * args.batch_size:04d}.png'),
                               np.transpose(gt_comparison_imgs, (1, 2, 0)))

                    # Save components
                    sv_images = torch.cat((images_gt[i].unsqueeze(1),
                                           same_view_rec.unsqueeze(1),
                                           same_view_masked_comps[i],
                                           same_view_masks[i]), dim=1)
                    dv_images = torch.cat((images_gt[i].unsqueeze(1),
                                           diff_view_rec.unsqueeze(1),
                                           diff_view_masked_comps[i],
                                           diff_view_masks[i]), dim=1)

                    comps_same_view_images = torchvision.utils.make_grid(sv_images.reshape(-1, 3, h, w),
                                                                         nrow=2*model.num_slots + 2,
                                                                         scale_each=False,
                                                                         normalize=True,
                                                                         range=(-1, 1)).cpu().detach().numpy()
                    comps_diff_view_images = torchvision.utils.make_grid(dv_images.reshape(-1, 3, h, w),
                                                                         nrow=2*model.num_slots + 2,
                                                                         scale_each=False,
                                                                         normalize=True,
                                                                         range=(-1, 1)).cpu().detach().numpy()
                    plt.imsave(os.path.join(sv_comps_dir, f'{i + batch_idx * args.batch_size:04d}.png'),
                               np.transpose(comps_same_view_images, (1, 2, 0)))
                    plt.imsave(os.path.join(dv_comps_dir, f'{i + batch_idx * args.batch_size:04d}.png'),
                               np.transpose(comps_diff_view_images, (1, 2, 0)))

        save_circles(model, args.results_dir, args.circle_source_img_path.split())

    with open(os.path.join(args.results_dir, "results.txt"), "w") as out_file:
        out_file.write("Evaluation Metric: score \n\n")
        out_file.write(f"Same view rec l2 loss: {np.mean(same_view_losses):10f} \n")
        out_file.write(f"Diff view rec l2 loss: {np.mean(diff_view_losses):10f} \n")
        out_file.write(f"Rec l2 loss: {np.mean(total_losses):10f} \n")

    print("\nFinal score: ")


def save_circles(model, results_dir, img_paths, num_renders=50):
    circles_dir = os.path.join(results_dir, 'circles')
    util.cond_mkdir(circles_dir)

    print('Generating circle views around scene')

    for j, img_path in enumerate(img_paths):
        circle_dir = os.path.join(circles_dir, f'{j:03d}')
        util.cond_mkdir(circle_dir)
        # Copy reference image into directory
        shutil.copy(img_path, os.path.join(circle_dir, '0000_ref_image.png'))

        img = torch.from_numpy(data_util.load_rgb(img_path).transpose(2, 0, 1)).to(device).unsqueeze(0)
        split = img_path.split('/')
        split[-1] = split[-1].split('.')[0] + '.txt'
        split[-2] = 'pose'
        ref_pose = data_util.load_pose('/'.join(split))
        sample_poses = data_util.gen_pose_circle(ref_pose, n_poses=num_renders, centre=[0., 0., 0.])
        actions = np.zeros([num_renders, 12])

        for i, target_pose in enumerate(sample_poses):
            actions[i] = (np.linalg.inv(target_pose) @ ref_pose).flatten()[:12]
        actions = torch.from_numpy(actions).float().to(device)

        with torch.no_grad():
            state = model.encoder(img)
            state = state.repeat(num_renders, 1, 1)

            state = model.transition_model(state, actions)

            out = model.decoder(state.reshape(-1, model.embedding_dim))
            _, _, recs = model.compose_image(out)

            for i, rec in enumerate(recs):
                torchvision.utils.save_image(rec,
                                             os.path.join(circle_dir, f'{i+1:04d}.png'),
                                             normalize=True,
                                             range=(-1, 1))
        print('Saved one circle view.')


def main():
    test()


if __name__ == '__main__':
    main()
