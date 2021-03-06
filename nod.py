import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F

import numpy as np

import modules
from custom_modules import attention, spd
import util

class NodModel(nn.Module):
    """Main module.

    Args:
        embedding_dim: Dimensionality of latent state space.
        input_dims: Shape of input observation.
        hidden_dim: Number of hidden units in encoder and transition model.
        action_dim: Dimensionality of action space.
        num_objects: Number of object slots.
    """

    def __init__(self, embedding_dim, input_dims, hidden_dim,
                 num_slots,
                 encoder='cswm', cnn_size='small',
                 decoder='broadcast',
                 trans_model='gnn',
                 identity_action=False, residual=False,
                 canonical=False):
        super(NodModel, self).__init__()
        self.embedding_dim = embedding_dim
        self.input_dims = input_dims
        self.hidden_dims = hidden_dim

        self.num_slots = num_slots
        self.identity_action_flag = identity_action
        self.canonical = canonical

        if encoder == 'cswm':
            self.encoder = modules.EncoderCSWM(input_dims=self.input_dims,
                                               embedding_dim=self.embedding_dim,
                                               num_objects=self.num_slots,
                                               cnn_size=cnn_size)

        if trans_model == 'gnn':
            self.transition_model = modules.TransitionGNN(input_dim=self.embedding_dim,
                                                          hidden_dim=512,
                                                          action_dim=12,
                                                          num_objects=self.num_slots,
                                                          residual=residual)
        elif trans_model == 'attention':
            self.transition_model = attention.MultiHeadCondAttention(n_head=5,
                                                                     input_feature_dim=self.embedding_dim + 12,
                                                                     out_dim=self.embedding_dim,
                                                                     dim_k=128,
                                                                     dim_v=128)

        if decoder == 'broadcast':
            self.decoder = spd.BroadcastDecoder(latent_dim=self.embedding_dim,
                                                output_dim=4,  # 3 rgb channels and one mask
                                                hidden_channels=32,
                                                num_layers=4,
                                                img_dims=self.input_dims[1:],  # width and height of square image
                                                act_fn='elu')
        elif decoder == 'cnn':
            out_shape = self.input_dims
            out_shape[0] += 1
            self.decoder = modules.DecoderCNNMedium(input_dim=self.embedding_dim,
                                                    hidden_dim=32,
                                                    num_objects=self.num_slots,
                                                    output_size=out_shape)

        print('Number of params in encoder ', util.count_params(self.encoder))
        print(f'Number of params in transition model ', util.count_params(self.transition_model))
        print('Number of params in decoder ', util.count_params(self.decoder))

        self.l2_loss = nn.MSELoss(reduction="mean")

    def get_pixel_loss(self, predicted, target):
        # if pixel_loss == 'bce':
            # rec_loss = F.binary_cross_entropy(rec, img1, reduction='sum') / obs.size(0) + \
            #            F.binary_cross_entropy(rec2, img2, reduction='sum') / obs.size(0)

        return self.l2_loss(predicted, target)

    def compose_image(self, model_out):
        """
        Compose the image using the output of the model forward pass.
        :param model_out: output of forward pass of model.
        :return: masks, masked image components and the full reconstruction.
        """
        comps, raw_masks = model_out[:, :3, :, :], model_out[:, 3, :, :]

        comps = comps.view(-1, self.num_slots, comps.size(1), comps.size(2), comps.size(3))
        raw_masks = raw_masks.view(-1, self.num_slots, raw_masks.size(1), raw_masks.size(2))
        masks = F.softmax(raw_masks, dim=1)

        masked_comps = torch.mul(masks.unsqueeze(2), comps)
        recs = masked_comps.sum(dim=1)

        return masks, masked_comps, recs

    def write_updates(self, writer, reconstructions, images_gt, masks, masked_comps,
                      iter, prefix="", num_img_pairs=3, num_input_pairs=10):
        """ Writes tensorboard summaries using tensorboardx api.
            num_img_pairs: Number of input image pairs to display.
        """
        w, h = images_gt.size(2), images_gt.size(3)
        batch_size = images_gt.size(0) // 2
        images_gt = torch.cat((images_gt[:batch_size].unsqueeze(1),
                               images_gt[batch_size:].unsqueeze(1)), dim=1)
        if num_img_pairs > batch_size:
            num_img_pairs = batch_size

        same_view_recs = reconstructions[:batch_size*2].reshape(2, batch_size, 3, w, h).transpose(0, 1)
        diff_view_recs = reconstructions[batch_size*2:].reshape(2, batch_size, 3, w, h).transpose(0, 1)

        # same_view_comps = comps[:batch_size*2].reshape(
        #     batch_size, 2, self.num_slots, 3, w, h)
        # diff_view_comps = comps[batch_size*2:].reshape(
        #     batch_size, 2, self.num_slots, 3, w, h)
        same_view_masked_comps = masked_comps[:batch_size*2].reshape(
            2, batch_size, self.num_slots, 3, w, h).transpose(0, 1)
        diff_view_masked_comps = masked_comps[batch_size*2:].reshape(
            2, batch_size, self.num_slots, 3, w, h).transpose(0, 1)
        same_view_masks = masks[batch_size*2:].reshape(2, batch_size, self.num_slots, w, h).transpose(0, 1)
        diff_view_masks = masks[batch_size*2:].reshape(2, batch_size, self.num_slots, w, h).transpose(0, 1)
        # Expand to have 3 channels so can concat with rgb images
        same_view_masks = same_view_masks.unsqueeze(3).repeat(1, 1, 1, 3, 1, 1)
        diff_view_masks = diff_view_masks.unsqueeze(3).repeat(1, 1, 1, 3, 1, 1)
        # Shift masks to be in range [-1, 1] like rgb
        same_view_masks = same_view_masks * 2 - 1
        diff_view_masks = diff_view_masks * 2 - 1

        # Input pairs to display
        input_disp = images_gt[:num_input_pairs].transpose(0, 1).flatten(end_dim=1)

        writer.add_image(prefix + "sample_image_pairs",
                         torchvision.utils.make_grid(input_disp,
                                                     nrow=num_img_pairs*2,
                                                     scale_each=False,
                                                     normalize=True,
                                                     range=(-1, 1)).cpu().detach().numpy(),
                         iter)

        # Choose random reconstructions to display
        disp_idx = sorted(np.random.choice(batch_size, num_img_pairs, replace=False))

        gt_disp = images_gt[disp_idx].transpose(0, 1).flatten(end_dim=1)
        same_view_disp = same_view_recs[disp_idx].transpose(0, 1).flatten(end_dim=1)
        diff_view_disp = diff_view_recs[disp_idx].transpose(0, 1).flatten(end_dim=1)

        gt_vs_rec_disp = torch.cat((gt_disp,
                                    same_view_disp,
                                    diff_view_disp), dim=0)

        writer.add_image(prefix + "gt_vs_same_view_recs_vs_diff_view_recs",
                         torchvision.utils.make_grid(gt_vs_rec_disp,
                                                     nrow=num_img_pairs*2,
                                                     scale_each=False,
                                                     normalize=True,
                                                     range=(-1, 1)).cpu().detach().numpy(),
                         iter)

        # Display masked components
        same_view_masked_comps = torch.cat((images_gt[disp_idx].unsqueeze(2),
                                            same_view_recs[disp_idx].unsqueeze(2),
                                            same_view_masked_comps[disp_idx],
                                            same_view_masks[disp_idx]),
                                           dim=2).transpose(0, 2)
        diff_view_masked_comps = torch.cat((images_gt[disp_idx].unsqueeze(2),
                                            diff_view_recs[disp_idx].unsqueeze(2),
                                            diff_view_masked_comps[disp_idx],
                                            diff_view_masks[disp_idx]),
                                           dim=2).transpose(0, 2)

        writer.add_image(prefix + "same_view_masked_components",
                         torchvision.utils.make_grid(same_view_masked_comps.reshape(-1, 3, h, w),
                                                     nrow=num_img_pairs*2,
                                                     scale_each=False,
                                                     normalize=True,
                                                     range=(-1, 1)).cpu().detach().numpy(),
                         iter)
        writer.add_image(prefix + "diff_view_masked_components",
                         torchvision.utils.make_grid(diff_view_masked_comps.reshape(-1, 3, h, w),
                                                     nrow=num_img_pairs*2,
                                                     scale_each=False,
                                                     normalize=True,
                                                     range=(-1, 1)).cpu().detach().numpy(),
                         iter)

    def forward(self, imgs, actions):
        """
        Foward pass of model.
        :param imgs: two sets of views concatenated [2*B, 3, h, w].
        :param actions: relative transformations to condition transition model.
        :return: Reconstructed images of samples.
        """
        batch_size = imgs.size(0) // 2
        # state, transf_state: [B*2, num_slots, embedding_dim]
        state = self.encoder(imgs)

        if self.identity_action_flag:
            duplicated_state = state.repeat(2, 1, 1)

            identity_action = torch.zeros_like(actions)
            identity_action[:, 0] = 1.
            identity_action[:, 5] = 1.
            identity_action[:, 10] = 1.

            actions = torch.cat((identity_action, actions), dim=0)
            transformed_state = self.transition_model(duplicated_state, actions)
            # duplicated_state: [B*4, num_slots, embedding_dim]

            # Swap order of transformed states so transformed images are reconstructed
            # in same order as input images.
            transformed_state = torch.cat((transformed_state[:batch_size*2],
                                           transformed_state[batch_size*3:],
                                           transformed_state[batch_size*2:batch_size*3]), dim=0)
        else:
            if self.canonical:
                # Actions are poses. First transition to canonical state with pose of image
                # Then transition to novel view state
                poses = actions
                novel_poses = torch.cat((actions[:batch_size], actions[batch_size:]), dim=0)
                transformed_state = self.transition_model(self.transition_model(state, novel_poses), poses)
            else:
                # Actions are relative poses
                transformed_state = self.transition_model(state, actions)
            transformed_state = torch.cat((state,
                                           transformed_state[batch_size:],
                                           transformed_state[:batch_size]), dim=0)

        # Decode embedding
        out = self.decoder(transformed_state.reshape(-1, self.embedding_dim))
        return out

