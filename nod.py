import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt

import modules

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
                 num_slots, encoder='cswm', decoder='broadcast'):
        super(NodModel, self).__init__()
        self.embedding_dim = embedding_dim
        self.input_dims = input_dims
        self.hidden_dims = hidden_dim

        self.num_slots = num_slots

        if encoder == 'cswm':
            self.encoder = modules.EncoderCSWM(input_dims=self.input_dims,
                                               embedding_dim=self.embedding_dim,
                                               num_objects=self.num_slots)

        self.transition_model = modules.TransitionGNN(input_dim=self.embedding_dim,
                                                      hidden_dim=512,
                                                      action_dim=12,
                                                      num_objects=self.num_slots,
                                                      residual=True)

        if decoder == 'broadcast':
            self.decoder = modules.BroadcastDecoder(latent_dim=self.embedding_dim,
                                                    output_dim=4,  # 3 rgb channels and one mask
                                                    hidden_channels=32,
                                                    num_layers=4,
                                                    img_dims=[128, 128],  # width and height of square image
                                                    act_fn='elu')
        if decoder == 'cnn':
            self.decoder = modules.DecoderCNNMedium(input_dim=self.embedding_dim,
                                                    hidden_dim=32,
                                                    num_objects=self.num_slots,
                                                    output_size=[4, 128, 128])

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
        images_gt = images_gt.reshape(-1, 2, 3, w, h)
        batch_size = images_gt.size(0)
        if num_img_pairs > batch_size:
            num_img_pairs = batch_size

        same_view_recs = reconstructions[:batch_size*2].reshape(batch_size, 2, 3, w, h)
        diff_view_recs = reconstructions[batch_size*2:].reshape(batch_size, 2, 3, w, h)

        # same_view_comps = comps[:batch_size*2].reshape(
        #     batch_size, 2, self.num_slots, 3, w, h)
        # diff_view_comps = comps[batch_size*2:].reshape(
        #     batch_size, 2, self.num_slots, 3, w, h)
        same_view_masked_comps = masked_comps[:batch_size*2].reshape(
            batch_size, 2, self.num_slots, 3, w, h)
        diff_view_masked_comps = masked_comps[batch_size*2:].reshape(
            batch_size, 2, self.num_slots, 3, w, h)
        same_view_masks = masks[batch_size*2:].reshape(batch_size, 2, self.num_slots, w, h)
        diff_view_masks = masks[batch_size*2:].reshape(batch_size, 2, self.num_slots, w, h)
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
        :param imgs: two sets of views concatenated [2*B, 3, 128, 128].
        :param actions: relative transformations to condition transition model.
        :return: Reconstructed images of samples.
        """
        batch_size = imgs.size(0) // 2
        # state, transf_state: [B*2, num_slots, embedding_dim]
        state = self.encoder(imgs)
        duplicated_state = state.repeat(2, 1, 1)

        actions = torch.cat((torch.zeros_like(actions), actions), dim=0)
        transformed_state = self.transition_model(duplicated_state, actions)
        # duplicated_state: [B*4, num_slots, embedding_dim]

        # Swap order of transformed states so transformed images are reconstructed
        # in same order as input images.
        transformed_state = torch.cat((transformed_state[:batch_size*2],
                                       transformed_state[batch_size*3:],
                                       transformed_state[batch_size*2:batch_size*3]), dim=0)

        # Decode embedding
        out = self.decoder(transformed_state.reshape(-1, self.embedding_dim))
        return out

