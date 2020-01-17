import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F

import numpy as np

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
                                                      num_objects=self.num_slots)

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

    @staticmethod
    def write_updates(writer, reconstructions, imgs, iter, prefix="", num_imgs=5):
        """Writes tensorboard summaries using tensorboardx api. """

        batch_size = imgs.size(0) // 2
        if num_imgs > batch_size:
            num_imgs = batch_size

        disp_idx = np.random.choice(batch_size*2, num_imgs, replace=False)

        same_view_recs = reconstructions[disp_idx]
        diff_view_recs = reconstructions[batch_size + disp_idx]

        ground_truth_imgs = imgs[disp_idx]

        sample_input = torch.cat((imgs[:num_imgs],
                                  imgs[batch_size:batch_size + num_imgs]), dim=0)
        same_view_recs_vs_gt = torch.cat((same_view_recs, ground_truth_imgs), dim=0)
        diff_view_recs_vs_gt = torch.cat((diff_view_recs, ground_truth_imgs), dim=0)

        writer.add_image(prefix + "samples_from_recent_batch",
                         torchvision.utils.make_grid(sample_input,
                                                     nrow=num_imgs,
                                                     scale_each=False,
                                                     normalize=True).cpu().detach().numpy(),
                         iter)

        writer.add_image(prefix + "same_view_recs_vs_gt",
                         torchvision.utils.make_grid(same_view_recs_vs_gt,
                                                     nrow=num_imgs,
                                                     scale_each=False,
                                                     normalize=True).cpu().detach().numpy(),
                         iter)

        writer.add_image(prefix + "diff_view_recs_vs_gt",
                         torchvision.utils.make_grid(diff_view_recs_vs_gt,
                                                     nrow=num_imgs,
                                                     scale_each=False,
                                                     normalize=True).cpu().detach().numpy(),
                         iter)

    def forward(self, imgs, actions):
        """
        Foward pass of model.
        :param imgs: two sets of views concatenated.
        :param actions: relative transformations to condition transition model.
        :return: Reconstructed images of samples.
        """
        batch_size = imgs.size(0) // 2
        # state: [B*2, num_slots, embedding_dim]
        state = self.encoder(imgs)

        transf_state = self.transition_model(state, actions)

        # state: [B*4, num_slots, embedding_dim]
        repeated_states = torch.cat((state,
                                     transf_state[:batch_size],
                                     transf_state[batch_size:]))

        # Decode embedding
        flat_state = repeated_states.reshape(-1, repeated_states.size(2))
        out = self.decoder(flat_state)

        comps, masks = out[:, :3, :, :], out[:, 3, :, :]

        comps = comps.view(-1, self.num_slots, comps.size(1), comps.size(2), comps.size(3))
        masks = masks.view(-1, self.num_slots, masks.size(1), masks.size(2))
        scaled_masks = F.softmax(masks, dim=1)

        masked_comps = torch.mul(scaled_masks.unsqueeze(2), comps)
        recs = masked_comps.sum(dim=1)

        return recs

