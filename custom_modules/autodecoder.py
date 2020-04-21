import util

import torch
from torch import nn
import torch.nn.functional as F
import torchvision

import modules
from custom_modules import spd


class StructuredAutoDecoder(nn.Module):
    """Structured auto decoder

    Args:
        embedding_dim: Dimensionality of latent state space.
        input_dims: Shape of input observation.
        hidden_dim: Number of hidden units in encoder and transition model.
        n_slots: Number of object slots.
    """

    def __init__(self, img_size, embedding_dim, n_slots,
                 action_dim, residual=True):
        super(StructuredAutoDecoder, self).__init__()

        self.img_size = img_size
        self.embedding_dim = embedding_dim
        self.n_slots = n_slots

        # Auto-decoder: each scene component gets its own code vector z
        self.latent_codes = nn.Embedding(n_slots, embedding_dim).cuda()
        nn.init.normal_(self.latent_codes.weight, mean=0, std=0.01)

        self.transition_model = modules.TransitionGNN(input_dim=embedding_dim,
                                                      hidden_dim=512,
                                                      action_dim=action_dim,
                                                      num_objects=n_slots,
                                                      residual=residual)

        self.comp_decoder = spd.BroadcastDecoder(latent_dim=embedding_dim,
                                                 output_dim=4,  # 3 rgb channels and one mask
                                                 hidden_channels=32,
                                                 num_layers=4,
                                                 img_dims=img_size,
                                                 act_fn='elu')

    @staticmethod
    def write_updates(writer, reconstructions, images_gt, masks, masked_comps,
                      iter, prefix="", n_images=5):
        """ Writes tensorboard summaries using tensorboardx api.
            num_images: Number of reconstructed images to display.
        """
        batch_size = images_gt.size(0)

        if n_images > batch_size:
            n_images = batch_size

        masks = masks.unsqueeze(2).repeat(1, 1, 3, 1, 1)
        masks = masks * 2 - 1

        gt_vs_recs = torch.cat((images_gt[:n_images].unsqueeze(1),
                                reconstructions[:n_images].unsqueeze(1),
                                masked_comps[:n_images],
                                masks[:n_images]), dim=1).transpose(0, 1)
        writer.add_image(prefix + "gt_vs_reconstructions",
                         torchvision.utils.make_grid(gt_vs_recs.flatten(end_dim=1),
                                                     nrow=n_images,
                                                     scale_each=False,
                                                     normalize=True,
                                                     range=(-1, 1)).cpu().detach().numpy(),
                         iter)


    def compose_image(self, model_out):
        """
        Compose the image using the output of the model forward pass.

        :param model_out: output of forward pass of model.
        :return: masks, masked image components and the full reconstruction.

        masks: [n_views, n_slots, h, w]
        masked_comps: [n_views, n_slots, 3, h, w]
        recs: [n_views, 3, h, w]
        """
        comps, raw_masks = model_out[:, :3, :, :], model_out[:, 3, :, :]

        comps = comps.view(-1, self.n_slots, comps.size(1), comps.size(2), comps.size(3))
        raw_masks = raw_masks.view(-1, self.n_slots, raw_masks.size(1), raw_masks.size(2))
        masks = F.softmax(raw_masks, dim=1)

        masked_comps = torch.mul(masks.unsqueeze(2), comps)
        recs = masked_comps.sum(dim=1)

        return masks, masked_comps, recs

    def forward(self, actions, z=None):
        """

        :param z: [n_slots, embedding_dim]
        :param actions: [n_views, action_dim]
        :return: [n_views, 3, w, h]
        """
        # If z is None then using autodecoder training
        if z is None:
            state = self.latent_codes.weight
        else:
            state = z

        n_views = actions.size(0)

        # state: [n_views, n_slots, embedding_dim]
        state = state.unsqueeze(0).expand(n_views, self.n_slots, self.embedding_dim)

        view_dep_states = self.transition_model(state, actions)

        out = self.comp_decoder(view_dep_states.view(-1, self.embedding_dim))

        return out
