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

        # self.transition_model = modules.TransitionGNN()

        if decoder == 'broadcast':
            self.decoder = modules.BroadcastDecoder(latent_dim=self.embedding_dim,
                                                    output_dim=4,  # 3 rgb channels and one mask
                                                    hidden_channels=32,
                                                    num_layers=4,
                                                    img_dim=128,  # width and height of square image
                                                    act_fn='elu')

        self.l2_loss = nn.MSELoss(reduction="mean")

    def get_pixel_loss(self, predicted, target):
        # if pixel_loss == 'bce':
            # rec_loss = F.binary_cross_entropy(rec, img1, reduction='sum') / obs.size(0) + \
            #            F.binary_cross_entropy(rec2, img2, reduction='sum') / obs.size(0)

        return self.l2_loss(predicted, target)

    @staticmethod
    def write_updates(writer, reconstructions, samples, iter, prefix="", num_imgs=5):
        """Writes tensorboard summaries using tensorboardx api. """

        batchsize = samples['image1'].size(0)
        if num_imgs > batchsize:
            num_imgs = batchsize

        if not iter % 100:
            rec1 = reconstructions[:num_imgs]
            rec2 = reconstructions[batchsize:batchsize+num_imgs]

            view1 = samples['image1'][:num_imgs].cuda()
            view2 = samples['image2'][:num_imgs].cuda()

            sample_input = torch.cat((view1, view2), dim=0)
            rec1_vs_imgs1 = torch.cat((rec1, view1), dim=0)
            rec2_vs_imgs2 = torch.cat((rec2, view2), dim=0)

            writer.add_image(prefix + "samples_from_recent_batch",
                             torchvision.utils.make_grid(sample_input,
                                                         nrow=num_imgs,
                                                         scale_each=False,
                                                         normalize=True).cpu().detach().numpy(),
                             iter)

            writer.add_image(prefix + "view1_reconstruction",
                             torchvision.utils.make_grid(rec1_vs_imgs1,
                                                         nrow=num_imgs,
                                                         scale_each=False,
                                                         normalize=True).cpu().detach().numpy(),
                             iter)

            writer.add_image(prefix + "view2_reconstruction",
                             torchvision.utils.make_grid(rec2_vs_imgs2,
                                                         nrow=num_imgs,
                                                         scale_each=False,
                                                         normalize=True).cpu().detach().numpy(),
                             iter)

    def forward(self, input):
        """
        Foward pass of model.
        :param input: two sets of views concatenated.
        :return: Reconstructed images of samples.
        """

        # state has shape (batchsize, num_slots, embedding_dim)
        state = self.encoder(input)

        # Decode embedding
        flat_state = state.reshape(-1, state.size(2))
        comps, masks = self.decoder(flat_state)

        comps = comps.view(-1, self.num_slots, comps.size(1), comps.size(2), comps.size(3))
        masks = masks.view(-1, self.num_slots, masks.size(1), masks.size(2))
        scaled_masks = F.softmax(masks, dim=1)

        masked_comps = torch.mul(scaled_masks.unsqueeze(2), comps)
        recs = masked_comps.sum(dim=1)

        return recs

