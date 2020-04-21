import util

import torch
from torch import nn
import torch.nn.functional as F


class BroadcastDecoder(nn.Module):
    def __init__(self, latent_dim, output_dim, hidden_channels,
                 num_layers, img_dims, act_fn='elu'):
        super(BroadcastDecoder, self).__init__()

        self.latent_dim = latent_dim
        self.width = img_dims[0]
        self.height = img_dims[1]
        self.num_layers = num_layers

        mods = [nn.Conv2d(latent_dim + 2, hidden_channels, 3),
                util.get_act_fn(act_fn)]

        for _ in range(num_layers - 1):
            mods.extend([nn.Conv2d(hidden_channels, hidden_channels, 3), util.get_act_fn(act_fn)])

        # 1x1 conv for output layer
        mods.append(nn.Conv2d(hidden_channels, output_dim, 1))
        self.seq = nn.Sequential(*mods)

    def sb(self, ins):
        """ Broadcast z spatially across image size and
            append x and y coordinates. """
        batch_size = ins.size(0)

        # Expand spatially: (n, z_dim) -> (n, z_dim, h, w)
        z_b = ins.view((batch_size, -1, 1, 1)).expand(-1, -1,
                                                      self.width + 2*self.num_layers,
                                                      self.height + 2*self.num_layers)

        # Coordinate axes:
        x = torch.linspace(-1, 1, self.width + 2*self.num_layers, device=ins.device)
        y = torch.linspace(-1, 1, self.height + 2*self.num_layers, device=ins.device)
        y_b, x_b = torch.meshgrid(y, x)
        # Expand from (h, w) -> (n, 1, h, w)
        x_b = x_b.expand(batch_size, 1, -1, -1)
        y_b = y_b.expand(batch_size, 1, -1, -1)
        # Concatenate along the channel dimension: final shape = (n, z_dim + 2, h, w)
        z_sb = torch.cat((z_b, x_b, y_b), dim=1)
        return z_sb

    def forward(self, ins):
        z_sb = self.sb(ins)
        return self.seq(z_sb)


"""
Implementation adapted from GENESIS (https://github.com/applied-ai-lab/genesis)
"""


class PixelCoords(nn.Module):
    def __init__(self, im_dim):
        super(PixelCoords, self).__init__()
        g_1, g_2 = torch.meshgrid(torch.linspace(-1, 1, im_dim),
                                  torch.linspace(-1, 1, im_dim))
        # Adds 2 extra dims to g_1
        self.register_buffer('g_1', g_1.view((1, 1) + g_1.shape))
        self.register_buffer('g_2', g_2.view((1, 1) + g_2.shape))

    def forward(self, x):
        # Expand first dim to batch size
        g_1 = self.g_1.expand(x.size(0), -1, -1, -1)
        g_2 = self.g_2.expand(x.size(0), -1, -1, -1)
        return torch.cat((x, g_1, g_2), dim=1)


class BroadcastLayer(nn.Module):
    def __init__(self, dim):
        super(BroadcastLayer, self).__init__()
        self.dim = dim
        self.pixel_coords = PixelCoords(dim)

    def forward(self, x):
        batch_size = x.size(0)
        # Broadcast
        if x.dim() == 2:
            x = x.view(batch_size, -1, 1, 1)
            x = x.expand(-1, -1, self.dim, self.dim)
        else:
            x = F.interpolate(x, self.dim)

        return self.pixel_coords(x)


class GenesisBroadcastDecoder(nn.Module):
    """ Broadcast decoder adapted from genesis code. """

    def __init__(self, latent_dim, output_dim, hidden_channels,
                 num_layers, img_dim, act_fn='elu'):
        super(GenesisBroadcastDecoder, self).__init__()
        broad_dim = img_dim + 2*num_layers

        mods = [BroadcastLayer(broad_dim),
                nn.Conv2d(latent_dim + 2, hidden_channels, 3),
                util.get_act_fn(act_fn)]

        for _ in range(num_layers - 1):
            mods.extend([nn.Conv2d(hidden_channels, hidden_channels, 3), util.get_act_fn(act_fn)])

        # 1x1 conv for output layer
        mods.append(nn.Conv2d(hidden_channels, output_dim, 1))
        self.seq = nn.Sequential(*mods)

    def forward(self, x):
        # Input should have shape (batch size, latent_dim)
        return self.seq(x)


