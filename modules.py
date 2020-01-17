import util

import numpy as np

import torch
from torch import nn
import torch.nn.functional as F


class EncoderCSWM(nn.Module):
    """Encoder for C-SWM.

    Args:
        embedding_dim: Dimensionality of latent state space.
        input_dims: Shape of input observation.
        hidden_dim: Number of hidden units in encoder and transition model.
        num_objects: Number of object slots.
    """

    def __init__(self, input_dims, embedding_dim, num_objects,
                 obj_extractor_hidden_dim=32, obj_encoder_hidden_dim=512):
        super(EncoderCSWM, self).__init__()

        self.embedding_dim = embedding_dim
        self.num_objects = num_objects

        num_channels = input_dims[0]
        width_height = input_dims[1:]

        self.obj_extractor = EncoderCNNLarge(
            input_dim=num_channels,
            hidden_dim=obj_extractor_hidden_dim,
            num_objects=num_objects)

        self.obj_encoder = EncoderMLP(
            input_dim=np.prod(width_height),
            hidden_dim=obj_encoder_hidden_dim,
            output_dim=embedding_dim,
            num_objects=num_objects)

        self.width = width_height[0]
        self.height = width_height[1]

    def forward(self, obs):
        return self.obj_encoder(self.obj_extractor(obs))


class TransitionGNN(torch.nn.Module):
    """GNN-based transition function."""
    def __init__(self, input_dim, hidden_dim, action_dim, num_objects,
                 act_fn='relu'):
        super(TransitionGNN, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_objects = num_objects
        self.action_dim = action_dim

        self.edge_mlp = nn.Sequential(
            nn.Linear(input_dim*2, hidden_dim),
            util.get_act_fn(act_fn),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            util.get_act_fn(act_fn),
            nn.Linear(hidden_dim, hidden_dim))

        node_input_dim = hidden_dim + input_dim + self.action_dim

        self.node_mlp = nn.Sequential(
            nn.Linear(node_input_dim, hidden_dim),
            util.get_act_fn(act_fn),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            util.get_act_fn(act_fn),
            nn.Linear(hidden_dim, input_dim))

        self.edge_list = None
        self.batch_size = 0

    def _edge_model(self, source, target, edge_attr):
        """
        Computes edge features. Not symmetric function wrt input nodes.
        :param source: Source node.
        :param target: Adjacent node for which we want to calculate the edge feature.
        :param edge_attr: Unused.
        :return:
        """
        del edge_attr  # Unused.
        out = torch.cat([source, target], dim=1)
        return self.edge_mlp(out)

    def _node_model(self, node_attr, edge_index, edge_attr):
        if edge_attr is not None:
            row, col = edge_index
            agg = util.unsorted_segment_sum(
                edge_attr, row, num_segments=node_attr.size(0))
            out = torch.cat([node_attr, agg], dim=1)
        else:
            out = node_attr
        return self.node_mlp(out)

    def _get_edge_list_fully_connected(self, batch_size, num_objects, cuda):
        # Only re-evaluate if necessary (e.g. if batch size changed).
        if self.edge_list is None or self.batch_size != batch_size:
            self.batch_size = batch_size

            # Create fully-connected adjacency matrix for single sample.
            adj_full = torch.ones(num_objects, num_objects)

            # Remove diagonal.
            adj_full -= torch.eye(num_objects)
            self.edge_list = adj_full.nonzero()

            # Copy `batch_size` times and add offset.
            self.edge_list = self.edge_list.repeat(batch_size, 1)
            offset = torch.arange(
                0, batch_size * num_objects, num_objects).unsqueeze(-1)
            offset = offset.expand(batch_size, num_objects * (num_objects - 1))
            offset = offset.contiguous().view(-1)
            self.edge_list += offset.unsqueeze(-1)

            # Transpose to COO format -> Shape: [2, num_edges].
            self.edge_list = self.edge_list.transpose(0, 1)

            if cuda:
                self.edge_list = self.edge_list.cuda()

        return self.edge_list

    def forward(self, states, action):

        cuda = states.is_cuda
        batch_size = states.size(0)
        num_nodes = states.size(1)

        # states: [batch_size (B), num_objects, embedding_dim]
        # node_attr: Flatten states tensor to [B * num_objects, embedding_dim]
        node_attr = states.view(-1, self.input_dim)

        edge_attr = None
        edge_index = None

        # Compute edge attributes
        if num_nodes > 1:
            # edge_index: [B * (num_objects*[num_objects-1]), 2] edge list
            edge_index = self._get_edge_list_fully_connected(
                batch_size, num_nodes, cuda)

            # print('edge index shape \n', edge_index.shape)

            row, col = edge_index
            # print('row col \n', row, col)
            edge_attr = self._edge_model(
                node_attr[row], node_attr[col], edge_attr)


        # Same action on all nodes
        action_vec = action.repeat(1, self.num_objects)
        action_vec = action_vec.view(-1, self.action_dim)

        # Attach action to each state
        node_attr = torch.cat([node_attr, action_vec], dim=-1)

        node_attr = self._node_model(
            node_attr, edge_index, edge_attr)

        # [batch_size, num_nodes, hidden_dim]
        return node_attr.view(batch_size, num_nodes, -1)


class EncoderCNNSmall(nn.Module):
    """CNN encoder, maps observation to obj-specific feature maps."""

    def __init__(self, input_dim, hidden_dim, num_objects, act_fn='sigmoid',
                 act_fn_hid='relu'):
        super(EncoderCNNSmall, self).__init__()
        self.cnn1 = nn.Conv2d(
            input_dim, hidden_dim, (10, 10), stride=10)
        self.cnn2 = nn.Conv2d(hidden_dim, num_objects, (1, 1), stride=1)
        self.ln1 = nn.BatchNorm2d(hidden_dim)
        self.act1 = util.get_act_fn(act_fn_hid)
        self.act2 = util.get_act_fn(act_fn)

    def forward(self, obs):
        h = self.act1(self.ln1(self.cnn1(obs)))
        return self.act2(self.cnn2(h))


class EncoderCNNMedium(nn.Module):
    """CNN encoder, maps observation to obj-specific feature maps."""

    def __init__(self, input_dim, hidden_dim, num_objects, act_fn='sigmoid',
                 act_fn_hid='leaky_relu'):
        super(EncoderCNNMedium, self).__init__()

        self.cnn1 = nn.Conv2d(
            input_dim, hidden_dim, (9, 9), padding=4)
        self.act1 = util.get_act_fn(act_fn_hid)
        self.ln1 = nn.BatchNorm2d(hidden_dim)

        self.cnn2 = nn.Conv2d(
            hidden_dim, num_objects, (5, 5), stride=5)
        self.act2 = util.get_act_fn(act_fn)

    def forward(self, obs):
        h = self.act1(self.ln1(self.cnn1(obs)))
        h = self.act2(self.cnn2(h))
        return h


class EncoderCNNLarge(nn.Module):
    """CNN encoder, maps observation to obj-specific feature maps."""

    def __init__(self, input_dim, hidden_dim, num_objects, act_fn='sigmoid',
                 act_fn_hid='relu'):
        super(EncoderCNNLarge, self).__init__()

        self.cnn1 = nn.Conv2d(input_dim, hidden_dim, (3, 3), padding=1)
        self.act1 = util.get_act_fn(act_fn_hid)
        self.ln1 = nn.BatchNorm2d(hidden_dim)

        self.cnn2 = nn.Conv2d(hidden_dim, hidden_dim, (3, 3), padding=1)
        self.act2 = util.get_act_fn(act_fn_hid)
        self.ln2 = nn.BatchNorm2d(hidden_dim)

        self.cnn3 = nn.Conv2d(hidden_dim, hidden_dim, (3, 3), padding=1)
        self.act3 = util.get_act_fn(act_fn_hid)
        self.ln3 = nn.BatchNorm2d(hidden_dim)

        self.cnn4 = nn.Conv2d(hidden_dim, num_objects, (3, 3), padding=1)
        self.act4 = util.get_act_fn(act_fn)

    def forward(self, obs):
        h = self.act1(self.ln1(self.cnn1(obs)))
        h = self.act2(self.ln2(self.cnn2(h)))
        h = self.act3(self.ln3(self.cnn3(h)))
        return self.act4(self.cnn4(h))


class EncoderMLP(nn.Module):
    """MLP encoder, maps observation to latent state."""

    def __init__(self, input_dim, output_dim, hidden_dim, num_objects,
                 act_fn='relu'):
        super(EncoderMLP, self).__init__()

        self.num_objects = num_objects
        self.input_dim = input_dim

        self.fc1 = nn.Linear(self.input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

        self.ln = nn.LayerNorm(hidden_dim)

        self.act1 = util.get_act_fn(act_fn)
        self.act2 = util.get_act_fn(act_fn)

    def forward(self, ins):
        h_flat = ins.view(-1, self.num_objects, self.input_dim)
        h = self.act1(self.fc1(h_flat))
        h = self.act2(self.ln(self.fc2(h)))
        return self.fc3(h)


class DecoderCNNMedium(nn.Module):
    """CNN decoder, maps latent state to image."""

    def __init__(self, input_dim, hidden_dim, num_objects, output_size,
                 act_fn='relu'):
        super(DecoderCNNMedium, self).__init__()

        width, height = output_size[1] // 5, output_size[2] // 5

        output_dim = width * height

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.ln = nn.LayerNorm(hidden_dim)

        self.deconv1 = nn.ConvTranspose2d(num_objects, hidden_dim,
                                          kernel_size=5, stride=5)
        self.deconv2 = nn.ConvTranspose2d(hidden_dim, output_size[0],
                                          kernel_size=9, padding=4)

        self.ln1 = nn.BatchNorm2d(hidden_dim)

        self.input_dim = input_dim
        self.num_objects = num_objects
        self.map_size = output_size[0], width, height

        self.act1 = util.get_act_fn(act_fn)
        self.act2 = util.get_act_fn(act_fn)
        self.act3 = util.get_act_fn(act_fn)

    def forward(self, ins):
        h = self.act1(self.fc1(ins))
        h = self.act2(self.ln(self.fc2(h)))
        h = self.fc3(h)

        h_conv = h.view(-1, self.num_objects, self.map_size[1],
                        self.map_size[2])
        h = self.act3(self.ln1(self.deconv1(h_conv)))
        out = self.deconv2(h)
        out = out.unsqueeze(1).expand(-1, self.num_objects, -1, -1, -1)
        out = out.reshape(-1, out.size(2), out.size(3), out.size(4))
        return out


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
        super(BroadcastDecoder, self).__init__()
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


