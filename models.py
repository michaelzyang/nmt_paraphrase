import numpy as np
import torch.nn as nn


class Resnet(nn.Module):
    raise NotImplementedError
    # def __init__(self, in_channels, block_channels, layer_blocks, kernel_sizes, strides, pool_size, num_classes):
    #     """
    #     Adapted from Recitation 6 code
    #     :param in_channels: the number of channels in the input data
    #     :param block_channels: the number of channels in each layer
    #     :param layer_blocks: the number of consecutive blocks in each layer
    #     :param strides: stride at the end of each layer
    #     :param num_classes:
    #     :param feat_dim:
    #     """
    #     super(Resnet, self).__init__()
    #     assert len(block_channels) == len(layer_blocks), \
    #         f"# block channels {block_channels} needs to equal # layer_blocks {layer_blocks}."
    #
    #     # Initial layers
    #     self.layers = []
    #     conv1 = nn.Conv2d(in_channels, block_channels[0], kernel_size=3, stride=1, padding=1, bias=False)
    #     self.layers.append(conv1)
    #     self.layers.append(nn.BatchNorm2d(block_channels[0]))
    #     self.layers.append(nn.ReLU(inplace=True))
    #
    #     # Residual block layers
    #     for i in range(len(block_channels)):
    #         num_blocks = layer_blocks[i]
    #         in_block_channels = block_channels[i] if i == 0 else block_channels[i - 1]
    #         block_layer = self._block_layer(in_block_channels, block_channels[i], kernel_sizes[i], strides[i],
    #                                         num_blocks)
    #         self.layers.append(block_layer)
    #
    #     self.net = nn.Sequential(*self.layers)
    #
    #     # pooling layer
    #     self.avg_pool = nn.AvgPool2d(pool_size) # (block_channels[-1], 16/pool, 16/pool)
    #
    #     # linear output layer
    #     pooled_feature_map_size = (32 // np.product(strides) // pool_size) ** 2
    #     self.linear_label = nn.Linear(block_channels[-1] * pooled_feature_map_size, num_classes, bias=False)
    #
    #
    # def forward(self, x, evalMode=False):
    #     embedding = self.net(x)
    #
    #     output = self.avg_pool(embedding)
    #     output = output.reshape(output.shape[0], -1)
    #
    #     label_output = self.linear_label(output)
    #     # label_output = label_output/torch.norm(self.linear_label.weight, dim=1)
    #
    #     return output, label_output
    #
    # def _block_layer(self, in_channels, block_channels, kernel_size, stride, num_blocks):
    #     assert num_blocks >= 2, f"At least 2 blocks per layer required; {num_blocks} given."
    #
    #     block_layer = []
    #     # first block
    #     block_layer.append(
    #         BasicBlock(in_channels, block_channels, kernel_size, stride=1)
    #     )
    #     # intermediate blocks
    #     for _ in range(num_blocks - 2):
    #         block_layer.append(
    #             BasicBlock(block_channels, block_channels, kernel_size, stride=1)
    #         )
    #     # downsample if necessary by striding
    #     block_layer.append(
    #         BasicBlock(block_channels, block_channels, kernel_size, stride=stride)
    #     )
    #
    #     return nn.Sequential(*block_layer)


class BasicBlock(nn.Module):
    raise NotImplementedError
    # def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
    #     super(BasicBlock, self).__init__()
    #     padding = int(kernel_size // 2)  # preserve image size
    #     self.reshape = stride > 1 or in_channels != out_channels  # whether x needs to be reshaped before adding
    #
    #     self.straight = nn.Sequential(
    #         nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
    #         nn.BatchNorm2d(out_channels),
    #         nn.ReLU(),
    #         nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding, bias=False),
    #         nn.BatchNorm2d(out_channels),
    #     )
    #
    #     if self.reshape:
    #         # self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=False),
    #         self.shortcut = nn.Sequential(
    #             nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=False),
    #             nn.BatchNorm2d(out_channels)
    #         )
    #     else:
    #         self.shortcut = nn.Identity()
    #
    #     self.relu = nn.ReLU()
    #
    # def forward(self, x):
    #     out = self.straight(x) + self.shortcut(x)  # add residual
    #     out = self.relu(out)
    #     return out


def init_weights(m):
    raise NotImplementedError
    # if type(m) == nn.Conv2d or type(m) == nn.Linear:
    #     nn.init.kaiming_uniform_(m.weight.data, nonlinearity='relu')
    #     # nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')