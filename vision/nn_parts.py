import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialSoftmax(nn.Module):
    """
    Credit to: https://gist.github.com/jeasinema/1cba9b40451236ba2cfb507687e08834 (08.11.22)
    """
    def __init__(self, height, width, channel, temperature=None, data_format='NCHW'):
        super(SpatialSoftmax, self).__init__()
        self.data_format = data_format
        self.height = height
        self.width = width
        self.channel = channel

        if temperature:
            self.temperature = nn.Parameter(torch.ones(1) * temperature)
        else:
            self.temperature = 1.

        pos_x, pos_y = np.meshgrid(np.linspace(-1., 1., self.height), np.linspace(-1., 1., self.width))
        pos_x = torch.from_numpy(pos_x.reshape(self.height * self.width)).float()
        pos_y = torch.from_numpy(pos_y.reshape(self.height * self.width)).float()
        self.register_buffer('pos_x', pos_x)
        self.register_buffer('pos_y', pos_y)

    def forward(self, feature):
        # Output:
        #   (N, C*2) x_0 y_0 ...
        if self.data_format == 'NHWC':
            feature = feature.transpose(1, 3).tranpose(2, 3).view(-1, self.height * self.width)
        else:
            feature = feature.reshape(-1, self.height * self.width)

        softmax_attention = F.softmax(feature / self.temperature, dim=-1)
        expected_x = torch.sum(self.pos_x * softmax_attention, dim=1, keepdim=True)
        expected_y = torch.sum(self.pos_y * softmax_attention, dim=1, keepdim=True)
        expected_xy = torch.cat([expected_x, expected_y], 1)
        feature_keypoints = expected_xy.view(-1, self.channel * 2)

        return feature_keypoints


class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, no_residual_blocks: int = 1):
        super().__init__()

        layers = [nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(out_c), nn.ReLU()]
        self.relu = nn.ReLU()
        for _ in range(no_residual_blocks - 1):
            layers.extend(
                [nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, bias=False),
                 nn.BatchNorm2d(out_c),
                 nn.ReLU()])

        self.multi_layer = nn.Sequential(*layers)

        # self.bn_end = nn.BatchNorm2d(out_c)

        if in_c == out_c:
            self.identity_process = nn.Identity()
        else:
            self.identity_process = nn.Sequential(nn.Conv2d(in_c, out_c, kernel_size=1, padding=0, bias=False),
                                                  nn.BatchNorm2d(out_c))

    def forward(self, inputs):
        identity = inputs.clone()

        x = self.multi_layer(inputs)

        x = x + self.identity_process(identity)
        x = self.relu(x)
        # x = self.bn_end(x)
        return x


class EncoderBlock(nn.Module):
    def __init__(self, in_c, out_c, no_residual_blocks: int, padding: int = 0):
        super().__init__()
        self.conv = ConvBlock(in_c, out_c, no_residual_blocks=no_residual_blocks)
        self.down = nn.MaxPool2d((2, 2), padding=padding)

    def forward(self, inputs):
        x = self.conv(inputs)
        p = self.down(x)

        return p


class DecoderBlock(nn.Module):
    def __init__(self, in_c, out_c, no_residual_blocks: int = 1, kernel_size: int = 2):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_c, out_c, kernel_size=kernel_size, stride=2, padding=0)
        self.conv = ConvBlock(out_c, out_c, no_residual_blocks)

    def forward(self, inputs):
        x = self.up(inputs)
        x = self.conv(x)
        return x


class Encoder(nn.Module):
    def __init__(self, image_dim: int) -> None:
        super().__init__()
        spatial_softmax_dim = 4 if image_dim == 128 else 7
        self.encoder = nn.Sequential(
            EncoderBlock(3, 8, no_residual_blocks=4),
            ConvBlock(8, 8, no_residual_blocks=4),
            EncoderBlock(8, 16, no_residual_blocks=4),
            ConvBlock(16, 16, no_residual_blocks=4),
            EncoderBlock(16, 32, no_residual_blocks=4),
            ConvBlock(32, 32, no_residual_blocks=4),
            EncoderBlock(32, 48, no_residual_blocks=4),
            ConvBlock(48, 48, no_residual_blocks=4),
            EncoderBlock(48, 64, no_residual_blocks=4),
            SpatialSoftmax(spatial_softmax_dim, spatial_softmax_dim, 64),
        )

    def forward(self, x):
        return self.encoder(x)


class Decoder(nn.Module):
    def __init__(self, image_dim: int, output_channels: int) -> None:
        super().__init__()
        kernel_size = 2 if image_dim == 128 else 3
        self.decoder = nn.Sequential(
            DecoderBlock(128, 48, no_residual_blocks=4, kernel_size=kernel_size),
            DecoderBlock(48, 32, no_residual_blocks=4, kernel_size=kernel_size),
            DecoderBlock(32, 24, no_residual_blocks=4),
            DecoderBlock(24, 16, no_residual_blocks=4),
            DecoderBlock(16, 14, no_residual_blocks=4),
            DecoderBlock(14, 12, no_residual_blocks=4),
            DecoderBlock(12, 10, no_residual_blocks=4),
            nn.Conv2d(10, output_channels, kernel_size=1, padding=0),
        )

    def forward(self, x):
        return self.decoder(x)
