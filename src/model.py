import torch
import torch.nn as nn
import torch.nn.functional as F


class LocalEncoder(nn.Module):

    def __init__(self):
        super(LocalEncoder, self).__init__()
        self.l1 = local_conv_layer(3, 128, 3)
        self.l2 = local_conv_layer(128, 128, 3)
        self.l3 = local_conv_layer(128, 256, 3)
        self.l4 = local_conv_layer(256, 256, 3)

    def forward(self, x):
        h = self.l1(x)
        h = self.l2(h)
        h = self.l3(h)
        h = self.l4(h)
        return h


class SensitiveEncoder(nn.Module):

    def __init__(self):
        super(SensitiveEncoder, self).__init__()
        self.l1 = sensitive_conv_layer(3, 128, 3)
        self.l2 = sensitive_conv_layer(128, 128, 3)
        self.l3 = sensitive_conv_layer(128, 256, 3)

    def forward(self, x):
        h = self.l1(x)
        h = self.l2(h)
        h = self.l3(h)
        return h


class Decoder(nn.Module):

    def __init__(self):
        super(Decoder, self).__init__()
        self.conv_block1 = decoder_layer(128, 512, 4, 0)
        self.conv_block2 = decoder_layer(512, 256, 4, 1)
        self.conv_block3 = decoder_layer(256, 128, 4, 1)
        self.conv4 = nn.ConvTranspose2d(128, 3, kernel_size=4,
                                        stride=2, padding=1)

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, 128, 1, 1)
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv4(x)
        return x


def local_conv_layer(input_dim, output_dim, kernel_size):
    return nn.Sequential(
            nn.Conv2d(
                in_channels=input_dim,
                out_channels=output_dim,
                kernel_size=kernel_size,
                padding=1,
                ),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            )


def sensitive_conv_layer(input_dim, output_dim, kernel_size):
    return nn.Sequential(
            nn.Conv2d(
                in_channels=input_dim,
                out_channels=output_dim,
                dilation=3,
                kernel_size=kernel_size,
                stride=2,
                ),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            )


def decoder_layer(input_dim, output_dim,
                  kernel_size, padding=1):
    return nn.Sequential(
        nn.ConvTranspose2d(input_dim, output_dim,
                           kernel_size=kernel_size, stride=2, padding=padding),
        nn.BatchNorm2d(output_dim),
        nn.ReLU(inplace=True),
    )
