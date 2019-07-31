import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.l1 = local_conv_layer(3, 64, 3)
        self.l2 = local_conv_layer(64, 64, 3)
        self.l3 = local_conv_layer(64, 128, 3)
        self.l4 = local_conv_layer(128, 256, 3, nn.Tanh)

    def forward(self, x):
        h1 = self.l1(x)
        h2 = self.l2(h1)
        h3 = self.l3(h2)
        h4 = self.l4(h3)
        return (h1, h2, h3, h4), h4

class Decoder(nn.Module):

    def __init__(self):
        super(Decoder, self).__init__()
        self.conv_block1 = decoder_layer(256, 512, 4, 1)
        self.conv_block2 = decoder_layer(512, 256, 4, 1)
        self.conv_block3 = decoder_layer(256, 128, 4, 1)
        self.conv4 = nn.ConvTranspose2d(128, 3, kernel_size=4,
                                        stride=2, padding=1)

    def forward(self, x):
        batch_size = x.size(0)
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv4(x)
        return x


def local_conv_layer(input_dim, output_dim, kernel_size, Active=nn.ReLU):
    return nn.Sequential(
            nn.Conv2d(
                in_channels=input_dim,
                out_channels=output_dim,
                kernel_size=kernel_size,
                padding=1,
                ),
            nn.Conv2d(
                in_channels=output_dim,
                out_channels=output_dim,
                kernel_size=kernel_size,
                padding=1,
                ),
            Active(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            )


def sensitive_conv_layer(input_dim, output_dim, kernel_size, Active=nn.ReLU):
    return nn.Sequential(
            nn.Conv2d(
                in_channels=input_dim,
                out_channels=output_dim,
                kernel_size=kernel_size,
                padding=1,
                ),
            nn.Conv2d(
                in_channels=output_dim,
                out_channels=output_dim,
                dilation=2,
                kernel_size=kernel_size,
                stride=2,
                ),
            Active(),
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
