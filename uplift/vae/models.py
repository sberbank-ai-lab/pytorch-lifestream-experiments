import torch
from torch.autograd import Variable
from torch import nn
import math


class Encoder(nn.Module):
    def __init__(self, image_size, channel_num, kernel_num, z_size, conv_layer, linear_layer):
        super().__init__()
        self.kernel_num = kernel_num

        self.conv = nn.Sequential(
            conv_layer(channel_num, kernel_num // 4),
            conv_layer(kernel_num // 4, kernel_num // 2),
            conv_layer(kernel_num // 2, kernel_num),
        )
        # encoded feature's size and volume
        self.feature_size = image_size // 8
        self.feature_volume = kernel_num * (self.feature_size ** 2)

        # q
        self.q_mean = linear_layer(self.feature_volume, z_size, relu=False)
        self.q_logvar = linear_layer(self.feature_volume, z_size, relu=False)

        # projection
        self.project = linear_layer(z_size, self.feature_volume, relu=False)

    def forward(self, x):
        encoded = self.conv(x)

        # sample latent code z from q given x.
        mean, logvar = self.q(encoded)
        z = self.z(mean, logvar)
        z_projected = self.project(z).view(
            -1, self.kernel_num,
            self.feature_size,
            self.feature_size,
        )
        return z_projected, mean, logvar

    def q(self, encoded):
        unrolled = encoded.view(-1, self.feature_volume)
        return self.q_mean(unrolled), self.q_logvar(unrolled)

    def z(self, mean, logvar):
        std = logvar.mul(0.5).exp_()
        eps = (
            Variable(torch.randn(std.size())).to(self.device)
        )
        return eps.mul(std).add_(mean)

    def sample(self, size):
        z = Variable(
            torch.randn(size, self.z_size).cuda()
        )
        z_projected = self.project(z).view(
            -1, self.kernel_num,
            self.feature_size,
            self.feature_size,
        )
        return self.decoder(z_projected).data


class Decoder(nn.Module):
    def __init__(self, kernel_num, channel_num, deconv_layer):
        super().__init__()
        self.decoder = nn.Sequential(
            deconv_layer(kernel_num, kernel_num // 2),
            deconv_layer(kernel_num // 2, kernel_num // 4),
            deconv_layer(kernel_num // 4, channel_num),
            nn.Sigmoid()
        )

    def forward(self, x):
        x_reconstructed = self.decoder(x)
        return x_reconstructed


class VAE(nn.Module):
    def __init__(self, label, image_size, channel_num, kernel_num, z_size, device):
        # configurations
        super().__init__()
        self.label = label
        self.image_size = image_size
        self.channel_num = channel_num
        self.kernel_num = kernel_num
        self.z_size = z_size
        self.device = device

        self.encoder = Encoder(image_size, channel_num, kernel_num, z_size, self._conv, self._linear)
        self.decoder = Decoder(kernel_num, channel_num, self._deconv)

        # DI, device
        self.encoder.device = device

    def forward(self, x):
        # encode x
        z_projected, mean, logvar = self.encoder(x)
        # reconstruct x from z
        x_reconstructed = self.decoder(z_projected)

        # return the parameters of distribution of q given x and the
        # reconstructed image.
        return (mean, logvar), x_reconstructed

    def reconstruction_loss(self, x_reconstructed, x):
        return nn.BCELoss(size_average=False)(x_reconstructed, x) / x.size(0)

    def kl_divergence_loss(self, mean, logvar):
        return ((mean**2 + logvar.exp() - 1 - logvar) / 2).mean()

    # =====
    # Utils
    # =====

    @property
    def name(self):
        return (
            'VAE'
            '-{kernel_num}k'
            '-{label}'
            '-{channel_num}x{image_size}x{image_size}'
        ).format(
            label=self.label,
            kernel_num=self.kernel_num,
            image_size=self.image_size,
            channel_num=self.channel_num,
        )

    """
    Layers
    """
    def _conv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels,
                kernel_size=4, stride=2, padding=1,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def _deconv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels, out_channels,
                kernel_size=4, stride=2, padding=1,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def _linear(self, in_size, out_size, relu=True):
        return nn.Sequential(
            nn.Linear(in_size, out_size),
            nn.ReLU(),
        ) if relu else nn.Linear(in_size, out_size)


class Layer(nn.Module):
    def __init__(self, layers_arch, kernel_size, stride, padding):
        """

        :param layers_arch:
        :param kernel_size:
        :param stride:
        :param padding:
        """
        super().__init__()
        mlp_seq = []
        for i in range(len(layers_arch) - 1):
            mlp_seq.append(Layer._layer_block(layers_arch[i], layers_arch[i + 1], kernel_size, stride, padding))

        self.mlp = nn.Sequential(
            *mlp_seq
        )

    def forward(self, x):
        return self.mlp(x)

    @staticmethod
    def _layer_block(in_channels, out_channels, kernel_size, stride, padding):
        """

        :param in_channels:
        :param out_channels:
        :param kernel_size:
        :param stride:
        :param padding:
        :return:
        """
        raise NotImplemented

    @staticmethod
    def get_output_shape(img_size, padding, kernel_size, stride):
        """
        Compute output shape of conv2D

        :param img_size:
        :param padding:
        :param kernel_size:
        :param stride:
        :return:
        """
        raise NotImplemented


class Conv2DLayer(Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def _layer_block(in_channels, out_channels, kernel_size, stride, padding):
        """

        :param in_channels:
        :param out_channels:
        :param kernel_size:
        :param stride:
        :param padding:
        :return:
        """
        return nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels,
                kernel_size=kernel_size, stride=stride, padding=padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    @staticmethod
    def get_output_shape(img_size, padding, kernel_size, stride):
        """
        Compute output shape of conv2D

        :param img_size:
        :param padding:
        :param kernel_size:
        :param stride:
        :return:
        """
        output_shape = (
            math.floor((img_size[0] + 2 * padding[0] - (kernel_size[0] - 1) - 1) / stride[0] + 1).astype(int),
            math.floor((img_size[1] + 2 * padding[1] - (kernel_size[1] - 1) - 1) / stride[1] + 1).astype(int)
        )
        return output_shape


class Deconv2DLayer(Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def _layer_block(in_channels, out_channels, kernel_size, stride, padding):

        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels, out_channels,
                kernel_size=kernel_size, stride=stride, padding=padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    @staticmethod
    def get_output_shape(img_size, padding, kernel_size, stride):
        """
        Compute output shape of conv2D

        :param img_size:
        :param padding:
        :param kernel_size:
        :param stride:
        :return:
        """
        output_shape = ((img_size[0] - 1) * stride[0] - 2 * padding[0] + kernel_size[0],
                    (img_size[1] - 1) * stride[1] - 2 * padding[1] + kernel_size[1])
        return output_shape
