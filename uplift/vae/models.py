import torch
from torch.autograd import Variable
from torch import nn
import math
import gin


@gin.configurable
class Encoder(nn.Module):
    def __init__(self, num_kernels, z_size, conv_layer, linear_layer, device):
        super().__init__()
        self.num_kernels = num_kernels
        self.device = device

        # TODO: init
        self.conv_layer = conv_layer

        # encoded feature's size and volume
        conv_layer_out_shape = conv_layer.get_output_shape()
        self.conv_layer_out_shape = conv_layer_out_shape[0]*conv_layer_out_shape[1]

        # q
        self.q_mean = linear_layer(self.conv_layer_out_shape, z_size, is_relu=False)
        self.q_logvar = linear_layer(self.conv_layer_out_shape, z_size, is_relu=False)

        # projection
        self.project = linear_layer(z_size, self.conv_layer_out_shape, is_relu=False)

    def forward(self, x):
        encoded = self.conv_layer(x)

        # sample latent code z from q given x.
        mean, logvar = self.q(encoded)
        z = self.z(mean, logvar)
        # TODO: review
        '''
        z_projected = self.project(z).view(
            -1, self.num_kernels,
            self.conv_layer_out_shape,
            self.conv_layer_out_shape,
        )
        '''
        z_projected = self.project(z).view(
            -1,
            self.num_kernels,
            4,
            4
        )
        return z_projected, mean, logvar

    def q(self, encoded):
        unrolled = encoded.view(-1, self.conv_layer_out_shape)
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
            -1, self.num_kernels,
            self.conv_layer_out_shape,
            self.conv_layer_out_shape,
        )
        return self.decoder(z_projected).data


@gin.configurable
class Decoder(nn.Module):
    def __init__(self, deconv_layer):
        super().__init__()
        self.decoder = nn.Sequential(
            deconv_layer,
            nn.Sigmoid()
        )

    def forward(self, x):
        x_reconstructed = self.decoder(x)
        return x_reconstructed


@gin.configurable
class VAE(nn.Module):
    def __init__(self, encoder, decoder, device):
        # configurations
        super().__init__()

        self.encoder = encoder().to(device)
        self.decoder = decoder().to(device)

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


"""
Layers
"""


class BaseLayer(nn.Module):
    def __init__(self, layer_sizes, img_size, kernel_size, stride, padding):
        """
        Universal layer base class.

        :param layer_sizes:
        :param kernel_size:
        :param stride:
        :param padding:
        """
        super().__init__()

        self.img_size = img_size
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        mlp_seq = []
        for i in range(len(layer_sizes) - 1):
            mlp_seq.append(self._layer_block(layer_sizes[i], layer_sizes[i + 1], kernel_size, stride, padding))

        self.mlp = nn.Sequential(
            *mlp_seq
        )

    def forward(self, x):
        return self.mlp(x)

    def _layer_block(self, in_channels, out_channels, kernel_size, stride, padding):
        """

        :param in_channels:
        :param out_channels:
        :param kernel_size:
        :param stride:
        :param padding:
        :return:
        """
        raise NotImplemented

    def get_output_shape(self):
        """
        Compute output shape of conv2D

        :return:
        """
        raise NotImplemented


@gin.configurable
class Conv2DLayer(BaseLayer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _layer_block(self, in_channels, out_channels, kernel_size, stride, padding):
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

    def get_output_shape(self):
        """
        Compute output shape of conv2D

        :return:
        """
        output_shape = (
            math.floor((self.img_size[0] + 2 * self.padding - (self.kernel_size - 1) - 1) / self.stride + 1),
            math.floor((self.img_size[1] + 2 * self.padding - (self.kernel_size - 1) - 1) / self.stride + 1)
        )
        return output_shape


@gin.configurable
class Deconv2DLayer(BaseLayer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _layer_block(self, in_channels, out_channels, kernel_size, stride, padding):

        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels, out_channels,
                kernel_size=kernel_size, stride=stride, padding=padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def get_output_shape(self):
        """
        Compute output shape of conv2D

        :return:
        """
        output_shape = (
            (self.img_size[0] - 1) * self.stride - 2 * self.padding + self.kernel_size,
            (self.img_size[1] - 1) * self.stride - 2 * self.padding + self.kernel_size
        )
        return output_shape


@gin.configurable
class LinearLayer(nn.Module):

    def __init__(self, in_size, out_size, is_relu=False):
        super().__init__()

        if is_relu:
            fn = nn.ReLU
        else:
            fn = nn.Sequential

        self.mlp = nn.Sequential(
            nn.Linear(in_size, out_size),
            fn()
        )

    def forward(self, x):
        return self.mlp(x)


# gin.parse_config_file('config.gin')
