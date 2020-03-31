import torch
from torch import optim
from torch.autograd import Variable
from tqdm import tqdm
import models

import sys
sys.path.append('..')

import data
import gin

import cloudpickle


@gin.configurable
def train_model(model,
                epochs,
                batch_size,
                sample_size,
                lr,
                weight_decay,
                loss_log_interval,
                model_save_path,
                device):

    # prepare optimizer and model
    model.train()
    optimizer = optim.Adam(
        model.parameters(), lr=lr,
        weight_decay=weight_decay,
    )

    for epoch in range(epochs):
        data_loader = data.get_cifar10_train_loader(batch_size, n_augments=0)
        # data_loader = utils.get_data_loader(dataset, batch_size, cuda=cuda)
        data_stream = tqdm(enumerate(data_loader, 1))

        for batch_index, (x, _) in data_stream:
            # where are we?
            iteration = (epoch-1)*(len(data_loader.dataset)//batch_size) + batch_index

            # prepare data on gpu if needed
            # x = Variable(x).cuda() if cuda else Variable(x)
            x = x.to(device)

            # flush gradients and run the model forward
            optimizer.zero_grad()
            (mean, logvar), x_reconstructed = model(x)
            reconstruction_loss = model.reconstruction_loss(x_reconstructed, x)
            kl_divergence_loss = model.kl_divergence_loss(mean, logvar)
            total_loss = reconstruction_loss + kl_divergence_loss

            # backprop gradients from the loss
            total_loss.backward()
            optimizer.step()

            # update progress
            data_stream.set_description((
                'epoch: {epoch} | '
                'iteration: {iteration} | '
                'progress: [{trained}/{total}] ({progress:.0f}%) | '
                'loss => '
                'total: {total_loss:.4f} / '
                're: {reconstruction_loss:.3f} / '
                'kl: {kl_divergence_loss:.3f}'
            ).format(
                epoch=epoch,
                iteration=iteration,
                trained=batch_index * len(x),
                total=len(data_loader.dataset),
                progress=(100. * batch_index / len(data_loader)),
                total_loss=total_loss.item(),
                reconstruction_loss=reconstruction_loss.item(),
                kl_divergence_loss=kl_divergence_loss.item(),
            ))

            if iteration % loss_log_interval == 0:
                losses = [
                    reconstruction_loss.item(),
                    kl_divergence_loss.item(),
                    total_loss.item()
                ]

    # torch.save(model, model_save_path, pickle_module=cloudpickle)
    torch.save(model, model_save_path)


if __name__ == '__main__':
    gin.parse_config_file('config.gin')

    linear = models.LinearLayer
    conv = models.Conv2DLayer()
    deconv = models.Deconv2DLayer()

    encoder = models.Encoder(conv_layer=conv, linear_layer=linear)
    decoder = models.Decoder(deconv_layer=deconv)

    model = models.VAE(encoder, decoder)

    train_model(model)

