import torch
import torch.nn as nn
from torchvision.utils import save_image
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from preprocessing import make_env
from models import Disc, Gen
import os

# Hyper params
batch_size = 128
n_epochs = 200
latent_dim = 100

# scale image method
# scales back to (0,1)
def scale_image(img):
    out = (img + 1) / 2
    return out

# Training loop
def train(G, D, data_loader, device, criterion, d_optimizer, g_optimizer, n_epochs, latent_dim):

    # labels
    ones_ = torch.ones(batch_size, 1).to(device)
    zeros_ = torch.zeros(batch_size, 1).to(device)

    # losses
    d_losses = []
    g_losses = []

    # loop
    for ep in range(n_epochs):
        for inputs, _ in data_loader:

            # reshape
            n = inputs.size(0)
            inputs = inputs.reshape(n, 784).to(device)

            # set ones/zeros to size
            ones = ones_[:n]
            zeros = zeros_[:n]

            #
            # Train Discriminator
            #

            # real images
            real_out = D.forward(inputs)
            d_loss_real = criterion(real_out, ones)

            # fake images
            noise = torch.randn(n, latent_dim).to(device)
            fake_image = G.forward(noise)
            fake_out = D.forward(fake_image)
            d_loss_fake = criterion(fake_out, zeros)

            # grad descent step
            d_loss = 0.5 * (d_loss_real + d_loss_fake)
            d_optimizer.zero_grad()
            g_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()

            #
            # Train Generator
            #

            for _ in range(2):

                # fake image
                noise = torch.randn(n, latent_dim).to(device)
                fake_image = G.forward(noise)
                fake_out = D.forward(fake_image)

                # reverse labels
                g_loss = criterion(fake_out, ones)

                # grad descent step
                d_optimizer.zero_grad()
                g_optimizer.zero_grad()
                g_loss.backward()
                g_optimizer.step()

            # save losses
            d_losses.append(d_loss.item())
            g_losses.append(g_loss.item())
    
        # print
        print(f"epoch: {ep} | d_loss: {d_loss.item()} | g_loss: {g_loss.item()}")

        # save
        fake_image = fake_image.reshape(-1, 1, 28, 28)
        save_image(scale_image(fake_image), f"gan_images/{ep+1}.png")

# main
if __name__ == '__main__':

    # initialise
    data_loader = make_env(batch_size)

    D = Disc()
    G = Gen(latent_dim)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    D = D.to(device)
    G = G.to(device)

    # loss and optimizers
    criterion = nn.BCEWithLogitsLoss()
    d_optimizer = torch.optim.Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999))
    g_optimizer = torch.optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))

    # Train
    train(G, D, data_loader, device, criterion, d_optimizer, g_optimizer, n_epochs, latent_dim)
