import torch
import torch.nn as nn
from torchvision.utils import save_image
import numpy as np
import matplotlib.pyplot as plt
from preprocessing import make_env
from models import Disc, Gene, weights_init
import os
import torchvision.utils as utils
import torch.optim as optim

# Hyper params
dataroot = 'celeba/'
image_size = 64
workers = 2
batch_size = 128
num_epochs = 5
lr = 0.0002
latent_dim = 100
ngf = 64
ndf = 64
input_dim = 3
beta1 = 0.5

# Training loop
def train(Gen, Dis, dataloader, device, criterion, D_optimizer, G_optimizer, num_epochs, latent_dim):

    img_list = []
    G_losses = []
    D_losses = []
    iters = 0

    for ep in range(num_epochs):
        for i, data in enumerate(dataloader, 0):

            #
            # Discriminator training
            #

            ### real batch

            Dis.zero_grad()

            # format batch
            real_cpu = data[0].to(device)
            size = real_cpu.size(0)
            label = torch.full((size,), real_label, device=device)

            # forward
            out = Dis(real_cpu).view(-1)

            # real loss
            d_err_real = criterion(out, label)
            d_err_real.backward()
            D_x = out.mean().item()

            ### fake batch

            noise = torch.randn(size, latent_dim, 1, 1, device=device)

            # gen image
            fake = Gen(noise)
            label.fill_(fake_label)

            # classify fake
            out = Dis(fake.detach()).view(-1)

            # fake loss
            d_err_fake = criterion(out, label)
            d_err_fake.backward()
            D_G_z1 = out.mean().item()

            # add grads
            d_err = d_err_real + d_err_fake

            # update D
            D_optimizer.step()

            #
            # Generator Training
            #

            Gen.zero_grad()

            label.fill_(real_label) # fake labels are real for generator cost

            # updated D forward pass
            out = Dis(fake).view(-1)

            # loss
            g_err = criterion(out, label)
            g_err.backward()
            D_G_z2 = out.mean().item()

            # update G
            G_optimizer.step()

            if i % 50 == 0:
                print(f'{ep}/{num_epochs} | {i}/{len(dataloader)} | D_loss: {d_err.item()} | G_loss: {g_err.item()}')
        
            # Save Losses for plotting later
            G_losses.append(g_err.item())
            D_losses.append(d_err.item())

            # Check how the generator is doing by saving G's output on fixed_noise
            if (iters % 500 == 0) or ((ep == num_epochs-1) and (i == len(dataloader)-1)):
                with torch.no_grad():
                    fake = Gen(fixed_noise).detach().cpu()
                img_list.append(utils.make_grid(fake, padding=2, normalize=True))

            iters += 1

def get_generator(latent_dim, input_dim, ngf):
    
    # Initialise generator
    G = Gene(latent_dim, output_dim = input_dim, ngf = ngf)

    G.apply(weights_init)

    G = G.to(device)

    return G

def get_discriminator(input_dim, ndf):
    
    # Initialise Discriminator
    Dis = Disc(input_dim, ndf)

    Dis.apply(weights_init)

    Dis = Dis.to(device)

    return Dis

# main
if __name__ == '__main__':

    # initialise
    dataloader = make_env(batch_size, dataroot, image_size, workers)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    Dis = get_discriminator(input_dim, ndf)
    Gen = get_generator(latent_dim, input_dim, ngf)
    
    # Initialise loss, noise and optim
    criterion = nn.BCELoss()

    fixed_noise = torch.randn(64, latent_dim, 1, 1, device=device)

    real_label = 1
    fake_label = 0

    D_optimizer = optim.Adam(Dis.parameters(), lr=lr, betas=(beta1, 0.999))
    G_optimizer = optim.Adam(Gen.parameters(), lr=lr, betas=(beta1, 0.999))

    # Train
    train(Gen, Dis, dataloader, device, criterion, D_optimizer, G_optimizer, num_epochs, latent_dim)
