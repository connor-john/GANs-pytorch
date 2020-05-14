import torch
import torch.nn as nn
import torch.nn.functional as F

# Discriminator Model
class Disc(nn.Module):
  def __init__(self, input_dim, ndf):
    super(Disc, self).__init__()

    self.main = nn.Sequential(
      
      # Input layer
      nn.Conv2d(input_dim, ndf, 4, 2, 1, bias = False),
      nn.LeakyReLU(0.2, inplace = True),

      # First layer
      nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias = False),
      nn.BatchNorm2d(ndf * 2),
      nn.LeakyReLU(0.2, inplace = True),

      # Second layer
      nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias = False),
      nn.BatchNorm2d(ndf * 4),
      nn.LeakyReLU(0.2, inplace = True),

      # Third layer
      nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias = False),
      nn.BatchNorm2d(ndf * 8),
      nn.LeakyReLU(0.2, inplace = True),

      # Output layer
      nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias = False),
      nn.Sigmoid())
    
  def forward(self, x):
    return self.main(x)

# Generator Model
class Gen(nn.Module):
  def __init__(self, latent_dim, output_dim, ngf):
    super(Gen, self).__init__()

    self.main = nn.Sequential(

      # Input layer
      nn.ConvTranspose2d(latent_dim, ngf * 8, 4, 1, 0, bias = False),
      nn.BatchNorm2d(ngf * 8),
      nn.ReLU(True),

      # First Layer
      nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias = False),
      nn.BatchNorm2d(ngf * 2),
      nn.ReLU(True),

      # Second Layer
      nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
      nn.BatchNorm2d(ngf),
      nn.ReLU(True),

      # Output Layer
      nn.ConvTranspose2d( ngf, output_dim, 4, 2, 1, bias=False),
      nn.Tanh())

  def forward(self, x):
    return self.main(x)
    

# Taken from pytorch docs
# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)