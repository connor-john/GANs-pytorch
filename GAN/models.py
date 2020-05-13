import torch
import torch.nn as nn
import torch.nn.functional as F

# Discriminator Model
class Disc(nn.Module):
  def __init__(self):
    super(Disc, self).__init__()

    self.fc1 = nn.Linear(784, 512)
    self.fc2 = nn.Linear(512, 256)
    self.fc3 = nn.Linear(256, 1)
  
  def forward(self, x):
    out = F.leaky_relu(self.fc1(x), negative_slope=0.2)
    out = F.leaky_relu(self.fc2(out), negative_slope=0.2)
    out = self.fc3(out)

    return out

# Generator Model
class Gen(nn.Module):
  def __init__(self, latent_dim):
    super(Gen, self).__init__()

    self.fc1 = nn.Linear(latent_dim, 256)
    self.bnorm1 = nn.BatchNorm1d(256, momentum = 0.7)
    self.fc2 = nn.Linear(256, 512)
    self.bnorm2 = nn.BatchNorm1d(512, momentum = 0.7)
    self.fc3 = nn.Linear(512, 1024)
    self.bnorm3 = nn.BatchNorm1d(1024, momentum = 0.7)
    self.fc4 = nn.Linear(1024, 784)

  def forward(self, x):
    out = F.leaky_relu(self.fc1(x), negative_slope=0.2)
    out = self.bnorm1(out)
    out = F.leaky_relu(self.fc2(out), negative_slope=0.2)
    out = self.bnorm2(out)
    out = F.leaky_relu(self.fc3(out), negative_slope=0.2)
    out = self.bnorm3(out)
    out = torch.tanh(self.fc4(out))

    return out