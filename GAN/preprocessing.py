import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image

# Get data
# Using MNIST
def get_data():
    
    # Transform to normalise pixels
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,),
                            std=(0.5,))])

    # Data set
    train_data = torchvision.datasets.MNIST(
        root='.',
        train=True,
        transform=transform,
        download=True)
    
    return train_data

# Prep data laoder
def prep_data(train_data, batch_size):

    data_loader = torch.utils.data.DataLoader(dataset=train_data,
                                            batch_size=batch_size, 
                                            shuffle=True)
    
    return data_loader

# make environemt
def make_env(batch_size):

    train_data = get_data()

    train_loader = prep_data(train_data, batch_size)

    return train_loader