import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as dset

# Get data
# Using Celeba
def get_data(dataroot, image_size):
    
    # Transform to normalise pixels
    dataset = dset.ImageFolder(root=dataroot,
                                transform=transforms.Compose([
                                transforms.Resize(image_size),
                                transforms.CenterCrop(image_size),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                            ]))
        
    return dataset

# Prep data laoder
def prep_data(dataset, batch_size, workers):

    # optional workers
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=workers)
    
    return dataloader

# make environemt
def make_env(batch_size, dataroot, image_size, workers):

    dataset = get_data(dataroot, image_size)

    dataloader = prep_data(dataset, batch_size, workers)

    return dataloader