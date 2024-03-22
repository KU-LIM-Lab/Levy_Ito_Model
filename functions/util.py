import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import torch
import matplotlib.animation as animation
import numpy as np
from models.ddpm import Model
from torchvision.datasets import CIFAR10

def image_grid(x):
  size = x.shape[1]
  channels = x.shape[-1]
  img = x.reshape(-1, size, size, channels)
  w = int(np.sqrt(x.shape[0]))
  img = img.reshape((w, w, size, size, channels)).transpose((0, 2, 1, 3, 4)).reshape((w * size, w * size, channels))
  return img

def get_model(config):
    print('config.model.model_type',config.model.model_type)
    model = Model(config)
    return model


def get_datasets(args,config):
    data= config.data.dataset
    image_size = config.data.image_size

    if data == 'CIFAR10':
        transform = transforms.Compose([transforms.Resize((image_size, image_size)),
                                    transforms.RandomHorizontalFlip(p=0.5),
                                    transforms.ToTensor()])
        dataset = CIFAR10('./data', train=True, transform=transform, download=True)
        transformer = transforms.Compose([transforms.Resize((image_size, image_size)),
                                      transforms.ToTensor()
                                      ])
        validation_dataset = CIFAR10(root='./data', train=False, download=True,
                                 transform=transformer)
    
    if args.ddp:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset,
                                                                    rank=args.rank)
        test_sampler = torch.utils.data.distributed.DistributedSampler(validation_dataset,
                                                                    rank=args.rank)
        data_loader = torch.utils.data.DataLoader(dataset,
                                                batch_size=config.training.batch_size,
                                                shuffle=False,
                                                num_workers=4,
                                                pin_memory=True,
                                                sampler=train_sampler,
                                                drop_last = True)
        val_loader = torch.utils.data.DataLoader(validation_dataset,
                                                batch_size=config.training.batch_size,
                                                shuffle=False,
                                                num_workers=4,
                                                pin_memory=True,
                                                sampler=test_sampler,
                                                drop_last = True)

    else:
        data_loader = torch.utils.data.DataLoader(dataset,
                                               batch_size=config.training.batch_size,
                                               shuffle=True,
                                               num_workers=4,
                                               pin_memory=True,
                                                drop_last = True)
        val_loader = torch.utils.data.DataLoader(validation_dataset,
                                                batch_size=config.training.batch_size,
                                                shuffle=False,
                                                num_workers=4,
                                                pin_memory=True,
                                                drop_last = True)
        train_sampler = None

    return data_loader, val_loader, train_sampler