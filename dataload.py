import os

import os
import torch
import tqdm
from tqdm.asyncio import trange, tqdm
from evaluate.fid_score import fid_score
import torchvision.utils as tvu

import glob
import cv2
import matplotlib.pyplot as plt
from torchvision.datasets import MNIST, CIFAR10, CelebA, CIFAR100
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.transforms import Compose, ToTensor, Lambda, ToPILImage, CenterCrop, Resize


def data_store(train_loader,path):
    transform = transforms.Compose([transforms.Resize((64, 64)),
                                      transforms.ToTensor()
                                      ])
    # dataset = CelebA('.', transform=transform, download=True)
    # data_loader = DataLoader(dataset, batch_size=100,
    #                          shuffle=True)
    j=0
    for x,y in tqdm(train_loader):
        x = x.to('cuda')
        n = len(x)
        for i in range(n):
            sam = x[i]
            j = j + 1
            tvu.save_image( sam, os.path.join(path, f"{j}.png"))


