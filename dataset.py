import os
from PIL import Image
import torch
from torch.utils import data
import numpy as np
from torchvision import transforms as T
import torchvision
import cv2
import sys


class Dataset(data.Dataset):

    def __init__(self, root, data_list_file, phase='train', input_shape=(1, 112, 112)):
        self.phase = phase
        self.input_shape = input_shape

        with open(os.path.join(data_list_file), 'r') as fd:
            imgs = fd.readlines()

        imgs = [os.path.join(root, img[:-1]) for img in imgs]
        self.imgs = np.random.permutation(imgs)

        # normalize = T.Normalize(mean=[0.5, 0.5, 0.5],
        #                         std=[0.5, 0.5, 0.5])

        normalize = T.Normalize(mean=[0.60746885, 0.47471561, 0.41313071], std=[0.2621818,0.23118107,0.2242216])
        #normalize = T.Normalize(mean=[0.5074], std=[0.2334])
        if self.phase == 'train':
            self.transforms = T.Compose([
                T.RandomCrop(self.input_shape[1:]),
                T.RandomHorizontalFlip(),
                T.RandomVerticalFlip(),
                T.ToTensor(),
                normalize
            ])
        else:
            self.transforms = T.Compose([
                T.CenterCrop(self.input_shape[1:]),
                T.ToTensor(),
                normalize
            ])

        self.targets = [int(sample.split()[1]) for sample in self.imgs]

    def __getitem__(self, index):
        sample = self.imgs[index]
        splits = sample.split()
        img_path = splits[0]
        data = Image.open(img_path)
        # data = data.convert('L')
        data = self.transforms(data)
        label = int(splits[1])
        return data.float(), label

    def __len__(self):
        return len(self.imgs)