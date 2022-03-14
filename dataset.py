# -*- coding: utf-8 -*-
"""
Created on Fri Aug  6 15:30:41 2021

@author: Administrator
"""

import torch
import os
from PIL import Image
import random
import numpy as np
import pickle
import torchvision.transforms as transforms
import cv2


class CelebADataset(torch.utils.data.Dataset):
    """docstring for CelebADataset"""

    def __init__(self, path, data1, label):
        super(CelebADataset, self).__init__()
        self.data1 = data1
        self.label = label
        self.path = path
        self.transform = transforms.ToTensor()
    def __getitem__(self, index):
        data1 = self.data1[self.path[index]]
        data1 = self.transform(data1)
        #print(data1.shape)
        #data1 = data1.permute(2, 0, 1)

        label = self.label[self.path[index]]
        #label = torch.from_numpy(np.array(label)).type(torch.FloatTensor).long()
        return data1, label

    def __len__(self):
        return len(self.path)