import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

class DriveDataset(Dataset):
    def __init__(self, images_path, masks_path):
        super(DriveDataset, self).__init__()

        self.images_path = images_path
        self.masks_path = masks_path
        self.n_samples = len(images_path)

    def __getitem__(self, index):
        # reading the image
        img = cv2.imread(self.images_path[index], cv2.IMREAD_COLOR)
        img = img/255.0 # image dim: (512, 512, 3)
        img = np.transpose(img, (2, 0, 1)) # new dim: (3, 512, 512)
        img = img.astype(np.float32)
        img = torch.from_numpy(img)

        # reading the mask
        mask = cv2.imread(self.masks_path[index], cv2.IMREAD_GRAYSCALE)
        mask = mask/255.0 # mask dim: (512, 512)
        mask = np.expand_dims(mask, axis=0) # new dim: (1, 512, 512)
        mask = mask.astype(np.float32)
        mask = torch.from_numpy(mask)

        return img, mask
    
    def __len__(self):
        return self.n_samples