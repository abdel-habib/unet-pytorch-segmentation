import os
import time
from glob import glob 

import torch
from torch.utils.data import DataLoader
import torch.nn as nn

from model.unet import UNet
from model.hyperparameters import get_hparams

from loss import DiceLoss, DiceBCELoss
from utils import seeding, create_dir, epoch_time


from loguru import logger

if __name__ == "__main__":
    seeding(42)

    create_dir("checkpoints")

    # load the dataset
    train_x = sorted(glob(os.path.join(os.getcwd(), 'data/augmented/train/image/*')))
    train_y = sorted(glob(os.path.join(os.getcwd(), 'data/augmented/train/mask/*')))
    
    valid_x = sorted(glob(os.path.join(os.getcwd(), 'data/augmented/test/image/*')))
    valid_y = sorted(glob(os.path.join(os.getcwd(), 'data/augmented/test/mask/*')))

    logger.info(f"train size: {len(train_x)}, valid size: {len(valid_x)}")

    hyperparameters = get_hparams()
