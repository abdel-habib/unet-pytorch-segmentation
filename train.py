import os
import time
from glob import glob 
import argparse

import torch
from torch.utils.data import DataLoader
import torch.nn as nn

from model.unet import UNet
from dataset import DriveDataset
from callbacks.early_stopping import EarlyStopping

from loss import DiceLoss, DiceBCELoss
from utils import seeding, create_dir, epoch_time

from loguru import logger
from torchsummary import summary


def train(model, train_loader, optimizer, loss_fn, device):
    epoch_loss = 0.0

    model.train()

    for x, y in train_loader:
        x = x.to(device, dtype=torch.float32)  # Move to device
        y = y.to(device, dtype=torch.float32)  # Move to device

        optimizer.zero_grad()

        y_pred = model(x)
        loss = loss_fn(y_pred, y)

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
    
    epoch_loss = epoch_loss / len(train_loader)

    return epoch_loss

def evaluate(model, valid_loader, loss_fn, device):
    epoch_loss = 0.0

    model.eval()

    with torch.no_grad():
        for x, y in valid_loader:
            x = x.to(device, dtype=torch.float32)  # Move to device
            y = y.to(device, dtype=torch.float32)  # Move to device

            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            epoch_loss += loss.item()
    
        epoch_loss = epoch_loss / len(valid_loader)
    return epoch_loss


if __name__ == "__main__":
    # command args
    parser = argparse.ArgumentParser()

    parser.add_argument('--train_path', type=str, default='data/train', help='root dir for train data')
    parser.add_argument('--val_path', type=str, default='data/val', help='root dir for validation data')
    parser.add_argument('--output', type=str, default='results/', help="output dir for saving the segmentation results")
    parser.add_argument('--dataset', type=str, default='kvasir', help='experiment_name')
    parser.add_argument('--max_epochs', type=int, default=50, help='maximum epoch number to train')
    parser.add_argument('--batch_size', type=int, default=12, help='batch_size per gpu')
    parser.add_argument('--base_lr', type=float, default=0.0001, help='segmentation network learning rate')
    parser.add_argument('--patience', type=int, default=12, help='patience for lr scheduler')
    parser.add_argument('--img_size', type=int, default=512, help='input patch size of network input')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--ckpt', type=str, default='checkpoints/', help='pretrained checkpoint')

    args = parser.parse_args()


    args.exp = args.dataset + '_' + str(args.img_size)
    output_path = os.path.join(args.output, "{}".format(args.exp))
    snapshot_path = output_path + '_epo' + str(args.max_epochs)
    snapshot_path = snapshot_path + '_bs' + str(args.batch_size)
    snapshot_path = snapshot_path + '_lr' + str(args.base_lr)
    snapshot_path = snapshot_path + '_s' + str(args.seed)

    checkpoint_path = os.path.join(output_path, args.ckpt)
    
    checkpoint_file = args.exp + '_epo' + str(args.max_epochs)
    checkpoint_file = checkpoint_file + '_bs' + str(args.batch_size)
    checkpoint_file = checkpoint_file + '_lr' + str(args.base_lr)
    checkpoint_file = checkpoint_file + '_s' + str(args.seed)
    checkpoint_file = checkpoint_file + '.pth'
    checkpoint_file = os.path.join(checkpoint_path ,checkpoint_file)

    log_path = os.path.join(output_path, "runs")

    seeding(args.seed)

    # create model checkpoints folder if it doesn't exist
    create_dir(checkpoint_path)
    create_dir(log_path)

    # load the augmented dataset
    train_x = sorted(glob(os.path.join(os.getcwd(), args.train_path, 'images/*')))
    train_y = sorted(glob(os.path.join(os.getcwd(), args.train_path, 'masks/*')))
    valid_x = sorted(glob(os.path.join(os.getcwd(), args.val_path, 'images/*')))
    valid_y = sorted(glob(os.path.join(os.getcwd(), args.val_path, 'masks/*')))

    logger.info(f"train size: {len(train_x)}, valid size: {len(valid_x)}")    

    # dataset and loader
    train_dataset = DriveDataset(train_x, train_y)
    valid_dataset = DriveDataset(valid_x, valid_y)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2
    )

    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2
    )

    # checking gpu availability and setting up the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logger.info(f"Using device: {device}")

    model = UNet()
    model = model.to(device)
    summary(model.cuda(), (3, args.img_size, args.img_size))
    
    if device.type == 'cuda':
        print(torch.cuda.get_device_name(0))
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')

    # setting up the optimizer and the lr scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=args.base_lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, verbose=True)

    # setting up the loss function
    loss_fn = DiceBCELoss()

    # training the model
    best_valid_loss = float('inf')
    early_stopping = EarlyStopping(patience=args.patience, verbose=True, path=checkpoint_file)
    runs_file = os.path.join(log_path, snapshot_path.split('/')[-1]+'.txt')

    with open(runs_file, "a") as f:
        f.write(f'--train_path "{args.train_path}" --val_path "{args.val_path}" --output "{args.output}" --dataset "{args.dataset}" --dataset "{args.dataset}" --max_epochs {args.max_epochs} --batch_size {args.batch_size} --base_lr {args.base_lr} --patience {args.patience} --img_size {args.img_size} --seed {args.seed} --ckpt "{args.ckpt}" \n')
        f.write(f'checkpoint_file path: {checkpoint_file}\n' )
        f.write(f'runs_file path: {runs_file}\n')

        for epoch in range(args.max_epochs):
            start_time = time.time()

            # train and validate
            train_loss = train(model, train_loader, optimizer, loss_fn, device)
            valid_loss = evaluate(model, valid_loader, loss_fn, device)

            # scheduler step based on the validation loss
            scheduler.step(valid_loss)
            after_lr = optimizer.param_groups[0]["lr"]

            # handle early stopping and saving model
            early_stopping(valid_loss, model)

            if early_stopping.early_stop:
                print(f"Early stopping triggered at epoch {epoch+1}. Ending model training.")
                break
            
            end_time = time.time()
            epoch_mins, epoch_secs = epoch_time(start_time, end_time)

            print(f'Epoch: {epoch+1:02}/{args.max_epochs} | Epoch Time: {epoch_mins}m {epoch_secs}s | lr: {after_lr} | Train Loss: {train_loss:.5f} | Val. Loss: {valid_loss:.5f}')
            f.write(f'Epoch: {epoch+1:02}/{args.max_epochs} | Epoch Time: {epoch_mins}m {epoch_secs}s | lr: {after_lr} | Train Loss: {train_loss:.5f} | Val. Loss: {valid_loss:.5f}\n')
