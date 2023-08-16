import os, time
from operator import add
import numpy as np
from glob import glob
import cv2
from tqdm import tqdm
import torch
from sklearn.metrics import accuracy_score, f1_score, jaccard_score, precision_score, recall_score
import argparse

from model.unet import UNet
from utils import create_dir, seeding

def calculate_metrics(y_true, y_pred):
    # gt
    y_true = y_true.cpu().numpy()
    y_true = y_true > 0.5
    y_true = y_true.astype(np.uint8)
    y_true = y_true.reshape(-1)

    # prediction
    y_pred = y_pred.cpu().numpy()
    y_pred = y_pred > 0.5
    y_pred = y_pred.astype(np.uint8)
    y_pred = y_pred.reshape(-1)

    score_jaccard = jaccard_score(y_true, y_pred)
    score_f1 = f1_score(y_true, y_pred)
    score_recall = recall_score(y_true, y_pred)
    score_precision = precision_score(y_true, y_pred)
    score_acc = accuracy_score(y_true, y_pred)

    return [score_jaccard, score_f1, score_recall, score_precision, score_acc]

def mask_parse(mask):
    mask = np.expand_dims(mask, axis=-1) # (512, 512, 1)
    mask = np.concatenate([mask, mask, mask], axis=-1) # (512, 512, 3)
    return mask


if __name__ == "__main__":
    # command args
    parser = argparse.ArgumentParser()

    parser.add_argument('--test_path', type=str, default='data/test/', help='root dir for the test data directory')
    parser.add_argument('--output', type=str, default='results/', help="output dir for saving the segmentation results")
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--ckpt', type=str, default='checkpoints/', help='pretrained checkpoint')
    parser.add_argument('--img_size', type=int, default=512, help='input patch size of network input')

    args = parser.parse_args()

    args.exp = args.ckpt.split('/')[1]
    output_path = os.path.join(args.output, "{}".format(args.exp))
    segmentation_results_path = os.path.join(output_path, 'segmentation_results')
    log_path = os.path.join(output_path, "runs")

    seeding(args.seed)

    # create results folder if it doesn't exist
    create_dir(segmentation_results_path)

    # load the test dataset, note that we have the test as the valid (not sufficient data)
    # the test has to be resized too in the augmentation process if it is different
    # thus we will use the same valid data for testing on this dataset
    test_x = sorted(glob(os.path.join(os.getcwd(), args.test_path ,'images/*')))
    test_y = sorted(glob(os.path.join(os.getcwd(), args.test_path, 'masks/*')))

    # load the checkpoint
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = UNet()
    model = model.to(device)
    model.load_state_dict(torch.load(os.path.join(os.getcwd(), args.ckpt), map_location=device))
    model.eval()

    # calculate the metrics
    metrics_score = [0.0, 0.0, 0.0, 0.0, 0.0]
    time_taken = []

    for idx, (x, y) in tqdm(enumerate(zip(test_x, test_y)), total=len(test_x)):
        name = x.split('/')[-1].split('.')[0]

        # reading the image
        img = cv2.imread(x, cv2.IMREAD_COLOR) # (512, 512, 3)
        img = cv2.resize(img, (args.img_size, args.img_size))

        x = np.transpose(img, (2, 0, 1)) # (3, 512, 512)
        x = x/255.0
        x = np.expand_dims(x, axis=0) # (1, 3, 512, 512), batch of 1 image
        x = x.astype(np.float32)
        x = torch.from_numpy(x)
        x = x.to(device)

        # reading the mask
        mask = cv2.imread(y, cv2.IMREAD_GRAYSCALE) # (512, 512)
        mask = cv2.resize(mask, (args.img_size, args.img_size))

        y = np.expand_dims(mask, axis=0) # (1, 512, 512)
        y = y/255.0
        y = np.expand_dims(y, axis=0) # (1, 1, 512, 512)
        y = y.astype(np.float32)
        y = torch.from_numpy(y)
        y = y.to(device)


        # prediction
        with torch.no_grad():
            start_time = time.time()
            
            pred_y = model(x)
            pred_y = torch.sigmoid(pred_y)

            total_time = time.time() - start_time
            time_taken.append(total_time)

            # calculate the metrics
            score = calculate_metrics(y, pred_y)
            metrics_score = list(map(add, metrics_score, score))

            # prediction
            pred_y = pred_y[0].cpu().numpy() # taking the 1st patch since it consists of batch of 1, (1, 512, 512)
            pred_y = np.squeeze(pred_y, axis=0) # (512, 512)
            pred_y = pred_y > 0.5
            pred_y = np.array(pred_y, dtype=np.uint8)

        # saving masks
        original_mask = mask_parse(mask)
        pred_y = mask_parse(pred_y)
        line = np.ones([args.img_size, 10, 3]) * 128

        cat_img = np.concatenate(
            [img, line, original_mask, line, pred_y * 255], axis=1
        )

        cv2.imwrite(os.path.join(os.getcwd(), segmentation_results_path, f'{name}.png'), cat_img)

    # calculate the mean score
    jaccard = metrics_score[0]/len(test_x)
    f1 = metrics_score[1]/len(test_x)
    recall = metrics_score[2]/len(test_x)
    precision = metrics_score[3]/len(test_x)
    acc = metrics_score[4]/len(test_x)
    print(f"Jaccard: {jaccard:1.4f} | F1: {f1:1.4f} | Recall: {recall:1.4f} | Precision: {precision:1.4f} | Accuracy: {acc:1.4f}")
    
    fps = 1/np.mean(time_taken)
    print("FPS: ", fps)

    with open(os.path.join(os.getcwd(), log_path, args.ckpt.split('/')[-1].split('.')[0]+"_test.txt"), 'w') as f:
        f.write(f'--test_path "{args.test_path}" --output "{args.output}" --seed {args.seed} --ckpt "{args.ckpt}" --img_size {args.img_size}\n\n')
        f.write("Metrics obtained:\n")
        f.write(f"Jaccard: {jaccard:1.4f}\n")
        f.write(f"F1: {f1:1.4f}\n")
        f.write(f"Recall: {recall:1.4f}\n")
        f.write(f"Precision: {precision:1.4f}\n")
        f.write(f"Accuracy: {acc:1.4f}\n\n")
        f.write(f"FPS: {fps}")



        




