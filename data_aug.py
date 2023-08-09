import os
import numpy as np
import cv2
import imageio.v2 as imageio
from glob  import glob
from tqdm import tqdm
from albumentations import HorizontalFlip, VerticalFlip, Rotate
from torchvision import transforms
import random

class Augmentation:
    def __init__(self):
        super(Augmentation, self).__init__()

        self.dataset_path = os.path.join(os.getcwd(), 'data')
        self.img_size = (512,512)
        self.rand_rotation = 180

    def load_data(self):
        train_x = sorted(glob(os.path.join(self.dataset_path, 'training', 'images', '*.tif')))
        train_y = sorted(glob(os.path.join(self.dataset_path, 'training', '1st_manual', '*.gif'))) # mask gt

        test_x = sorted(glob(os.path.join(self.dataset_path, 'test', 'images', '*.tif')))
        test_y = sorted(glob(os.path.join(self.dataset_path, 'test', '1st_manual', '*.gif'))) # mask gt
        
        return (train_x, train_y), (test_x, test_y)
    
    def create_dirs(self, paths):
        for path in paths:
            if not os.path.exists(path):
                os.makedirs(path)

    def augment_data(self, images, masks, save_dir, augment=True):
        for idx, (x, y) in tqdm(enumerate(zip(images, masks)), total=len(images)):
            img_name = x.split('/')[-1].split('.')[0]

            train = cv2.imread(x, cv2.IMREAD_COLOR)
            mask  = imageio.imread(y) 

            # Create andd apply the transformations
            if (augment):
                # # Training augmentation
                # # The behaviour of RandomRotation on the edges are not always recommended
                # # Good alternative is iaa.Affine(rotate=(-20, 20), mode='symmetric') as in the following reference
                # # https://colab.research.google.com/drive/109vu3F1LTzD1gdVV6cho9fKGx7lzbFll#scrollTo=8q8a2Ha9pnaz
                # transformations = transforms.Compose([
                #     transforms.ToPILImage(),
                #     transforms.Resize(size=self.img_size),
                #     transforms.RandomHorizontalFlip(),
                #     transforms.RandomVerticalFlip(),
                #     transforms.RandomRotation(self.rand_rotation),
                #     # transforms.RandomAffine(self.rand_rotation, fill=255)
                # ])

                # Good approach to apply the same transformation on both train and mask images
                augmentations = [
                    HorizontalFlip(p=1.0),
                    VerticalFlip(p=1.0),
                    Rotate(limit=45, p=1.0)
                ]

                X = []
                Y = []

                # appending the original images
                X.append(train)
                Y.append(mask)

                # appending the augmented images
                for aug in augmentations:
                    augmented = aug(image=train, mask=mask)
                    X.append(augmented["image"])
                    Y.append(augmented["mask"])


            else:
                # Val augmentation, only resize
                # transformations = transforms.Compose([
                #     transforms.ToPILImage(),
                #     transforms.Resize(size=self.img_size),
                # ])

                X = [train]
                Y = [mask]

            for index, (i, m) in enumerate(zip(X, Y)):
                i = cv2.resize(i, self.img_size)
                m = cv2.resize(m, self.img_size)

                # Export the images
                image_path = os.path.join(save_dir, "image", f"{img_name}_{index}.png")
                mask_path = os.path.join(save_dir, "mask", f"{img_name}_{index}.png")
                
                cv2.imwrite(image_path, i)
                cv2.imwrite(mask_path, m)

            # transformed_img = transformations(train)
            # transformed_mask = transformations(mask)

            # Convert PIL images to NumPy arrays
            # transformed_img = np.array(transformed_img)[:, :, ::-1]  # Convert RGB to BGR
            # transformed_mask = np.array(transformed_mask)

if __name__ == "__main__":
    random.seed(42)

    # init class
    aug = Augmentation()

    # load train and test data
    (train_x, train_y), (test_x, test_y) = aug.load_data()

    # validate data length
    assert len(train_x) == len(train_y), "Train data and ground truth are not equal in length"
    assert len(test_x) == len(test_y), "Test data and ground truth are not equal in length"

    # create new dir for augmented data
    aug.create_dirs(
        [os.path.join(os.getcwd(), 'data', 'augmented', 'train/image'),
         os.path.join(os.getcwd(), 'data', 'augmented', 'train/mask'),
         os.path.join(os.getcwd(), 'data', 'augmented', 'test/image'),
         os.path.join(os.getcwd(), 'data', 'augmented', 'test/mask')]
    )

    aug.augment_data(train_x, train_y, os.path.join(os.getcwd(), 'data/augmented/train/'), augment=True)
    aug.augment_data(test_x, test_y, os.path.join(os.getcwd(), 'data/augmented/test/'), augment=False)
