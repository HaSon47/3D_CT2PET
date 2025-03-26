import glob
import random
import os
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import torch
import cv2





class ImageDataset(Dataset):
    def __init__(self, root, crop_scale=0.70, flip_prob=0.8):
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])
        self.crop_scale = crop_scale  # Tỷ lệ giữ lại khi crop
        self.flip_prob = flip_prob    # Xác suất flip dọc
        self.files_A = sorted(glob.glob(f"{root}/A/*"))
        self.files_B = sorted(glob.glob(f"{root}/B/*"))
    
    def __getitem__(self, index):
        B_path = self.files_B[index % len(self.files_B)]
        A_path = self.files_A[index % len(self.files_A)]
        A_image = np.load(A_path, allow_pickle=True)
        B_image = np.load(B_path, allow_pickle=True)

        # Áp dụng Augmentations giữ alignment
        A_image, B_image = self.random_augment(A_image, B_image)

        # Resize
        A_image = self.pad_to_4(A_image)
        B_image = self.pad_to_4(B_image)

        # Min-max normalization
        A_image = (A_image - np.min(A_image)) / (np.max(A_image) - np.min(A_image))
        B_image = (B_image - np.min(B_image)) / (np.max(B_image) - np.min(B_image))

        # Scale to (-1, 1)
        A_image = (A_image * 2.0) - 1.0
        B_image = (B_image * 2.0) - 1.0

        A_image = Image.fromarray(A_image)
        B_image = Image.fromarray(B_image)
        A_image = self.transform(A_image)
        B_image = self.transform(B_image)

        return {'A': A_image, 'B': B_image}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))

    def pad_to_4(self, img, pad_value=0):
        h, w = img.shape
        if h < 256:
            pad_h = 256 - h
        else:
            pad_h = (4 - (h % 4)) % 4  # Số hàng cần padding để h chia hết cho 4
        padded_image = np.pad(img, ((0, pad_h), (0, 0)), mode='constant', constant_values=pad_value)
        return padded_image
    

    def random_augment(self, img_A, img_B):
        """ Áp dụng augmentations giữ alignment giữa A và B (dữ liệu dạng numpy array) """

        # Random Vertical Flip
        if random.random() < self.flip_prob:
            img_A = np.flipud(img_A)  # Lật dọc
            img_B = np.flipud(img_B)

        # Random Center Crop & Resize
        h, w = img_A.shape  # Giả sử ảnh có dạng (H, W)
        crop_h, crop_w = int(h * self.crop_scale), int(w * self.crop_scale)

        top = (h - crop_h) // 2
        left = (w - crop_w) // 2
        bottom = top + crop_h
        right = left + crop_w

        img_A = img_A[top:bottom, left:right]
        img_B = img_B[top:bottom, left:right]

        # Resize về kích thước ban đầu (h, w)
        img_A = cv2.resize(img_A, (w, h), interpolation=cv2.INTER_LINEAR)
        img_B = cv2.resize(img_B, (w, h), interpolation=cv2.INTER_LINEAR)

        return img_A, img_B


class ValDataset(Dataset):
    def __init__(self, root):
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])
        self.files_A = sorted(glob.glob("%s/A/*" % root))
        self.files_B = sorted(glob.glob("%s/B/*" % root))
        
    def __getitem__(self, index):
        B_path = self.files_B[index % len(self.files_B)]
        A_path = self.files_A[index % len(self.files_A)]
        
        A_image = np.load(A_path, allow_pickle=True)
        B_image = np.load(B_path, allow_pickle=True)

        # Resize
        A_image = self.pad_to_4(A_image)
        B_image = self.pad_to_4(B_image)
        
            
        # Min-max normalization
        A_image = (A_image - np.min(A_image)) / (np.max(A_image) - np.min(A_image))
        B_image = (B_image - np.min(B_image)) / (np.max(B_image) - np.min(B_image))
        
        # Scale to (-1, 1)
        A_image = (A_image * 2.0) - 1.0
        B_image = (B_image * 2.0) - 1.0
        
        A_image = Image.fromarray(A_image)
        B_image = Image.fromarray(B_image)
        A_image = self.transform(A_image)
        B_image = self.transform(B_image)
        
        
        return {'A': A_image, 'B': B_image, 'base_name': os.path.basename(A_path)}
    def __len__(self):
        return max(len(self.files_A), len(self.files_B))
       
    def pad_to_4(self, img, pad_value=0): # img shape h x 256
        h, w = img.shape 
        pad_h = (4 - (h % 4)) % 4  # Số hàng cần padding để h chia hết cho 4
        padded_image = np.pad(img, ((0, pad_h), (0, 0)), mode='constant', constant_values=pad_value)
        return padded_image  
    