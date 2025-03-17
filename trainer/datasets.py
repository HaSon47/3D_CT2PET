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
    def __init__(self, root):
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.ToTensor()
        ])
        self.files_A = sorted(glob.glob("%s/A/*" % root))
        self.files_B = sorted(glob.glob("%s/B/*" % root))
       
    def __getitem__(self, index):
        B_path = self.files_B[index % len(self.files_B)]
        A_path = self.files_A[index % len(self.files_A)]
        A_image = np.load(A_path, allow_pickle=True)
        B_image = np.load(B_path, allow_pickle=True)
        
        # Min-max normalization
        A_image = (A_image - np.min(A_image)) / (np.max(A_image) - np.min(A_image))
        B_image = (B_image - np.min(B_image)) / (np.max(B_image) - np.min(B_image))
        
        # Scale to (-1, 1)
        A_image = (A_image * 2.0) - 1.0
        B_image = (B_image * 2.0) - 1.0

        # Resize
        A_image = self.resize_and_pad(A_image)
        B_image = self.resize_and_pad(B_image)
        
        A_image = Image.fromarray(A_image)
        B_image = Image.fromarray(B_image)
        A_image = self.transform(A_image)
        B_image = self.transform(B_image)
        
        return {'A': A_image, 'B': B_image}
    
    def __len__(self):
        return max(len(self.files_A), len(self.files_B))
   
    def resize_and_pad(self, img, target_size=(256, 256), pad_value=0):
        h, w = img.shape
        target_h, target_w = target_size

        # Nếu ảnh lớn hơn target_size -> Resize
        if h > target_h or w > target_w:
            img = cv2.resize(img, (min(w, target_w), min(h, target_h)), interpolation=cv2.INTER_LINEAR)

        # Tạo ảnh mới có kích thước target_size và điền giá trị pad_value
        padded_img = np.full((target_h, target_w), pad_value, dtype=img.dtype)
        
        # Lấy kích thước mới của ảnh sau khi resize
        new_h, new_w = img.shape
        
        # Chèn ảnh vào giữa (align center)
        start_h = (target_h - new_h) // 2
        start_w = (target_w - new_w) // 2
        padded_img[start_h:start_h + new_h, start_w:start_w + new_w] = img

        return padded_img
    


class ValDataset(Dataset):
    def __init__(self, root):
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.ToTensor()
        ])
        self.files_A = sorted(glob.glob("%s/A/*" % root))
        self.files_B = sorted(glob.glob("%s/B/*" % root))
        
    def __getitem__(self, index):
        B_path = self.files_B[index % len(self.files_B)]
        A_path = self.files_A[index % len(self.files_A)]
        
        A_image = np.load(A_path, allow_pickle=True)
        B_image = np.load(B_path, allow_pickle=True)
        
            
        # Min-max normalization
        A_image = (A_image - np.min(A_image)) / (np.max(A_image) - np.min(A_image))
        B_image = (B_image - np.min(B_image)) / (np.max(B_image) - np.min(B_image))
        
        # Scale to (-1, 1)
        A_image = (A_image * 2.0) - 1.0
        B_image = (B_image * 2.0) - 1.0

        # Resize
        A_image = self.resize_and_pad(A_image)
        B_image = self.resize_and_pad(B_image)
        
        A_image = Image.fromarray(A_image)
        B_image = Image.fromarray(B_image)
        A_image = self.transform(A_image)
        B_image = self.transform(B_image)
        
        
        return {'A': A_image, 'B': B_image, 'base_name': os.path.basename(A_path)}
    def __len__(self):
        return max(len(self.files_A), len(self.files_B))
    
    def resize_and_pad(self, img, target_size=(256, 256), pad_value=0):
        h, w = img.shape
        target_h, target_w = target_size

        # Nếu ảnh lớn hơn target_size -> Resize
        if h > target_h or w > target_w:
            img = cv2.resize(img, (min(w, target_w), min(h, target_h)), interpolation=cv2.INTER_LINEAR)

        # Tạo ảnh mới có kích thước target_size và điền giá trị pad_value
        padded_img = np.full((target_h, target_w), pad_value, dtype=img.dtype)
        
        # Lấy kích thước mới của ảnh sau khi resize
        new_h, new_w = img.shape
        
        # Chèn ảnh vào giữa (align center)
        start_h = (target_h - new_h) // 2
        start_w = (target_w - new_w) // 2
        padded_img[start_h:start_h + new_h, start_w:start_w + new_w] = img

        return padded_img
    