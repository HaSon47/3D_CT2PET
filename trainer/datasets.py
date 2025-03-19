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

        # # Resize
        # A_image = self.pad_to_same_size(A_image)
        # B_image = self.pad_to_same_size(B_image)
        
        A_image = Image.fromarray(A_image)
        B_image = Image.fromarray(B_image)
        A_image = self.transform(A_image)
        B_image = self.transform(B_image)
        
        return {'A': A_image, 'B': B_image}
    
    def __len__(self):
        return max(len(self.files_A), len(self.files_B))
   
    # def pad_to_same_size(self, img, target_size=512, pad_value=0):
        
    #     h, w = img.shape
    #     new_size = max(target_size, h,w)

    #     # pad to all have new_size
    #     padded_img = np.full((new_size, new_size), pad_value, dtype=img.dtype)
    #     # align center
    #     start_h = (new_size - h)//2
    #     start_w = (new_size - w)//2
    #     padded_img[start_h:start_h + h, start_w: start_w + w] = img
        
    #     return padded_img   


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
        
            
        # Min-max normalization
        A_image = (A_image - np.min(A_image)) / (np.max(A_image) - np.min(A_image))
        B_image = (B_image - np.min(B_image)) / (np.max(B_image) - np.min(B_image))
        
        # Scale to (-1, 1)
        A_image = (A_image * 2.0) - 1.0
        B_image = (B_image * 2.0) - 1.0

        # # Resize
        # A_image = self.pad_to_same_size(A_image)
        # B_image = self.pad_to_same_size(B_image)
        
        A_image = Image.fromarray(A_image)
        B_image = Image.fromarray(B_image)
        A_image = self.transform(A_image)
        B_image = self.transform(B_image)
        
        
        return {'A': A_image, 'B': B_image, 'base_name': os.path.basename(A_path)}
    def __len__(self):
        return max(len(self.files_A), len(self.files_B))
    
    # def pad_to_same_size(self, img, target_size=512, pad_value=0):
        
    #     h, w = img.shape
    #     new_size = max(target_size, h,w)

    #     # pad to all have new_size
    #     padded_img = np.full((new_size, new_size), pad_value, dtype=img.dtype)
    #     # align center
    #     start_h = (new_size - h)//2
    #     start_w = (new_size - w)//2
    #     padded_img[start_h:start_h + h, start_w: start_w + w] = img
        
    #     return padded_img   
    