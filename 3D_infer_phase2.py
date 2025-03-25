#!/usr/bin/python3

import argparse
import itertools
import os
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
from PIL import Image
import torch
import torch.nn.functional as F
import numpy as np
from trainer import CycTrainer,P2p_Trainer,Nice_Trainer
import yaml


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def get_config(config):
    with open(config, 'r') as stream:
        return yaml.safe_load(stream)
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='Yaml/CycleGan.yaml', help='Path to the config file.')
    opts = parser.parse_args()
    config = get_config(opts.config)
    
    if config['name'] == 'CycleGan':
        trainer = CycTrainer.Cyc_Trainer(config)

    elif config['name'] == 'P2p':
        trainer = P2p_Trainer(config)
    txt_list_patients = '/home/PET-CT/hachi/Reg-GAN/data.txt'
    # read files 
    with open(txt_list_patients) as f:
        lines = f.readlines()
        patient_list = [x.strip() for x in lines]
    out_inperences = '/home/PET-CT/huutien/Reg-GAN/hachi/data_dienbien/pet_npy/20095316797'
    trainer._3D_inference(patient_list, out_inperences)
    
    



###################################
if __name__ == '__main__':
    main()