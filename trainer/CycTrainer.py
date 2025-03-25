#!/usr/bin/python3

import argparse
import itertools
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
import os
from .utils import LambdaLR,Logger,ReplayBuffer
from .utils import weights_init_normal,get_config
from .datasets import ImageDataset,ValDataset
from Model.CycleGan import *
from .utils import Resize,ToTensor,smooothing_loss
from .utils import Logger
from .reg import Reg
from torchvision.transforms import RandomAffine,ToPILImage
from .transformer import Transformer_2D
from skimage import measure
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm

# use tensorboard 
from torch.utils.tensorboard import SummaryWriter

class Cyc_Trainer():
    def __init__(self, config):
        super().__init__()
        self.config = config
        ## def networks
        self.netG_A2B = Generator(config['input_nc'], config['output_nc']).cuda()
        self.netD_B = Discriminator(config['input_nc']).cuda()
        self.optimizer_D_B = torch.optim.Adam(self.netD_B.parameters(), lr=config['lr'], betas=(0.5, 0.999))
        
        if config['regist']:
            self.R_A = Reg(config['size'], config['size'],config['input_nc'],config['input_nc']).cuda()
            self.spatial_transform = Transformer_2D().cuda()
            self.optimizer_R_A = torch.optim.Adam(self.R_A.parameters(), lr=config['lr'], betas=(0.5, 0.999))
        if config['bidirect']:
            self.netG_B2A = Generator(config['input_nc'], config['output_nc']).cuda()
            self.netD_A = Discriminator(config['input_nc']).cuda()
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A2B.parameters(), self.netG_B2A.parameters()),lr=config['lr'], betas=(0.5, 0.999))
            self.optimizer_D_A = torch.optim.Adam(self.netD_A.parameters(), lr=config['lr'], betas=(0.5, 0.999))

        else:
            self.optimizer_G = torch.optim.Adam(self.netG_A2B.parameters(), lr=config['lr'], betas=(0.5, 0.999))
            

        # Lossess
        self.MSE_loss = torch.nn.MSELoss()
        self.L1_loss = torch.nn.L1Loss()

        # Inputs & targets memory allocation
        Tensor = torch.cuda.FloatTensor if config['cuda'] else torch.Tensor
        self.Tensor = torch.cuda.FloatTensor if config['cuda'] else torch.Tensor
        # self.input_A = Tensor(config['batchSize'], config['input_nc'], config['size'], config['size'])
        # self.input_B = Tensor(config['batchSize'], config['output_nc'], config['size'], config['size'])
        self.target_real = Variable(Tensor(config['batchSize'],1).fill_(1.0), requires_grad=False)
        self.target_fake = Variable(Tensor(config['batchSize'],1).fill_(0.0), requires_grad=False)


        self.fake_A_buffer = ReplayBuffer()
        self.fake_B_buffer = ReplayBuffer()

        #Dataset loader
        level = config['noise_level']  # set noise level
        

        self.dataloader = DataLoader(ImageDataset(config['dataroot']),
                                batch_size=config['batchSize'], shuffle=True, num_workers=config['n_cpu'], drop_last=True)

        val_transforms = [ToTensor(),
                          Resize(size_tuple = (config['size'], config['size']))]
        
        self.val_data = DataLoader(ValDataset(config['val_dataroot']),
                                batch_size=config['batchSize'], shuffle=False, num_workers=config['n_cpu'], drop_last=True)
        
        # where to log
        os.makedirs(os.path.dirname(self.config["val_log_path"]), exist_ok=True)

 
    #    # Loss plot  
        #  use tensorboard
        self.logger = SummaryWriter(log_dir = config['log_root'])
        os.makedirs(config['log_root'], exist_ok=True)    
        
    def train(self):
        ###### Training ######
        if not os.path.exists(self.config["save_root"]):
            os.makedirs(self.config["save_root"])
        for epoch in range(self.config['epoch'], self.config['n_epochs']):
            epoch_loss = 0
            tbar = tqdm(enumerate(self.dataloader), total=len(self.dataloader), desc=f"Epoch {epoch}/{self.config['n_epochs']}", leave=True)
            for i, batch in tbar:
                # Set model input
                real_A = batch['A'].cuda().float()  # Giả sử batch['A'] đã ở dạng tensor
                real_B = batch['B'].cuda().float()
                # regist
                self.optimizer_R_A.zero_grad()
                self.optimizer_G.zero_grad()
                #### regist sys loss
                fake_B = self.netG_A2B(real_A)
                Trans = self.R_A(fake_B,real_B) 
                SysRegist_A2B = self.spatial_transform(fake_B,Trans)
                SR_loss = self.config['Corr_lamda'] * self.L1_loss(SysRegist_A2B,real_B)###SR
                pred_fake0 = self.netD_B(fake_B)
                adv_loss = self.config['Adv_lamda'] * self.MSE_loss(pred_fake0, self.target_real)
                ####smooth loss
                SM_loss = self.config['Smooth_lamda'] * smooothing_loss(Trans)
                toal_loss = SM_loss+adv_loss+SR_loss
                toal_loss.backward()

                # Cleanup
                del fake_B, Trans, SysRegist_A2B, pred_fake0
                torch.cuda.empty_cache()

                epoch_loss += toal_loss.item()
                self.optimizer_R_A.step()
                self.optimizer_G.step()
                self.optimizer_D_B.zero_grad()

                with torch.no_grad():
                    fake_B = self.netG_A2B(real_A)
                pred_fake0 = self.netD_B(fake_B)
                pred_real = self.netD_B(real_B)
                loss_D_B = self.config['Adv_lamda'] * self.MSE_loss(pred_fake0, self.target_fake)+self.config['Adv_lamda'] * self.MSE_loss(pred_real, self.target_real)


                loss_D_B.backward()
                self.optimizer_D_B.step()

                # Cleanup tiếp
                del fake_B, pred_fake0, pred_real
                torch.cuda.empty_cache()
                             
                # log to tensorboard
                if i % 30 == 0:
                    with torch.no_grad():
                        step = epoch * len(self.dataloader) + i
                        self.logger.add_scalar('toal_loss', toal_loss.detach().item(), step)

            self.logger.add_scalar('epoch_loss', epoch_loss, epoch)
            tbar.close()
            print(f"Epoch {epoch}, epoch_loss: {epoch_loss}")
            # save for resume training
                # print(self.MAE(fake_B, real_B))
                # self.logger.log({'loss_D_B': loss_D_B,'SR_loss':SR_loss})
            
            # Save models checkpoints
            # if (epoch % 2 == 0):
            torch.save(self.netG_A2B.state_dict(), f"{self.config['save_root']}netG_A2B_epoch{epoch}.pth")
            
                
        self.logger.close()   

    def _3D_inference(self, patient_list, result_path):
        self.netG_A2B.load_state_dict(torch.load(self.config['save_root'] + 'netG_A2B_epoch3.pth'))

        def pad_to_4(img, pad_value=0): # img shape h x 256
            h, w = img.shape 
            if (h<256):
                pad_h = 256 - h
            else:
                pad_h = (4 - (h % 4)) % 4  # Số hàng cần padding để h chia hết cho 4
            padded_image = np.pad(img, ((0, pad_h), (0, 0)), mode='constant', constant_values=pad_value)
            return padded_image  

        
        def preprocess(ct_slice): # ct_voxel: W * H (512 * 512)
            transform = transforms.Compose([
            transforms.Resize(256),
            transforms.ToTensor()
            ])
            ct_slice = pad_to_4(ct_slice)
            ct_slice = (ct_slice - ct_slice.min())/(ct_slice.max() - ct_slice.min())
            ct_slice = (ct_slice - 0.5)*2
            #ct_slice = pad_to_same_size(ct_slice, pad_value=-1.0)
            
            
            A_image = Image.fromarray(ct_slice)
            A_image = transform(A_image)
           
            return A_image
        
        def postprocess(fake_B, max_pixel = 32767):
            fake_B = fake_B.detach().cpu().numpy().squeeze()  
            image = fake_B
            image = (image * 0.5 + 0.5).clip(0, 1)
            image = (image * max_pixel).clip(0, max_pixel)
            return image



        # Duyệt qua từng thư mục bệnh nhân trong DATA_PATH
        for patient_folder in tqdm(patient_list):
            patient_path = patient_folder
            
            # Kiểm tra xem có phải là thư mục không
            if os.path.isdir(patient_path):
                # Tìm file pet.npy bên trong thư mục bệnh nhân
                pet_file_path = os.path.join(patient_path, 'phase1_pet.npy')
                
                # Kiểm tra tệp có tồn tại hay không
                if os.path.exists(pet_file_path):
                    pet_img = np.load(pet_file_path, allow_pickle=True)
                    predicted_slices = []

                    # Lặp qua các lát cắt để dự đoán
                    
                    for i in tqdm(range(pet_img.shape[1])):
                        pet_slice = pet_img[:, i, :]
                        A_image = preprocess(pet_slice).unsqueeze(0)
                        #print(A_image.shape)
                        #print(A_image.unsqueeze(0).shape)
                        #print(A_image.max(), A_image.min())
                        input_A = self.Tensor(self.config['batchSize'], 1, A_image.shape[2], A_image.shape[3])
                        real_A = Variable(input_A.copy_(A_image))
                        fake_B = self.netG_A2B(real_A)
                        #print(fake_B.max(), fake_B.min())
                        fake_B = postprocess(fake_B)
                        predicted_slices.append(fake_B)
                    # Chuyển danh sách các lát cắt đã dự đoán thành một khối 3D numpy array
                    predicted_volume = np.stack(predicted_slices, axis=1)
                    print(predicted_volume.shape)

                    # Tạo thư mục kết quả riêng cho bệnh nhân nếu chưa tồn tại
                    patient_result_path = os.path.join(result_path, os.path.basename(patient_folder))
                    os.makedirs(patient_result_path, exist_ok=True)
                    # print(f"Saving result to {patient_result_path}")
                    # Lưu kết quả dự đoán vào thư mục của bệnh nhân
                    output_file_path = os.path.join(patient_result_path, 'coronal_pet_ep3.npy')
                    np.save(output_file_path, predicted_volume)
                    print(f"Saved predicted volume to {output_file_path}")      
                         
    def test(self,):
        self.netG_A2B.load_state_dict(torch.load(self.config['save_root'] + 'netG_A2B.pth'))
        #self.R_A.load_state_dict(torch.load(self.config['save_root'] + 'Regist.pth'))
        with torch.no_grad():
                MAE = 0
                PSNR = 0
                SSIM = 0
                num = 0
                for i, batch in enumerate(self.val_data):
                    real_A = Variable(self.input_A.copy_(batch['A']))
                    real_B = Variable(self.input_B.copy_(batch['B'])).detach().cpu().numpy().squeeze()
                    
                    fake_B = self.netG_A2B(real_A)
                    fake_B = fake_B.detach().cpu().numpy().squeeze()                                                 
                    mae = self.MAE(fake_B,real_B)
                    psnr = self.PSNR(fake_B,real_B)
                    ssim = measure.compare_ssim(fake_B,real_B)
                    MAE += mae
                    PSNR += psnr
                    SSIM += ssim 
                    num += 1
                print ('MAE:',MAE/num)
                print ('PSNR:',PSNR/num)
                print ('SSIM:',SSIM/num)
    
    def PSNR(self,fake,real):
       x,y = np.where(real!= -1)# Exclude background
       mse = np.mean(((fake[x][y]+1)/2. - (real[x][y]+1)/2.) ** 2 )
       if mse < 1.0e-10:
          return 100
       else:
           PIXEL_MAX = 1
           return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))
            
            
    # def MAE(self,fake,real):
    #     x,y = np.where(real!= -1)  # Exclude background
    #     mae = np.abs(fake[x,y]-real[x,y]).mean()
    #     return mae/2     #from (-1,1) normaliz  to (0,1)
    def MAE(self, fake, real):
        batch_size = real.shape[0]
        maes = []
        for i in range(batch_size):
            x, y = np.where(real[i] != -1)  # Xử lý từng ảnh trong batch
            mae = np.abs(fake[i, x, y] - real[i, x, y]).mean()
            maes.append(mae)
        return np.mean(maes) / 2  # Chuyển từ (-1,1) về (0,1)


    # def MAE(self, fake, real):
    #     mask = real != -1  # Tạo mask trên GPU
    #     mae = torch.abs(fake[mask] - real[mask]).mean()
    #     return (mae / 2).item()  # Chuyển về giá trị float


            

    def save_deformation(self,defms,root):
        heatmapshow = None
        defms_ = defms.data.cpu().float().numpy()
        dir_x = defms_[0]
        dir_y = defms_[1]
        x_max,x_min = dir_x.max(),dir_x.min()
        y_max,y_min = dir_y.max(),dir_y.min()
        dir_x = ((dir_x-x_min)/(x_max-x_min))*255
        dir_y = ((dir_y-y_min)/(y_max-y_min))*255
        tans_x = cv2.normalize(dir_x, heatmapshow, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        #tans_x[tans_x<=150] = 0
        tans_x = cv2.applyColorMap(tans_x, cv2.COLORMAP_JET)
        tans_y = cv2.normalize(dir_y, heatmapshow, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        #tans_y[tans_y<=150] = 0
        tans_y = cv2.applyColorMap(tans_y, cv2.COLORMAP_JET)
        gradxy = cv2.addWeighted(tans_x, 0.5,tans_y, 0.5, 0)

        cv2.imwrite(root, gradxy) 
