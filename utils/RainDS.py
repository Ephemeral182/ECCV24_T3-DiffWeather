from torch.utils.data import Dataset
from PIL import Image
import torch
import os
import random
import numpy as np
import cv2
import random
from torchvision.transforms import functional as F
import torchvision.transforms as tfs
from doctest import FAIL_FAST
import os
import sys

import torch
import torch.utils.data as data
import numpy as np
from PIL import Image
import glob
import cv2
import torch.utils.data as data
import torchvision.transforms as tfs
from torchvision.transforms import functional as FF
import torchvision
import os,sys
import random
from PIL import Image
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from torchvision.utils import make_grid
random.seed(1143)


class RainDS_Dataset(data.Dataset):
    def __init__(self,path,train,size=240,format='.png',dataset_type='all'):
        super(RainDS_Dataset,self).__init__()
        self.size=size
        print('crop size',size)
        self.train=train
        self.format=format
        
        dir_tmp = 'train' if self.train else 'test'
        
        self.gt_path = os.path.join(path,dir_tmp,'gt')
        
        self.gt_list = []
        self.rain_list = []
        
        raindrop_path = os.path.join(path,dir_tmp,'raindrop')
        rainstreak_path = os.path.join(path,dir_tmp,'rainstreak')
        streak_drop_path = os.path.join(path,dir_tmp,'rainstreak_raindrop')
        
        raindrop_names = os.listdir(raindrop_path)
        rainstreak_names = os.listdir(rainstreak_path)
        streak_drop_names = os.listdir(streak_drop_path)
        
        rd_input = []
        rd_gt = []
        
        rs_input = []
        rs_gt = []
        
        rd_rs_input=[]
        rd_rs_gt = []
        
        for name in raindrop_names:
            rd_input.append(os.path.join(raindrop_path,name))
            gt_name = name.replace('rd','norain')
            rd_gt.append(os.path.join(self.gt_path,gt_name))
            
        for name in rainstreak_names:
            rs_input.append(os.path.join(rainstreak_path,name))
            gt_name = name.replace('rain','norain')
            rs_gt.append(os.path.join(self.gt_path,gt_name))
            
        for name in streak_drop_names:
            rd_rs_input.append(os.path.join(streak_drop_path,name))
            gt_name = name.replace('rd-rain','norain')
            rd_rs_gt.append(os.path.join(self.gt_path,gt_name))
        
        
        if dataset_type=='all':
            self.gt_list += rd_gt
            self.rain_list += rd_input
            self.gt_list += rs_gt
            self.rain_list += rs_input
            self.gt_list += rd_rs_gt
            self.rain_list += rd_rs_input
        elif dataset_type=='rs':
            self.gt_list += rs_gt
            self.rain_list += rs_input           
        elif dataset_type=='rd':
            self.gt_list += rd_gt
            self.rain_list += rd_input
        elif dataset_type=='rsrd':
            self.gt_list += rd_rs_gt
            self.rain_list += rd_rs_input
                      
    def __getitem__(self, index):
        rain=Image.open(self.rain_list[index])
        clear_path = self.gt_list[index]
        clear=Image.open(clear_path)
    
        if not isinstance(self.size,str):
            i,j,h,w=tfs.RandomCrop.get_params(clear,output_size=(self.size,self.size))
            clear=FF.crop(clear,i,j,h,w)
            rain = FF.crop(rain,i,j,h,w)
        clear_blur = tfs.GaussianBlur(7,(0.1,2))(clear)
        if self.train:
            rain,clear,clear_blur =self.augData(rain.convert("RGB") ,clear.convert("RGB"), clear_blur.convert("RGB"))
        else:
            rain=tfs.ToTensor()(rain.convert("RGB"))
            clear=tfs.ToTensor()(clear.convert("RGB")) 
            clear_blur=tfs.ToTensor()(clear_blur.convert("RGB"))           
        return rain,clear#torch.cat([rain,clear],dim=0),clear_blur

    def augData(self,data,target,target_blur):
        rand_hor=random.randint(0,1)
        rand_rot=random.randint(0,3)
        data=tfs.RandomHorizontalFlip(rand_hor)(data)
        target=tfs.RandomHorizontalFlip(rand_hor)(target)
        if rand_rot:
            data=FF.rotate(data,90*rand_rot)
            target=FF.rotate(target,90*rand_rot)
        data=tfs.ToTensor()(data)
        target=tfs.ToTensor()(target)
        target_blur=tfs.ToTensor()(target_blur)
        return data,target,target_blur
    def __len__(self):
        return len(self.rain_list)
