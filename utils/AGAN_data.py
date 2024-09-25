import os
import random
import numpy as np
import torchvision.transforms as tfs
from torchvision.utils import make_grid
from torchvision.transforms import functional as FF
import torch.utils.data as data
from PIL import Image
random.seed(1143)

class AGAN_Dataset(data.Dataset):
    def __init__(self, path, train=False, size=256, format='.png', rand_inpaint=False, crop=True):
        super(AGAN_Dataset, self).__init__()
        self.size = size
        self.rand_inpaint = rand_inpaint
        self.InpaintSize = 64
        self.crop = crop
        print('crop size', size)
        self.train = train
        self.format = format
        self.haze_imgs_dir = os.listdir(os.path.join(path, 'data'))
        print('======>total number for training:', len(self.haze_imgs_dir))
        self.haze_imgs = [os.path.join(path, 'data', img) for img in self.haze_imgs_dir]
        self.clear_dir = os.path.join(path, 'gt')

    def __getitem__(self, index):
        haze = Image.open(self.haze_imgs[index])
        self.format = self.haze_imgs[index].split('/')[-1].split(".")[-1]
        
        # Ensure the image is large enough for cropping
        while haze.size[0] < self.size or haze.size[1] < self.size:
            if isinstance(self.size, int):
                index = random.randint(0, len(self.haze_imgs) - 1)
                haze = Image.open(self.haze_imgs[index])
        
        img = self.haze_imgs[index]
        id = img.split('/')[-1].split("_")[0]
        clear_name = id + '_clean' + '.' + self.format
        clear = Image.open(os.path.join(self.clear_dir, clear_name))
        clear = tfs.CenterCrop(haze.size[::-1])(clear)
        
        if self.crop and not isinstance(self.size, str):
            i, j, h, w = tfs.RandomCrop.get_params(haze, output_size=(self.size, self.size))
            haze = FF.crop(haze, i, j, h, w)
            clear = FF.crop(clear, i, j, h, w)
        
        haze, clear = self.augData(haze.convert("RGB"), clear.convert("RGB"))
        
        return haze, clear, id

    def augData(self, data, target):
        if self.train:
            rand_hor = random.randint(0, 1)
            rand_rot = random.randint(0, 3)
            
            data = tfs.RandomHorizontalFlip(rand_hor)(data)
            target = tfs.RandomHorizontalFlip(rand_hor)(target)
            
            if rand_rot:
                data = FF.rotate(data, 90 * rand_rot)
                target = FF.rotate(target, 90 * rand_rot)
        
        data = tfs.ToTensor()(data)
        target = tfs.ToTensor()(target)
        
        return data, target

    def __len__(self):
        return len(self.haze_imgs)
    
