import os
import torch.utils.data as data
import numpy as np
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as tfs
from torchvision.transforms import functional as FF
import os,sys
import random
from scipy.linalg import orth
from PIL import Image
import cv2
random.seed(2)
np.random.seed(2)

def paired_random_crop(img_gts, img_lqs, gt_patch_size, scale, gt_path=None):
    # if gt_path == '../dataset/OTS/clear_images/2578.jpg':
    #     print('here')
    #     print(img_lqs.shape)
    #     print(img_gts.shape)
    """Paired random crop. Support Numpy array and Tensor inputs.

    It crops lists of lq and gt images with corresponding locations.

    Args:
        img_gts (list[ndarray] | ndarray | list[Tensor] | Tensor): GT images. Note that all images
            should have the same shape. If the input is an ndarray, it will
            be transformed to a list containing itself.
        img_lqs (list[ndarray] | ndarray): LQ images. Note that all images
            should have the same shape. If the input is an ndarray, it will
            be transformed to a list containing itself.
        gt_patch_size (int): GT patch size.
        scale (int): Scale factor.
        gt_path (str): Path to ground-truth. Default: None.

    Returns:
        list[ndarray] | ndarray: GT images and LQ images. If returned results
            only have one element, just return ndarray.
    """

    if not isinstance(img_gts, list):
        img_gts = [img_gts]
    if not isinstance(img_lqs, list):
        img_lqs = [img_lqs]

    # determine input type: Numpy array or Tensor
    input_type = 'Tensor' if torch.is_tensor(img_gts[0]) else 'Numpy'

    if input_type == 'Tensor':
        h_lq, w_lq = img_lqs[0].size()[-2:]
        h_gt, w_gt = img_gts[0].size()[-2:]
    else:
        h_lq, w_lq = img_lqs[0].shape[0:2]
        h_gt, w_gt = img_gts[0].shape[0:2]
    lq_patch_size = gt_patch_size // scale

    if h_gt != h_lq * scale or w_gt != w_lq * scale:
        raise ValueError(f'Scale mismatches. GT ({h_gt}, {w_gt}) is not {scale}x ',
                         f'multiplication of LQ ({h_lq}, {w_lq}).')
    if h_lq < lq_patch_size or w_lq < lq_patch_size:
        raise ValueError(f'LQ ({h_lq}, {w_lq}) is smaller than patch size '
                         f'({lq_patch_size}, {lq_patch_size}). '
                         f'Please remove {gt_path}.')

    # randomly choose top and left coordinates for lq patch
    top = random.randint(0, h_lq - lq_patch_size)
    left = random.randint(0, w_lq - lq_patch_size)

    # crop lq patch
    if input_type == 'Tensor':
        img_lqs = [v[:, :, top:top + lq_patch_size, left:left + lq_patch_size] for v in img_lqs]
    else:
        img_lqs = [v[top:top + lq_patch_size, left:left + lq_patch_size, ...] for v in img_lqs]

    # crop corresponding gt patch
    top_gt, left_gt = int(top * scale), int(left * scale)
    if input_type == 'Tensor':
        img_gts = [v[:, :, top_gt:top_gt + gt_patch_size, left_gt:left_gt + gt_patch_size] for v in img_gts]
    else:
        img_gts = [v[top_gt:top_gt + gt_patch_size, left_gt:left_gt + gt_patch_size, ...] for v in img_gts]
    if len(img_gts) == 1:
        img_gts = img_gts[0]
    if len(img_lqs) == 1:
        img_lqs = img_lqs[0]
    return img_gts, img_lqs



def img2tensor(imgs, bgr2rgb=True, float32=True):
    """Numpy array to tensor.

    Args:
        imgs (list[ndarray] | ndarray): Input images.
        bgr2rgb (bool): Whether to change bgr to rgb.
        float32 (bool): Whether to change to float32.

    Returns:
        list[tensor] | tensor: Tensor images. If returned results only have
            one element, just return tensor.
    """

    def _totensor(img, bgr2rgb, float32):
        if img.shape[2] == 3 and bgr2rgb:
            if img.dtype == 'float64':
                img = img.astype('float32')
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img.transpose(2, 0, 1))
        if float32:
            img = img.float()
        return img

    if isinstance(imgs, list):
        return [_totensor(img, bgr2rgb, float32) for img in imgs]
    else:
        return _totensor(imgs, bgr2rgb, float32)

def uint2single(img):
    return np.float32(img/255.)

def single2uint(img):
    return np.uint8((img.clip(0, 1)*255.).round())
  
def add_Gaussian_noise(img, noise_level1=2, noise_level2=25):
    noise_level = random.randint(noise_level1, noise_level2)
    rnum = np.random.rand()
    if rnum > 0.6:   # add color Gaussian noise
        img += np.random.normal(0, noise_level/255.0, img.shape).astype(np.float32)
    elif rnum < 0.4: # add grayscale Gaussian noise
        img += np.random.normal(0, noise_level/255.0, (*img.shape[:2], 1)).astype(np.float32)
    else:            # add  noise
        L = noise_level2/255.
        D = np.diag(np.random.rand(3))
        U = orth(np.random.rand(3,3))
        conv = np.dot(np.dot(np.transpose(U), D), U)
        img += np.random.multivariate_normal([0,0,0], np.abs(L**2*conv), img.shape[:2]).astype(np.float32)
    img = np.clip(img, 0.0, 1.0)
    return img

def add_JPEG_noise(img):
    quality_factor = random.randint(30, 95)
    img = cv2.cvtColor(single2uint(img), cv2.COLOR_RGB2BGR)
    result, encimg = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), quality_factor])
    img = cv2.imdecode(encimg, 1)
    img = cv2.cvtColor(uint2single(img), cv2.COLOR_BGR2RGB)
    return img

import torch
def Synthetic_Deg(input):
    # adjust luminance
    img_lq = np.array(input)
    #if np.random.rand(1) < 0.8:
    img_lq = np.power(img_lq, np.random.rand(1) * 1.5 + 1.5)
    # add gaussian noise
    #if np.random.rand(1) < 0.8:
    img_lq = add_Gaussian_noise(img_lq)

    # add JPEG noise
    #if np.random.rand(1) < 0.8:
    img_lq = add_JPEG_noise(img_lq)
    
    return img_lq

class AllWeather(data.Dataset):
    def __init__(self, path, train, size=240, format='.png', crop=True):
        super(AllWeather, self).__init__()
        self.size = size
        self.train = train
        self.format = format
        self.crop = crop
        
        self.haze_imgs_dir = os.listdir(os.path.join(path, 'input'))
        self.haze_imgs = [os.path.join(path, 'input', img) for img in self.haze_imgs_dir]
        self.clear_dir = os.path.join(path, 'gt')
        
        print(f'Crop size: {size}')

    def __getitem__(self, index):
        haze_path = self.haze_imgs[index]
        haze_img = self._load_image(haze_path)
        clear_img = self._load_clear_image(haze_path)
        
        if self.crop and not isinstance(self.size, str):
            haze_img, clear_img = self._crop_images(haze_img, clear_img)
        
        haze_img, clear_img = self._normalize_images(haze_img, clear_img)
        
        haze_tensor, clear_tensor = img2tensor([haze_img, clear_img], bgr2rgb=True, float32=True)
        
        if self.train:
            haze_img, clear_img = self.augData(haze_tensor, clear_tensor)
        
        id = haze_path.split('/')[-1].split('.')[0]
        return haze_tensor, clear_tensor, id

    def _load_image(self, img_path):
        img = cv2.imread(img_path)
        return Image.fromarray(img)

    def _load_clear_image(self, haze_path):
        haze_name = haze_path.split('/')[-1]
        clear_name = haze_name
        
        clear_path = os.path.join(self.clear_dir, clear_name)
        try:
            clear_img = cv2.imread(clear_path)
            return Image.fromarray(clear_img)
        except Exception as e:
            print(f'Error loading clear image: {clear_name}, {haze_path}')
            raise e

    def _crop_images(self, haze_img, clear_img):
        i, j, h, w = tfs.RandomCrop.get_params(haze_img, output_size=(self.size, self.size))
        haze_img = FF.crop(haze_img, i, j, h, w)
        clear_img = FF.crop(clear_img, i, j, h, w)
        return haze_img, clear_img

    def _normalize_images(self, haze_img, clear_img):
        haze_img = np.array(haze_img).astype(np.float32) / 255.0
        clear_img = np.array(clear_img).astype(np.float32) / 255.0
        return haze_img, clear_img

    def augData(self, data, target):
        rand_hor = random.randint(0, 1)
        rand_rot = random.randint(0, 3)
        
        data = tfs.RandomHorizontalFlip(rand_hor)(data)
        target = tfs.RandomHorizontalFlip(rand_hor)(target)
        
        if rand_rot:
            data = FF.rotate(data, 90 * rand_rot)
            target = FF.rotate(target, 90 * rand_rot)
        
        return data, target
    def __len__(self):
        return int(len(self.haze_imgs))

class Test1(data.Dataset):
    def __init__(self, path, train, size=240, format='.png', crop=True):
        super(Test1, self).__init__()
        self.size = size
        self.train = train
        self.format = format
        self.crop = crop
        
        self.haze_imgs_dir = os.listdir(os.path.join(path, 'in1'))
        self.haze_imgs = [os.path.join(path, 'in1', img) for img in self.haze_imgs_dir]
        self.clear_dir = os.path.join(path, 'gt1')
        
        print(f'Crop size: {size}')

    def __getitem__(self, index):
        haze_path = self.haze_imgs[index]
        haze_img = self._load_image(haze_path)
        clear_img = self._load_clear_image(haze_path)
        
        if self.crop and not isinstance(self.size, str):
            haze_img, clear_img = self._crop_images(haze_img, clear_img)
        
        haze_img, clear_img = self._normalize_images(haze_img, clear_img)
        
        haze_tensor, clear_tensor = img2tensor([haze_img, clear_img], bgr2rgb=True, float32=True)
        
        if self.train:
            haze_img, clear_img = self.augData(haze_tensor, clear_tensor)
        
        return haze_tensor, clear_tensor, haze_path.split('/')[-1].split('.')[0]

    def _load_image(self, img_path):
        img = cv2.imread(img_path)
        return Image.fromarray(img)

    def _load_clear_image(self, haze_path):
        haze_name = haze_path.split('/')[-1]
        name_syn = haze_name.split('_')[1]
        clear_name = f'im_{name_syn}.png'
        
        clear_path = os.path.join(self.clear_dir, clear_name)
        try:
            clear_img = cv2.imread(clear_path)
            return Image.fromarray(clear_img)
        except Exception as e:
            print(f'Error loading clear image: {clear_name}, {haze_path}, {name_syn}')
            raise e

    def _crop_images(self, haze_img, clear_img):
        i, j, h, w = tfs.RandomCrop.get_params(haze_img, output_size=(self.size, self.size))
        haze_img = FF.crop(haze_img, i, j, h, w)
        clear_img = FF.crop(clear_img, i, j, h, w)
        return haze_img, clear_img

    def _normalize_images(self, haze_img, clear_img):
        haze_img = np.array(haze_img).astype(np.float32) / 255.0
        clear_img = np.array(clear_img).astype(np.float32) / 255.0
        return haze_img, clear_img

    def augData(self, data, target):
        rand_hor = random.randint(0, 1)
        rand_rot = random.randint(0, 3)
        
        data = tfs.RandomHorizontalFlip(rand_hor)(data)
        target = tfs.RandomHorizontalFlip(rand_hor)(target)
        
        if rand_rot:
            data=FF.rotate(data,90*rand_rot)
            target=FF.rotate(target,90*rand_rot)

        return  data , target
    def __len__(self):
        return len(self.haze_imgs)


class Snow100kTest(data.Dataset):
    def __init__(self, path, train, size=240, format='.png', crop=True):
        super(Snow100kTest, self).__init__()
        self.size = size
        self.train = train
        self.format = format
        self.crop = crop
        
        self.haze_imgs_dir = os.listdir(os.path.join(path, 'synthetic'))
        self.haze_imgs = [os.path.join(path, 'synthetic', img) for img in self.haze_imgs_dir]
        self.clear_dir = os.path.join(path, 'gt')
        
        print(f'Crop size: {size}')

    def __getitem__(self, index):
        haze_path = self.haze_imgs[index]
        haze_img = self._load_image(haze_path)
        clear_img = self._load_clear_image(haze_path)
        
        if self.crop and not isinstance(self.size, str):
            haze_img, clear_img = self._crop_images(haze_img, clear_img)
        
        haze_img, clear_img = self._normalize_images(haze_img, clear_img)
        
        haze_tensor, clear_tensor = img2tensor([haze_img, clear_img], bgr2rgb=True, float32=True)
        
        if self.train:
            haze_img, clear_img = self.augData(haze_tensor, clear_tensor)
        
        id = haze_path.split('/')[-1].split('.')[0]
        return haze_tensor, clear_tensor, id

    def _load_image(self, img_path):
        img = cv2.imread(img_path)
        return Image.fromarray(img)

    def _load_clear_image(self, haze_path):
        haze_name = haze_path.split('/')[-1].split('.')[0]
        clear_name = haze_name + '.jpg'
        
        clear_path = os.path.join(self.clear_dir, clear_name)
        try:
            clear_img = cv2.imread(clear_path)
            return Image.fromarray(clear_img)
        except Exception as e:
            print(f'Error loading clear image: {clear_name}, {haze_path}')
            raise e

    def _crop_images(self, haze_img, clear_img):
        i, j, h, w = tfs.RandomCrop.get_params(haze_img, output_size=(self.size, self.size))
        haze_img = FF.crop(haze_img, i, j, h, w)
        clear_img = FF.crop(clear_img, i, j, h, w)
        return haze_img, clear_img

    def _normalize_images(self, haze_img, clear_img):
        haze_img = np.array(haze_img).astype(np.float32) / 255.0
        clear_img = np.array(clear_img).astype(np.float32) / 255.0
        return haze_img, clear_img

    def augData(self, data, target):
        rand_hor = random.randint(0, 1)
        rand_rot = random.randint(0, 3)
        
        data = tfs.RandomHorizontalFlip(rand_hor)(data)
        target = tfs.RandomHorizontalFlip(rand_hor)(target)
        
        if rand_rot:
            data = FF.rotate(data, 90 * rand_rot)
            target = FF.rotate(target,90*rand_rot)

        return  data , target
    def __len__(self):
        return len(self.haze_imgs)
    
class Real(data.Dataset):
    def __init__(self, path, train, size=240, format='.png', crop=True):
        super(Real, self).__init__()
        self.size = size
        self.train = train
        self.format = format
        self.crop = crop
        
        self.haze_imgs = self._load_images(path)
        
    def _load_images(self, path):
        haze_imgs_dir = os.listdir(path)
        return [os.path.join(path, img) for img in haze_imgs_dir]
    
    def __getitem__(self, index):
        haze = self._load_and_preprocess_haze_image(index)
        id = self._extract_image_id(index)
        
        return haze, id
    
    def _load_and_preprocess_haze_image(self, index):
        haze = cv2.imread(self.haze_imgs[index])
        haze = Image.fromarray(haze)
        
        if self.crop:
            haze = self._random_crop(haze)
        
        haze = np.array(haze).astype(np.float32) / 255.0
        haze = img2tensor(haze, bgr2rgb=True, float32=True)
        haze = self.augData(haze)
        
        return haze
    
    def _random_crop(self, image):
        i, j, h, w = tfs.RandomCrop.get_params(image, output_size=(self.size, self.size))
        return FF.crop(image, i, j, h, w)
    
    def _extract_image_id(self, index):
        img = self.haze_imgs[index]
        name_syn = img.split('/')[-1].split('.')[0]
        return name_syn
    
    def augData(self, data):
        if self.train:
            data = self._random_horizontal_flip(data)
            data = self._random_rotate(data)
        
        return data
    
    def _random_horizontal_flip(self, data):
        rand_hor = random.randint(0, 1)
        return tfs.RandomHorizontalFlip(rand_hor)(data)
    
    def _random_rotate(self, data):
        rand_rot = random.randint(0, 3)
        if rand_rot:
            data = FF.rotate(data, 90 * rand_rot)
        return data
    
    def __len__(self):
        return len(self.haze_imgs)
    
