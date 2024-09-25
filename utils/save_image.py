import os
import cv2
import torch
from torchvision.utils import save_image
import torchvision.transforms.functional as F
import numpy as np

def save_colormapped_image(tensor, filename, colormap=cv2.COLORMAP_JET):
    # ensure tensor is in CPU and detach it from the computation graph
    np_image = tensor.cpu().detach().numpy()
    # normalize the image to 0-255
    np_image -= np_image.min()
    np_image /= np_image.max()
    np_image *= 255.0
    np_image = np_image.astype(np.uint8)
    
    # apply the colormap
    colored_images = []
    for i in range(np_image.shape[0]):
        # reshape the image to have the channel as the last dimension
        image = np.transpose(np_image[i], (1, 2, 0))
        # if it's a single-channel image, use that channel; otherwise, convert to grayscale
        if image.shape[2] == 1:  # Single channel
            gray_image = image[:, :, 0]
        else:  #convert multi-channel image to single-channel grayscale
            gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        colored_image = cv2.applyColorMap(gray_image, colormap)
        colored_images.append(colored_image)
    
    # convert list of arrays to 4D numpy array (B, H, W, C)
    colored_images = np.stack(colored_images)
    
    # convert BGR to RGB (OpenCV to matplotlib conversion) and make a copy of the array
    colored_images = colored_images[..., ::-1].copy()

    # convert to tensor
    colored_tensor = torch.from_numpy(colored_images).permute(0, 3, 1, 2)
    
    # cave the image using torchvision (assumes 'colored_tensor' is a 4D tensor)
    save_image(colored_tensor.float() / 255.0, filename, nrow=1, normalize=False)
