from ast import Param
import os
from pathlib import Path
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable
import pathlib

import cv2


def main():
    img_list = list(pathlib.Path(os.path.join("data", "test")).glob('**/*.png'))
    print(img_list[5])

    img = [ [ [255,   0,   0], [  0,   0,   0], [255,   0,   0] ],
            [ [  0,   0,   0], [255, 255, 255], [  0,   0,   0] ],
            [ [255, 255, 255], [  0,   0,   0], [255, 255, 255] ], ]
    
    # im2 = [ [ [  0,   0,   0], [   0,   0,   0], [255,   0,   0] ],
    #         [ [  0,   0,   0], [ 255, 255, 255], [  0,   0,   0] ],
    #         [ [255, 255, 255], [   0,   0,   0], [255, 255, 255] ], ]
    
    np_img = np.uint8(np.array(img))

    # np_im1 = np.int16(np.array(im1))
    # np_im2 = np.int16(np.array(im2))
    # np_img = np.uint8(np.clip(np_im1 + np_im2 , 0 , 255))

    ToTensor = transforms.ToTensor()
    tensor_img = ToTensor(np_img)
    np_img = np.uint8(torch.permute(tensor_img, (1, 2, 0)).to('cpu').detach().numpy().copy() * 255)
    plt.imshow(tensor_img)
    plt.show()

    print(np_img)
    # print(type(np_img))
    # print(np_img.dtype)
    plt.imshow(np_img)
    plt.show()

    pil_img = Image.fromarray(np_img)
    plt.imshow(pil_img)
    plt.show()
    # pil_img.save('sample.png')

    cv_img = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)
    plt.imshow(cv_img)
    plt.show()

if __name__ == "__main__":
    main()