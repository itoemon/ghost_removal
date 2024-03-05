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
import cv2

from util.myDataset import myDataset
from network.UNet_h5 import UNet
from network.LightEstimator import LightEstimator


def main():

    dataset =  myDataset(
        img_dir=os.path.join(".", "data"),
        amount = 1,
    )

    input, ghost, scene, mask = dataset[0]

    
    xxx = ghost.size()
    ghosts = ghost.view(1, xxx[0], xxx[1], xxx[2]).to("cpu")
    xxx = mask.size()
    masks = mask.view(1, xxx[0], xxx[1], xxx[2]).to("cpu")


    showImg(input, "input")
    showImg(ghost, "ghost")
    showImg(scene, "scene")
    showImg(mask, "mask")

def showImg(img_tensor, img_name):
    img_tensor  = torch.round(img_tensor*255)
    img_tensor  = img_tensor.to(torch.uint8).detach().to("cpu")
    func = transforms.ToPILImage()
    img_pil = func(img_tensor)

    plt.imshow(img_pil)
    plt.show()
    
    img_dir = "./sample"
    os.makedirs(img_dir, exist_ok=True)
    img_pil.save(img_dir +"/"+ img_name + ".png")



if __name__ == "__main__":
    main()