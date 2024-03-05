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
import pathlib
import torchvision.transforms.functional as F

from util.myDataset import myDataset
from network.UNet_h5 import UNet
from network.LightEstimator import LightEstimator
from network.LRM import LRM
from network.MPRNet import MPRNet

from util.util import cv2pil, pil2cv, tensor2pil, pil2tensor

def main(i):
    device = "cpu"

    epoch = 49
    folder = "LRM"
    # model = UNet().to(device)
    # model = LightEstimator().to(device)
    model = LRM(device).to(device)
    # model = MPRNet().to(device)

    param = os.path.join(".", "param", folder,"param_epoch"+str(epoch)+".pth")
    model.load_state_dict(torch.load(param, map_location=device), strict=False)
    model.eval()

    test_mode = True
    clip_size = 500
    dataset =  myDataset(
        img_dir = os.path.join(".", "data"),
        amount = 1,
        test_mode = test_mode,
        clip_size = clip_size
    )

    if test_mode :
        input, frame, top, bottom, left, right, img_name = dataset[i]
    else :
        input, _, _, _ = dataset[0]

    xxx = input.size()
    inputs = input.view(1, xxx[0], xxx[1], xxx[2]).to(device)

    with torch.no_grad():
        # outputs = model(inputs)
        # ghost_ests = torch.clamp(outputs, min=0, max=1)
        # scene_ests = torch.clamp(inputs - ghost_ests, min=0, max=1)

        #LRM
        b, _, h, w = inputs.shape
        h0 = Variable(torch.zeros(b, 64, h, w, device=device))
        c0 = Variable(torch.zeros(b, 64, h, w, device=device))
        fake_outputs = inputs.clone().detach()
        h1, c1, _, _, _, _, scene_ests01 = model(inputs, fake_outputs, h0, c0)
        h2, c2, _, _, _, _, scene_ests02 = model(inputs, scene_ests01, h1, c1)
        _, _, _, ghost_ests, _, _, scene_ests = model(inputs, scene_ests02, h2, c2)
        ghost_ests = torch.clamp(ghost_ests, min=0, max=1)
        scene_ests = torch.clamp(scene_ests, min=0, max=1)

        # #MPR
        # scene_ests, _, _ = model(inputs)
        # scene_ests = torch.clamp(scene_ests, min=0, max=1)
        # ghost_ests = torch.clamp(inputs - scene_ests, min=0, max=1)


    if test_mode :
        saveImg(inputs[0], folder=folder + "/input", img_name=img_name)
        saveImg(scene_ests[0], folder=folder + "/scene_est", img_name=img_name)
        saveImg(ghost_ests[0], folder=folder + "/ghost_est", img_name=img_name)

        scene_est = F.resize(img=scene_ests[0], size=(clip_size, clip_size))
        scene_est_pad = pil2tensor(cv2pil(cv2.copyMakeBorder(
            pil2cv(tensor2pil(scene_est)), top, bottom, left, right, cv2.BORDER_CONSTANT, (0,0,0)
            )))
        scene_est = scene_est_pad + frame

        ghost_est = F.resize(img=ghost_ests[0], size=(clip_size, clip_size))
        ghost_est_pad = pil2tensor(cv2pil(cv2.copyMakeBorder(
            pil2cv(tensor2pil(ghost_est)), top, bottom, left, right, cv2.BORDER_CONSTANT, (0,0,0)
            )))
        
        saveImg(scene_est, folder=folder + "/output_scene", img_name=img_name)
        saveImg(ghost_est_pad, folder=folder + "/output_ghost", img_name=img_name)
    
    else :
        saveImg(inputs[0], folder=folder + "/input", img_name="input (" + str(i) + ").png")
        saveImg(scene_ests[0], folder=folder + "/scene_est", img_name="scene_est (" + str(i) + ").png")
        saveImg(ghost_ests[0], folder=folder + "/ghost_est", img_name="ghost_est (" + str(i) + ").png")


def saveImg(img_tensor, folder, img_name):
    img_dir = os.path.join(".", "test_results", folder)
    img_tensor  = torch.round(img_tensor*255)
    img_tensor  = img_tensor.to(torch.uint8).detach().to("cpu")
    func = transforms.ToPILImage()
    img_pil = func(img_tensor)
    os.makedirs(img_dir, exist_ok=True)
    img_pil.save(img_dir +"/"+ img_name)
    

def showImg(img_tensor):
    img_tensor  = torch.round(img_tensor*255)
    img_tensor  = img_tensor.to(torch.uint8).detach().to("cpu")
    func = transforms.ToPILImage()
    img_pil = func(img_tensor)
    plt.imshow(img_pil)
    plt.show()



if __name__ == "__main__":

    img_list = list(pathlib.Path(os.path.join("data", "test")).glob('**/*.png'))
    loop = len(img_list)
    # loop = 1

    for i in range(0, loop):
        main(i)