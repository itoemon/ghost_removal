#!C:\ProgramData\Anaconda3
# coding: utf-8
import os
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
import torch
import numpy as np
import cv2
import random
from torchvision import transforms
import torchvision.transforms.functional as F
import torch.nn as nn
import pathlib

from util.util import gamma_correction, pil2cv, cv2pil, tensor2pil, pil2tensor
from util.clip_ghost import clip_ghost


# Original Dataset
class myDataset(Dataset):

    def __init__(self, img_dir, amount=100, test_mode=False, clip_size=500):
        super().__init__()
        self.img_dir   = img_dir
        self.transform = transforms.ToTensor()
        self.amount = amount
        self.test_mode = test_mode
        self.clip_size = clip_size
        print("dataset-loaded")

    def __getitem__(self, index):

        if self.test_mode :

            test, img_name = self.get_test_img(self.img_dir, index)

            input, frame, top, bottom, left, right = clip_ghost(pil2cv(test), self.clip_size, self.clip_size)

            input = cv2pil(input)
            frame = cv2pil(frame)

            if self.transform is not None:
                input = self.transform(input)
                frame = self.transform(frame)
            input = F.resize(img=input, size=(256, 256))
            # frame_size = frame.size()
            # frame = F.resize(img=frame, size=(int(frame_size[1] * 256 / 500), int((frame_size[2] * 256 / 500))))

            return input, frame, top, bottom, left, right, img_name

        else:

            ghost = self.get_img(self.img_dir, "ghost")
            scene = self.get_img(self.img_dir, "scene")

            if self.transform is not None:
                ghost = self.transform(ghost)
                scene = self.transform(scene)
            
            scene_size = random.randint(500, 700)
            RandomCrop = transforms.RandomCrop(scene_size)
            scene = RandomCrop(scene)
            
            ghost_height = ghost.size()[1]
            ghost_width = ghost.size()[2]

            dif_h = scene_size - ghost_height
            dif_w = scene_size - ghost_width

            padding_top = random.randint(0, dif_h)
            padding_bottom = dif_h - padding_top
            padding_right = random.randint(0, dif_w)
            padding_left = dif_w - padding_right

            ZeroPad = nn.ZeroPad2d((padding_left, padding_right, padding_top, padding_bottom))

            ghost = ZeroPad(ghost)

            ghost = F.resize(img=ghost, size=(256, 256))
            scene = F.resize(img=scene, size=(256, 256))

            mask = ghost.clone()

            luminance = 0.299 * mask[0] + 0.587 * mask[1] + 0.114 *mask[2]

            mask[0] = (luminance <= 0.08)*0 + (luminance > 0.08)
            mask[1] = (luminance <= 0.08)*0 + (luminance > 0.08)
            mask[2] = (luminance <= 0.08)*0 + (luminance > 0.08)

            
            # ghost_cv = pil2cv(tensor2pil(ghost))
            # ghost_gamma_cv = gamma_correction(ghost_cv , 0.8)
            # ghost_gamma = pil2tensor(cv2pil(ghost_gamma_cv))

            
            # input = ghost_gamma + scene
            input = ghost + scene
            input = torch.clamp(input, min=0, max=1)

            return input, ghost, scene, mask

    def __len__(self):
        return self.amount
    
    def get_img(self, img_dir, file_name):
        amount = len(list(Path(os.path.join(img_dir, file_name)).glob('**/*.png')))
        return Image.open(os.path.join(img_dir, file_name, file_name + f" ({str(random.randint(1, amount))}).png")).convert('RGB')
    
    def get_test_img(self, img_dir, index):
        img_list = list(pathlib.Path(os.path.join(img_dir, "test")).glob('**/*.png'))
        _, file_name = os.path.split(img_list[index])
        return Image.open(img_list[index]).convert('RGB'), file_name

    
### end class


