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

from util.GradWarmup_scheduler import GradualWarmupScheduler
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR

from util.SSIMLoss import SSIMLoss as nn_SSIMLoss
from util.PerceptualLossVGG16 import PerceptualLossVGG16 as nn_VGGLoss
from util.CriterionLosses import CharbonnierLoss as nn_CharLoss
from util.CriterionLosses import EdgeLoss as nn_EdgeLoss

from network.UNet_h5 import UNet
from network.LightEstimator import LightEstimator
from network.LRM import LRM
from network.MPRNet import MPRNet

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_size = 6
    epochs = 50
    learning_rate = 2e-4
 
    Parallel = False
    device_ids = [0,1]

    # _model = LightEstimator().to(device)
    # _model = UNet().to(device)
    _model = LRM(device).to(device)
    # _model = MPRNet().to(device)

    if Parallel :
        print("-- parallel:true --")
        model = torch.nn.DataParallel(_model, device_ids=device_ids)
    else:
        print("-- parallel:false --")
        model = _model

    load_dir = "./param/LRM/model_original.pth"
    model.load_state_dict(torch.load(load_dir))

    optimizer_model = optim.Adam(
        model.parameters(),
        lr=learning_rate,
        betas=[0.5, 0.999]
    )

    scheduler_cosine = CosineAnnealingLR(optimizer_model, epochs-3, eta_min=1e-6)
    scheduler_model = GradualWarmupScheduler(optimizer_model, multiplier=1, total_epoch=3, after_scheduler=scheduler_cosine)


    dataset =  myDataset(
        img_dir=os.path.join(".", "data"),
        amount = batch_size * 150
    )

    dataloader = DataLoader(
        dataset, batch_size=batch_size, 
        shuffle=True, drop_last=True
    )

    MSELoss = nn.MSELoss().to(device)
    L1Loss = nn.L1Loss().to(device)
    SSIMLoss = nn_SSIMLoss().to(device)
    VGGLoss = nn_VGGLoss().to(device)
    CharLoss = nn_CharLoss().to(device)
    EdgeLoss = nn_EdgeLoss(device).to(device)

    for epoch in range(epochs):
        # train(model, device, epoch, dataloader, optimizer_model, L1Loss, VGGLoss)
        LRM_train(model, device, epoch, dataloader, optimizer_model, MSELoss, L1Loss, SSIMLoss, VGGLoss)
        # MPR_train(model, device, epoch, dataloader, optimizer_model, CharLoss, EdgeLoss, L1Loss, VGGLoss)
        scheduler_model.step()
    
        if Parallel :
            torch.save(
                model.module.state_dict(),
                os.path.join(".", "param", "param_epoch"+str(epoch)+".pth")
            )

        else:
            torch.save(
                model.state_dict(),
                os.path.join(".", "param", "param_epoch"+str(epoch)+".pth")
            )

def train(model, device, epoch, dataloader, optimizer_model, L1Loss, VGGLoss):
    model.train()
    for batch_idx, (inputs, ghosts, scenes, masks) in enumerate(dataloader):

        inputs = inputs.to(device)
        ghosts = ghosts.to(device)
        scenes = scenes.to(device)
        masks = masks.to(device)

        optimizer_model.zero_grad()

        outputs = model(inputs)
        ghost_ests = torch.clamp(outputs, min=0, max=1)
        scene_ests = torch.clamp(inputs - ghost_ests, min=0, max=1)

        ghost_l1_loss = L1Loss(ghost_ests * masks, ghosts * masks)
        ghost_vgg_loss = VGGLoss(ghost_ests * masks, ghosts * masks)

        ghost_loss = ghost_l1_loss + ghost_vgg_loss

        scene_loss = L1Loss(scene_ests, scenes)

        loss = 0.6 * ghost_loss + 0.4 * scene_loss

        loss.backward()
        optimizer_model.step()        

        if batch_idx + 1 == len(dataloader):
            text = "Train Epoch: {} [Batch {:>3}/{}] [Loss: {:<1.6f}]".format(
                epoch, batch_idx + 1, len(dataloader), loss.item()
            )
            print(text)

def LRM_train(model, device, epoch, dataloader, optimizer_model, MSELoss, L1Loss, SSIMLoss, VGGLoss):
    model.train()
    for batch_idx, (inputs, ghosts, scenes, masks) in enumerate(dataloader):

        inputs = inputs.to(device)
        ghosts = ghosts.to(device)
        scenes = scenes.to(device)
        masks = masks.to(device)

        optimizer_model.zero_grad()

        b, _, h, w = inputs.shape
        h0 = Variable(torch.zeros(b, 64, h, w, device=device))
        c0 = Variable(torch.zeros(b, 64, h, w, device=device))
        fake_outputs = inputs.clone().detach()

        # 1st iteration
        h1, c1, rcmap01, ghost_ests01, _, _, scene_ests01 = model(inputs, fake_outputs, h0, c0)
        # 2nd iteration
        h2, c2, rcmap02, ghost_ests02, _, _, scene_ests02 = model(inputs, scene_ests01, h1, c1)
        # 3rd iteration
        _,  _,  rcmap03, ghost_ests03, _, _, scene_ests03 = model(inputs, scene_ests02, h2, c2)

        scene_ests01 = torch.clamp(scene_ests01, min=0, max=1)
        scene_ests02 = torch.clamp(scene_ests02, min=0, max=1)
        scene_ests03 = torch.clamp(scene_ests03, min=0, max=1)

        ghost_ests01 = torch.clamp(ghost_ests01, min=0, max=1)
        ghost_ests02 = torch.clamp(ghost_ests02, min=0, max=1)
        ghost_ests03 = torch.clamp(ghost_ests03, min=0, max=1)

        # Calculate composition loss
        input_ests03 = (1 - rcmap03) * scenes + ghosts
        loss_c_03 = (0.85 ** 0) * MSELoss(inputs, input_ests03)
        input_ests02 = (1 - rcmap02) * scenes + ghosts
        loss_c_02 = (0.85 ** 1) * MSELoss(inputs, input_ests02)
        input_ests01 = (1 - rcmap01) * scenes + ghosts
        loss_c_01 = (0.85 ** 2) * MSELoss(inputs, input_ests01)

        # Calculate perceptual loss
        loss_VGG_03 = VGGLoss(scenes, scene_ests03)
        loss_VGG_02 = VGGLoss(scenes, scene_ests02)
        loss_VGG_01 = VGGLoss(scenes, scene_ests01)

        # Calculate pixel and SSIM loss
        loss_pixel_03 = (0.85 ** 0) * L1Loss(scenes, scene_ests03)
        loss_pixel_02 = (0.85 ** 1) * L1Loss(scenes, scene_ests02)
        loss_pixel_01 = (0.85 ** 2) * L1Loss(scenes, scene_ests01)
        loss_SSIM_03 = (0.85 ** 0) * SSIMLoss(scenes, scene_ests03)
        loss_SSIM_02 = (0.85 ** 1) * SSIMLoss(scenes, scene_ests02)
        loss_SSIM_01 = (0.85 ** 2) * SSIMLoss(scenes, scene_ests01)
        loss_mix_03 = 0.84 * loss_SSIM_03 + 0.16 * loss_pixel_03
        loss_mix_02 = 0.84 * loss_SSIM_02 + 0.16 * loss_pixel_02
        loss_mix_01 = 0.84 * loss_SSIM_01 + 0.16 * loss_pixel_01

        # Calculate model loss
        loss_lrm = 0.4 * (loss_c_01 + loss_c_02 + loss_c_03) + 0.2 * (loss_VGG_01 + loss_VGG_02 + loss_VGG_03) + 0.4 * (loss_mix_01 + loss_mix_02 + loss_mix_03) 

        # Calculate train loss
        ghost_l1_loss = L1Loss(ghost_ests03 * masks, ghosts * masks)
        ghost_vgg_loss = VGGLoss(ghost_ests03 * masks, ghosts * masks)

        ghost_loss = ghost_l1_loss + ghost_vgg_loss

        scene_loss = L1Loss(scene_ests03, scenes)

        loss_train = 0.6 * ghost_loss + 0.4 * scene_loss

        # Calculate total loss
        loss = loss_lrm + loss_train

        loss.backward()
        optimizer_model.step()

        if (batch_idx + 1) % (len(dataloader) / 5) == 0 :
            text = "Train Epoch: {} [Batch {:>3}/{}] [Loss: {:<1.6f}]".format(
                epoch, batch_idx + 1, len(dataloader), loss.item()
            )
            print(text)
### end fuction


# training function for 1 epoch
def MPR_train(model, device, epoch, dataloader, optimizer_model, CharLoss, EdgeLoss, L1Loss, VGGLoss):
    model.train()
    for batch_idx, (inputs, ghosts, scenes, masks) in enumerate(dataloader):

        # GPU上に変数を移動
        inputs = inputs.to(device)
        ghosts = ghosts.to(device)
        scenes = scenes.to(device)
        masks = masks.to(device)

        # 各パラメータにある勾配情報を初期化
        optimizer_model.zero_grad()

        # モデルに適用して出力を得る
        scene_ests3, scene_ests2, scene_ests1 = model(inputs)

        scene_ests1 = torch.clamp(scene_ests1, min=0, max=1)
        scene_ests2 = torch.clamp(scene_ests2, min=0, max=1)
        scene_ests3 = torch.clamp(scene_ests3, min=0, max=1)

        ghost_ests1 = torch.clamp(inputs - scene_ests1, min=0, max=1)
        ghost_ests2 = torch.clamp(inputs - scene_ests2, min=0, max=1)
        ghost_ests3 = torch.clamp(inputs - scene_ests3, min=0, max=1)

        # ロスの計算
        loss_char3 = CharLoss(scene_ests3, scenes)
        loss_char2 = CharLoss(scene_ests2, scenes)
        loss_char1 = CharLoss(scene_ests1, scenes)
        loss_char = loss_char3 + loss_char2 + loss_char1
        
        loss_edge3 = EdgeLoss(scene_ests3, scenes)
        loss_edge2 = EdgeLoss(scene_ests2, scenes)
        loss_edge1 = EdgeLoss(scene_ests1, scenes)
        loss_edge = loss_edge3 + loss_edge2 + loss_edge1

        # # Calculate model loss
        loss_mpr = (loss_char) + (0.05*loss_edge)

        # Calculate train loss
        ghost_l1_loss1 = L1Loss(ghost_ests1 * masks, ghosts * masks)
        ghost_vgg_loss1 = VGGLoss(ghost_ests1 * masks, ghosts * masks)
        ghost_loss1 = ghost_l1_loss1 + ghost_vgg_loss1

        ghost_l1_loss2 = L1Loss(ghost_ests2 * masks, ghosts * masks)
        ghost_vgg_loss2 = VGGLoss(ghost_ests2 * masks, ghosts * masks)
        ghost_loss2 = ghost_l1_loss2 + ghost_vgg_loss2

        ghost_l1_loss3 = L1Loss(ghost_ests3 * masks, ghosts * masks)
        ghost_vgg_loss3 = VGGLoss(ghost_ests3 * masks, ghosts * masks)
        ghost_loss3 = ghost_l1_loss3 + ghost_vgg_loss3

        ghost_loss = ghost_loss1 + ghost_loss2 + ghost_loss3

        scene_loss1 = L1Loss(scene_ests1, scenes)
        scene_loss2 = L1Loss(scene_ests2, scenes)
        scene_loss3 = L1Loss(scene_ests3, scenes)

        scene_loss = scene_loss1 + scene_loss2 + scene_loss3

        # Calculate total loss
        loss_train = ghost_loss + scene_loss

        loss = loss_mpr + loss_train

        # ロスを誤差逆伝搬
        loss.backward()

        # 伝搬された勾配情報からパラメータを更新
        optimizer_model.step()

        if (batch_idx + 1) % (len(dataloader) / 5) == 0 :
            text = "Train Epoch: {} [Batch {:>3}/{}] [Loss: {:<1.6f}]".format(
                epoch, batch_idx + 1, len(dataloader), loss.item()
            )
            print(text)




if __name__ == "__main__":
    main()