from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
import torchvision
from torchvision import transforms
from torchinfo import summary

### OpenCV -> PIL
def cv2pil(image):
    new_image = image.copy()
    if new_image.ndim == 2:
        pass
    elif new_image.shape[2] == 3:
        new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)
    elif new_image.shape[2] == 4:
        new_image = cv2.cvtColor(new_image, cv2.COLOR_BGRA2RGBA)
    new_image = Image.fromarray(new_image)
    return new_image
### end function

### PIL -> OpenCV 
def pil2cv(image):
    new_image = np.array(image, dtype=np.uint8)
    if new_image.ndim == 2:
        pass
    elif new_image.shape[2] == 3:
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)
    elif new_image.shape[2] == 4:
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGBA2BGRA)
    return new_image
### end function

def tensor2pil(img_tensor):
    img_tensor  = torch.round(img_tensor*255)
    img_tensor  = img_tensor.to(torch.uint8).detach().to("cpu")
    func = transforms.ToPILImage()
    img_pil = func(img_tensor)
    return img_pil

def pil2tensor(img_pil):
    func = transforms.ToTensor()
    img_tensor = func(img_pil)
    return img_tensor

### gamma correction ガンマ補正（変換）cv2
def gamma_correction(img, gamma):
    """
    **********************************************************
    【ガンマ補正の公式】
    Y = 255(X/255)**(1/γ)

    【γの設定方法】
    ・γ>1の場合：画像が明るくなる
    ・γ<1の場合：画像が暗くなる
    **********************************************************
    """
    img2gamma = np.zeros((256,1),dtype=np.uint8)  # ガンマ変換初期値
    # 公式適用
    for i in range(256):
        img2gamma[i][0] = 255 * (float(i)/255) ** (1.0 /gamma)
    # 読込画像をガンマ変換
    gamma_img = cv2.LUT(img,img2gamma)
    # 画像を表示
    return gamma_img
### end function