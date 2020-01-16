import functools
import cv2
import numpy as np
import imageio
from glob import glob
import os
import shutil
import skimage
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torchvision


def load_rgb(path, sidelength=None):
    img = imageio.imread(path)[:, :, :3]
    img = skimage.img_as_float32(img)

    # img = square_crop_img(img)

    if sidelength is not None:
        img = cv2.resize(img, (sidelength, sidelength), interpolation=cv2.INTER_AREA)

    img -= 0.5
    img *= 2.
    # img = img.transpose(2, 0, 1)
    return img


def load_pose(filename):
    lines = open(filename).read().splitlines()
    if len(lines) == 1:
        pose = np.zeros((4, 4), dtype=np.float32)
        for i in range(16):
            pose[i // 4, i % 4] = lines[0].split(" ")[i]
        return pose.squeeze()
    else:
        lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines[:4])]
        return np.asarray(lines).astype(np.float32).squeeze()


def cond_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def glob_imgs(path):
    imgs = []
    for ext in ['*.png', '*.jpg', '*.JPEG', '*.JPG']:
        imgs.extend(glob(os.path.join(path, ext)))
    return imgs


def show_batch_pairs(batch):
    images = torch.cat((batch['image1'], batch['image2']), 0)
    images /= 2.
    images += 0.5
    img = torchvision.utils.make_grid(images, nrow=images.size()[0]//2)
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)), interpolation='nearest')
    plt.show()
