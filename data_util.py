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

def normalize(x: np.ndarray) -> np.ndarray:
    assert x.ndim == 1, 'x must be a vector (ndim: 1)'
    return x / np.linalg.norm(x)


def look_at(
    eye,
    target,
    up,
) -> np.ndarray:
    """Returns transformation matrix with eye, at and up.
    Parameters
    ----------
    eye: (3,) float
        Camera position.
    target: (3,) float
        Camera look_at position.
    up: (3,) float
        Vector that defines y-axis of camera (z-axis is vector from eye to at).
    Returns
    -------
    T_cam2world: (4, 4) float (if return_homography is True)
        Homography transformation matrix from camera to world.
        Points are transformed like below:
            # x: camera coordinate, y: world coordinate
            y = trimesh.transforms.transform_points(x, T_cam2world)
            x = trimesh.transforms.transform_points(
                y, np.linalg.inv(T_cam2world)
            )
    """
    eye = np.asarray(eye, dtype=float)

    if target is None:
        target = np.array([0, 0, 0], dtype=float)
    else:
        target = np.asarray(target, dtype=float)

    if up is None:
        up = np.array([0, 0, -1], dtype=float)
    else:
        up = np.asarray(up, dtype=float)

    assert eye.shape == (3,), 'eye must be (3,) float'
    assert target.shape == (3,), 'target must be (3,) float'
    assert up.shape == (3,), 'up must be (3,) float'

    # create new axes
    z_axis = normalize(target - eye)
    x_axis = normalize(np.cross(up, z_axis))
    y_axis = normalize(np.cross(z_axis, x_axis))

    # create rotation matrix: [bs, 3, 3]
    R = np.vstack((x_axis, y_axis, z_axis))
    t = eye

    T_cam2world = np.zeros([4, 4])
    T_cam2world[:3, :3] = R.T
    T_cam2world[:3, 3] = t
    T_cam2world[3, 3] = 1.
    return T_cam2world


def gen_pose_circle(ref_pose, n_poses, centre=[0., 0., 0.]):
    """ Generate poses in a circle around the origin which go through the given pose.
        Poses are generated with equal polar angle to the reference pose and uniformly
        sampled azimuthal angles.
        phi is azimuthal angle.
        theta is polar angle. """

    ref_loc = ref_pose[:3, 3]
    r = np.linalg.norm(ref_loc)  # radial distance
    ref_theta = np.arccos(ref_loc[2] / r)
    ref_phi = np.arctan(ref_loc[1] / ref_loc[0])

    # print(ref_loc, r)
    # print(ref_phi * 180 / np.pi)
    # print(ref_theta * 180 / np.pi)

    poses = []
    sampled_phis = np.linspace(0, 2 * np.pi - 0.05 / n_poses, n_poses)  # Don't sample right up to 2*pi
    for phi in sampled_phis:
        loc = np.array([r * np.sin(ref_theta) * np.cos(phi),
                        r * np.sin(ref_theta) * np.sin(phi),
                        r * np.cos(ref_theta)])
        poses.append(look_at(loc,
                             centre,
                             np.array([0., 0., 1.])).astype(float))

    return poses
