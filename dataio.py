import os

import torch
from torch.utils import data

import data_util

import numpy as np
from glob import glob


class TwoViewsDataset(data.Dataset):
    """Create dataset of (I_1, I_2, T_{21}) image pairs and relative transform for each scene instance."""

    def __init__(self, data_dir, num_pairs_per_scene=-1, num_scenes=-1, sidelength=None):
        """
        Args:
            data_dir (string): Path to directory containing all instances
            num_pairs_per_scene: Number of image pairs per scene instance.
                                if -1 use all images for scene instance.
            num_scenes: if -1 use all scene in dir
        """
        self.sidelength = sidelength

        self.instance_dirs = sorted(glob(os.path.join(data_dir, "*/")))
        assert (len(self.instance_dirs) != 0), "No scene instances in the data directory"

        if num_scenes != -1:
            self.instance_dirs = self.instance_dirs[:num_scenes]

        self.rgb_paths = []
        self.pose_paths = []

        for instance_dir in self.instance_dirs:
            color_dir = os.path.join(instance_dir, "rgb")
            pose_dir = os.path.join(instance_dir, "pose")

            if not os.path.isdir(color_dir) or not os.path.isdir(pose_dir):
                print("Error! root dir %s is wrong" % data_dir)
                return

            scene_rgb_paths = sorted(data_util.glob_imgs(color_dir))
            scene_pose_paths = sorted(glob(os.path.join(pose_dir, "*.txt")))

            if num_pairs_per_scene == -1:
                self.rgb_paths += scene_rgb_paths
                self.pose_paths += scene_pose_paths
            else:
                self.rgb_paths += scene_rgb_paths[0:num_pairs_per_scene * 2]
                self.pose_paths += scene_pose_paths[0:num_pairs_per_scene * 2]

    def __len__(self):
        return len(self.rgb_paths) // 2

    def __getitem__(self, idx):

        img1 = data_util.load_rgb(self.rgb_paths[idx*2], sidelength=self.sidelength)
        img2 = data_util.load_rgb(self.rgb_paths[idx*2 + 1], sidelength=self.sidelength)
        img1 = img1.transpose(2, 0, 1)
        img2 = img2.transpose(2, 0, 1)

        pose1 = data_util.load_pose(self.pose_paths[idx*2])
        pose2 = data_util.load_pose(self.pose_paths[idx*2 + 1])
        transf21 = np.linalg.inv(pose2) @ pose1
        transf12 = np.linalg.inv(transf21)

        sample = {
            'image1': img1,
            'image2': img2,
            'transf21': transf21.flatten()[:12],
            'transf12': transf12.flatten()[:12],
            'pose1': pose1.flatten()[:12],
            'pose2': pose2.flatten()[:12]
        }
        return sample


class PosedImagesDataset(data.Dataset):
    """Create dataset of (I, T}) image and pose pairs."""

    def __init__(self, data_dir, num_views=1000000, sidelength=None):
        self.sidelength = sidelength

        color_dir = os.path.join(data_dir, "rgb")
        pose_dir = os.path.join(data_dir, "pose")

        if not os.path.isdir(color_dir) or not os.path.isdir(pose_dir):
            print("Error! root dir %s is wrong" % data_dir)
            return

        self.rgb_paths = sorted(data_util.glob_imgs(color_dir))[:num_views]
        self.pose_paths = sorted(glob(os.path.join(pose_dir, "*.txt")))[:num_views]

    def __len__(self):
        return len(self.rgb_paths)

    def __getitem__(self, idx):

        img = data_util.load_rgb(self.rgb_paths[idx],
                                 sidelength=self.sidelength).transpose(2, 0, 1)
        pose = data_util.load_pose(self.pose_paths[idx])

        sample = {
            'image': img,
            'pose': pose.flatten()[:12]
        }
        return sample
