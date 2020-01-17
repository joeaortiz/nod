import os

import torch
from torch.utils import data

import data_util

import numpy as np
from glob import glob


class TwoViewsDataset(data.Dataset):
    """Create dataset of (I_1, I_2, T_{21}) image pairs and relative transform for each scene instance."""

    def __init__(self, data_dir, num_pairs_per_instance):
        """
        Args:
            instances_dir (string): Path to directory containing all instances
        """
        self.instance_dirs = sorted(glob(os.path.join(data_dir, "*/")))
        assert (len(self.instance_dirs) != 0), "No objects in the data directory"

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

            self.rgb_paths += scene_rgb_paths[0:num_pairs_per_instance*2]
            self.pose_paths += scene_pose_paths[0:num_pairs_per_instance*2]

    def __len__(self):
        return len(self.rgb_paths) // 2

    def __getitem__(self, idx):

        img1 = data_util.load_rgb(self.rgb_paths[idx*2], sidelength=None)
        img2 = data_util.load_rgb(self.rgb_paths[idx*2 + 1], sidelength=None)
        img1 = img1.transpose(2, 0, 1)
        img2 = img2.transpose(2, 0, 1)

        pose1 = data_util.load_pose(self.pose_paths[idx*2])
        pose2 = data_util.load_pose(self.pose_paths[idx*2 + 1])
        transf21 = pose2 @ np.linalg.inv(pose1)
        transf12 = np.linalg.inv(transf21)

        sample = {
            'image1': img1,
            'image2': img2,
            'transf21': transf21.flatten()[:12],
            'transf12': transf12.flatten()[:12]
        }
        return sample
