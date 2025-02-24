# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Adapted from code originally written by Jason Zhang.
"""

import gzip
import json
import os
import os.path as osp
import random
import cv2

import numpy as np
import torch
from PIL import Image, ImageFile
from pytorch3d.renderer import PerspectiveCameras
from pytorch3d import io as py3d_io
from torch.utils.data import Dataset
from torchvision import transforms

from util.normalize_cameras import normalize_cameras

from multiprocessing import Pool
import tqdm
from util.camera_transform import adjust_camera_to_bbox_crop_, adjust_camera_to_image_scale_, bbox_xyxy_to_xywh

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True


class lmDataset(Dataset):
    def __init__(
        self,
        category=None,
        ref_images_num=16,
        transform=None,
        img_size=224,
        eval_time=True,
        mask_images=False,
        LM_DIR=None,
    ):
        """
        Args:
            category (iterable): List of categories to use. If "all" is in the list,
                all training categories are used.
            num_images (int): Default number of images in each batch.
            normalize_cameras (bool): If True, normalizes cameras so that the
                intersection of the optical axes is placed at the origin and the norm
                of the first camera translation is 1.
            first_camera_transform (bool): If True, tranforms the cameras such that
                camera 1 has extrinsics [I | 0].
            mask_images (bool): If True, masks out the background of the images.
        """

        # only support one category
        assert category in ALL_CATEGORIES
        self.obj_name = category
        self.category = [category]

        if LM_DIR == None:
            raise ValueError("LINEMOD_DIR is not specified")
        print(f"LINEMOD_DIR is {LM_DIR}")
        self.LM_DIR = LM_DIR # /scratch/liudan/data/linemod/LM_dataset

        self.to_meter_scale = 1e-3
        self.add_detector_noise = False
        self.use_yolo_box = True
        self.ref_images_num = ref_images_num

        self.name2classID = {'ape': 1, "benchvise": 2,  
                            'cam': 4, "can": 5,
                            "cat": 6, "driller": 8, 
                            "duck": 9, "eggbox": 10, 
                            "glue": 11, "holepuncher": 12, 
                            "iron": 13, "lamp": 14, "phone": 15,
            }

        self.camK = np.array([
            [572.4114 ,   0.     , 325.2611 ],
            [  0.     , 573.57043, 242.049  ],
            [  0.     ,   0.     ,   1.     ]], dtype=np.float32)


        self.all_data = []

        self.query_list = {}
        self.ref_list = {}
        self.obj_pointcloud = {}

        for c in self.category:
            obj_id = self.name2classID[c]
            obj_full_name = "-".join(["0801", "lm" + str(int(obj_id)), "others"])
            obj_path = osp.join(self.LM_DIR, obj_full_name)

            query_path = osp.join(obj_path, "-".join(["lm" + str(int(obj_id)), "3"]))
            ref_path = osp.join(obj_path, "-".join(["lm" + str(int(obj_id)), "1"]))

            query_rgb_path = osp.join(query_path, "color")
            query_pose_path = osp.join(query_path, "poses_ba")
            query_intrin_path = osp.join(query_path, "intrin")

            ref_rgb_path = osp.join(ref_path, "color")
            ref_pose_path = osp.join(ref_path, "poses_ba")
            ref_intrin_path = osp.join(ref_path, "intrin")

            model_path = osp.join(obj_path, "model_eval.ply")
            assert osp.exists(model_path), f"{model_path}"
            obj_pointcloud = py3d_io.load_ply(model_path)[0].numpy()  # NOTE: models' units are m
            self.obj_pointcloud[c] = obj_pointcloud

            diameter_path = osp.join(obj_path, "diameter.txt")
            self.diameter = np.loadtxt(diameter_path)

            self.query_list[c] = []
            for i in os.listdir(query_rgb_path):
                t = i.replace('png', 'txt')
                img_path = osp.join(query_rgb_path, i)

                gt_pose_path = osp.join(query_pose_path, t)
                assert osp.exists(gt_pose_path), f"{gt_pose_path}"
                gt_pose = torch.from_numpy(np.loadtxt(gt_pose_path))  # [4*4]

                intrin_path = osp.join(query_intrin_path, t)
                assert osp.exists(intrin_path), f"{intrin_path}"
                K_crop = torch.from_numpy(np.loadtxt(intrin_path))  # [3*3]

                obj_info = {
                    'rgb_path': img_path,
                    'pose': gt_pose,
                    'K': K_crop,
                }
                self.query_list[c].append(obj_info)

                # collect all query images
                self.all_data.append(obj_info)


            self.ref_list[c] = []
            for i in os.listdir(ref_rgb_path):
                t = i.replace('png', 'txt')
                img_path = osp.join(ref_rgb_path, i)

                gt_pose_path = osp.join(ref_pose_path, t)
                assert osp.exists(gt_pose_path), f"{gt_pose_path}"
                gt_pose = torch.from_numpy(np.loadtxt(gt_pose_path))  # [4*4]

                intrin_path = osp.join(ref_intrin_path, t)
                assert osp.exists(intrin_path), f"{intrin_path}"
                K_crop = torch.from_numpy(np.loadtxt(intrin_path))  # [3*3]

                obj_info = {
                    'rgb_path': img_path,
                    'pose': gt_pose,
                    'K': K_crop,
                }
                self.ref_list[c].append(obj_info)



    def __len__(self):
        return len(self.all_data)


    def __getitem__(self, idx):
        data_dict = dict()

        query_image = self.all_data[idx]

        cur_obj = self.obj_name
        
        ref_len = len(self.ref_list[cur_obj])
        ids = np.random.choice(ref_len, self.ref_images_num, replace=False)

        data_dict['query_image'] = {}
        data_dict['ref_images'] = {}

        query_rgb_path = query_image['rgb_path']
        query_pose = query_image['pose']
        query_R = query_pose[:, :3]
        query_T = query_pose[:, 3]
        K = query_image['K']

        image_crop = cv2.imread(query_rgb_path)

        data_dict['query_image']['image'] = torch.as_tensor(image_crop)
        data_dict['query_image']['pose'] = torch.as_tensor(query_pose)
        data_dict['query_image']['R'] = torch.as_tensor(query_R)
        data_dict['query_image']['T'] = torch.as_tensor(query_T)


        ref_images = []
        ref_poses = []
        ref_Rs = []
        ref_Ts = []
        for i in ids:
            ref = self.ref_list[cur_obj][i]
            ref_rgb_path = ref['rgb_path']
            ref_pose = ref['pose']
            ref_R = ref_pose[:, :3]
            ref_T = ref_pose[:, 3]
            K = ref['K']


            image_crop = cv2.imread(ref_rgb_path)

            ref_images.append(torch.as_tensor(image_crop)) # 3xHxW
            ref_poses.append(torch.as_tensor(ref_pose))
            ref_Rs.append(torch.as_tensor(ref_R)) # 3x3
            ref_Ts.append(torch.as_tensor(ref_T)) # 3

        data_dict['ref_images']['image'] = torch.stack(ref_images)
        data_dict['ref_images']['pose'] = torch.stack(ref_poses)
        data_dict['ref_images']['R'] = torch.stack(ref_Rs)
        data_dict['ref_images']['T'] = torch.stack(ref_Ts)

        return data_dict
    

ALL_CATEGORIES = [
    'ape', 
    "benchvise",  
    'cam', 
    "can",
    "cat", 
    "driller", 
    "duck", 
    "eggbox", 
    "glue", 
    "holepuncher", 
    "iron", 
    "lamp", 
    "phone"
]

TEST_CATEGORIES = ["benchvise"]