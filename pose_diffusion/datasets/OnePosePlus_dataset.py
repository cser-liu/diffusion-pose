from loguru import logger

try:
    import ujson as json
except ImportError:
    import json
import torch
import torch.nn.functional as F
import numpy as np
import os.path as osp
import os
from pycocotools.coco import COCO
from torch.utils.data import Dataset

from kornia import homography_warp, normalize_homography, normal_transform_pixel
from src.utils.data_io import read_grayscale
from src.utils import data_utils
from src.utils.sample_homo import sample_homography_sap


class OnePosePlusDataset(Dataset):
    def __init__(
        self,
        data_dir,
        anno_file,
        pad=True,
        img_pad=False,
        img_resize=512,
        coarse_scale=1 / 8,
        df=8,
        shape3d=10000,
        percent=1.0,
        split="train",
        ref_images_num=16,
    ):
        super(Dataset, self).__init__()

        assert split in ["train", "test", "val"]
        self.split = split
        self.data_dir = osp.join(data_dir, f"{self.split}_data") # scratch/liudan/data/onepose/...
        self.ref_images_num = ref_images_num

        self.category = []
        self.all_query = []
        self.query_images = {}
        self.ref_images = {}
        for c in os.listdir(self.data_dir):
            obj_name = c.split("-")[1]
            self.category.append(obj_name)
            obj_path = osp.join(data_dir, c)

            self.query_images[obj_name] = []
            query_dir1 = osp.join(obj_path, f"{obj_name}-1")
            for i in os.listdir(query_dir1):
                img_path = osp.join(query_dir1, i)
                self.query_images[obj_name].append(img_path)

                obj_info = {
                    "img_path": img_path,
                    "obj_name": obj_name,
                }
                self.all_query.append(obj_info)

            query_dir2 = osp.join(obj_path, f"{obj_name}-2")
            for i in os.listdir(query_dir2):
                img_path = osp.join(query_dir2, i)
                self.query_images[obj_name].append(img_path)

                obj_info = {
                    "img_path": img_path,
                    "obj_name": obj_name,
                }
                self.all_query.append(obj_info)

            query_dir3 = osp.join(obj_path, f"{obj_name}-3")
            for i in os.listdir(query_dir3):
                img_path = osp.join(query_dir3, i)
                self.query_images[obj_name].append(img_path) 

                obj_info = {
                    "img_path": img_path,
                    "obj_name": obj_name,
                }
                self.all_query.append(obj_info)

            self.ref_images[obj_name] = []
            ref_dir = osp.join(obj_path, f"{obj_name}-4")
            for i in os.listdir(ref_dir):
                img_path = osp.join(ref_dir, i)
                self.ref_images[obj_name].append(img_path)

            print(f"category {obj_name} has {len(self.ref_images[obj_name])} reference images, {len(self.query_images[obj_name])} query images")


    def get_intrin_by_color_pth(self, img_path):
        img_ext = osp.splitext(img_path)[1]
        intrin_path = img_path.replace("/color/", "/intrin_ba/").replace(img_ext, ".txt")
        K_crop = torch.from_numpy(np.loadtxt(intrin_path))  # [3*3]
        return K_crop

    def get_gt_pose_by_color_pth(self, img_path):
        img_ext = osp.splitext(img_path)[1]
        gt_pose_path = img_path.replace("/color/", "/poses_ba/").replace(img_ext, ".txt")
        pose_gt = torch.from_numpy(np.loadtxt(gt_pose_path))  # [4*4]
        return pose_gt

    def __getitem__(self, index):
    
        data_dict = dict()

        obj_info = self.all_query[index]
        obj_name = obj_info["obj_name"]
        img_path = obj_info["img_path"]
        ref_len = len(self.ref_images[obj_name])
        ids = np.random.choice(ref_len, self.ref_images_num, replace=False)

        data_dict['query_image'] = {}
        data_dict['ref_images'] = {}

        


    def __len__(self):
        return len(self.all_query)
