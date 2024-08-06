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
import cv2
from torch.utils.data import Dataset
from torchvision import transforms


class OnePosePlusDataset(Dataset):
    def __init__(
        self,
        data_dir,
        img_resize=256,
        split="train",
        ref_images_num=16,
        transform=None,
    ):
        super(Dataset, self).__init__()

        assert split in ["train", "test", "val"]
        self.split = split
        self.data_dir = osp.join(data_dir, f"{self.split}_data") # scratch/liudan/data/onepose/...
        self.ref_images_num = ref_images_num

        if transform is None:
            self.transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Resize((img_resize, img_resize), antialias=True),
                ]
            )
        else:
            self.transform = transform

        self.category = []
        self.all_query = []
        self.query_images = {}
        self.ref_images = {}
        for c in os.listdir(self.data_dir):
            obj_name = c.split("-")[1]
            self.category.append(obj_name)
            obj_path = osp.join(self.data_dir, c)

            self.query_images[obj_name] = []
            query_dir1 = osp.join(obj_path, f"{obj_name}-1/color")
            for i in os.listdir(query_dir1):
                img_path = osp.join(query_dir1, i)
                K_crop = self.get_intrin_by_color_pth(img_path)
                pose_gt = self.get_gt_pose_by_color_pth(img_path)
                obj = {
                    "img_path": img_path,
                    "pose": pose_gt,
                    "cam_K": K_crop,
                }
                self.query_images[obj_name].append(obj)

                obj_info = {
                    "img_path": img_path,
                    "obj_name": obj_name,
                }
                self.all_query.append(obj_info)

            query_dir2 = osp.join(obj_path, f"{obj_name}-2/color")
            for i in os.listdir(query_dir2):
                img_path = osp.join(query_dir2, i)
                K_crop = self.get_intrin_by_color_pth(img_path)
                pose_gt = self.get_gt_pose_by_color_pth(img_path)
                obj = {
                    "img_path": img_path,
                    "pose": pose_gt,
                    "cam_K": K_crop,
                }
                self.query_images[obj_name].append(obj)

                obj_info = {
                    "img_path": img_path,
                    "obj_name": obj_name,
                }
                self.all_query.append(obj_info)

            query_dir3 = osp.join(obj_path, f"{obj_name}-3/color")
            for i in os.listdir(query_dir3):
                img_path = osp.join(query_dir3, i)
                K_crop = self.get_intrin_by_color_pth(img_path)
                pose_gt = self.get_gt_pose_by_color_pth(img_path)
                obj = {
                    "img_path": img_path,
                    "pose": pose_gt,
                    "cam_K": K_crop,
                }
                self.query_images[obj_name].append(obj) 

                obj_info = {
                    "img_path": img_path,
                    "obj_name": obj_name,
                }
                self.all_query.append(obj_info)

            self.ref_images[obj_name] = []
            ref_dir = osp.join(obj_path, f"{obj_name}-4/color")
            for i in os.listdir(ref_dir):
                img_path = osp.join(ref_dir, i)
                K_crop = self.get_intrin_by_color_pth(img_path)
                pose_gt = self.get_gt_pose_by_color_pth(img_path)
                obj = {
                    "img_path": img_path,
                    "pose": pose_gt,
                    "cam_K": K_crop,
                }
                self.ref_images[obj_name].append(obj)

            print(f"category {obj_name} has {len(self.ref_images[obj_name])} reference images, {len(self.query_images[obj_name])} query images")
        

    def get_intrin_by_color_pth(self, img_path):
        img_ext = osp.splitext(img_path)[1]
        
        intrin_path = img_path.replace("/color/", "/intrin_ba/").replace(img_ext, ".txt")
        K_crop = torch.from_numpy(np.loadtxt(intrin_path))  # [3*3]
        assert K_crop.shape[0]==3 and K_crop.shape[1]==3
        return K_crop

    def get_gt_pose_by_color_pth(self, img_path):
        img_ext = osp.splitext(img_path)[1]
        gt_pose_path = img_path.replace("/color/", "/poses_ba/").replace(img_ext, ".txt")
        pose_gt = torch.from_numpy(np.loadtxt(gt_pose_path))  # [4*4]
        assert pose_gt.shape[0]==4 and pose_gt.shape[1]==4
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

        query_image = cv2.imread(img_path)
        if self.transform:
            query_image = self.transform(query_image)
        query_pose = self.get_gt_pose_by_color_pth(img_path)

        data_dict['query_image']['image'] = torch.as_tensor(query_image, dtype=torch.float32)
        data_dict['query_image']['pose'] = query_pose
        data_dict['query_image']['R'] = query_pose[:3, :3]
        data_dict['query_image']['T'] = query_pose[:3, 3]

        ref_images = []
        ref_poses = []
        ref_Rs = []
        ref_Ts = []
        for i in ids:
            ref = self.ref_images[obj_name][i]
            ref_rgb_path = ref['img_path']
            ref_image = cv2.imread(ref_rgb_path)

            ref_pose = ref['pose']
            ref_R = ref_pose[:3, :3]
            ref_T = ref_pose[:3, 3]
            K = ref['cam_K']

            if self.transform:
                ref_image = self.transform(ref_image)
            ref_image = torch.as_tensor(ref_image, dtype=torch.float32)

            ref_images.append(ref_image) # 3xHxW
            ref_poses.append(ref_pose)
            ref_Rs.append(ref_R) # 3x3
            ref_Ts.append(ref_T) # 3

        data_dict['ref_images']['image'] = torch.stack(ref_images)
        data_dict['ref_images']['pose'] = torch.stack(ref_poses)
        data_dict['ref_images']['R'] = torch.stack(ref_Rs)
        data_dict['ref_images']['T'] = torch.stack(ref_Ts)

        # print(data_dict['ref_images']['image'].shape)

        return data_dict
        


    def __len__(self):
        return len(self.all_query)
