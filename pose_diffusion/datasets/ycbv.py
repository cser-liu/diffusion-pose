import gzip
import json
import os.path as osp
import random
import os 
import cv2

import numpy as np
import torch
from PIL import Image, ImageFile
from pytorch3d.renderer import PerspectiveCameras
from pytorch3d.renderer.cameras import get_ndc_to_screen_transform
from pytorch3d import io as py3d_io
from torch.utils.data import Dataset
from torchvision import transforms
import pickle

from util.normalize_cameras import normalize_cameras

import h5py
from io import BytesIO

from multiprocessing import Pool
from tqdm import tqdm
import mmcv

from util.camera_transform import adjust_camera_to_bbox_crop_, adjust_camera_to_image_scale_, bbox_xyxy_to_xywh
from util.data_utils import get_image_crop_resize, get_K_crop_resize

import matplotlib.pyplot as plt


Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True


class YcbvDataset(Dataset):
    def __init__(
        self,
        split="train",
        transform=None,
        img_size=224,
        ref_images_num=16,
        eval_time=False,
        mask_images=True,
        YCBV_DIR=None,
    ):
        
        self.ycbv_DIR = YCBV_DIR
        if YCBV_DIR == None:
            raise NotImplementedError
        print(f"ycbv_DIR is {YCBV_DIR}")

        
        if split == "train":
            self.category = [i for i in range(1, 21)]
        elif split == "test":
            self.category = [21]

        # /scratch/liudan/data/ycbv/...
        self.data_dir = os.path.join(YCBV_DIR, "train_pbr")

        self.scenes = [f"{i:06d}" for i in range(1)]
        self.add_detector_noise = False
        
        self.obj_data = {}
        self.all_data = []

        for scene in tqdm(self.scenes):
            # scene information
            scene_id = int(scene)
            scene_root = osp.join(self.data_dir, scene)
            with open(osp.join(scene_root, "scene_gt.json"), "r") as f:
                gt_dict = json.load(f)
            with open(osp.join(scene_root, "scene_gt_info.json"), "r") as f_info:
                gt_info_dict = json.load(f_info)
            with open(osp.join(scene_root, "scene_camera.json"), "r") as f_cam:
                cam_dict = json.load(f_cam)


            for str_im_id in tqdm(gt_dict, postfix=f"{scene_id}"):
                int_im_id = int(str_im_id)
                # int_im_id: 000000 - 000999
                rgb_path = osp.join(scene_root, "rgb/{:06d}.jpg").format(int_im_id)
                assert osp.exists(rgb_path), rgb_path

                # depth_path = osp.join(scene_root, "depth/{:06d}.png".format(int_im_id))
                
                K = np.array(cam_dict[str_im_id]["cam_K"], dtype=np.float32)
                focal_length = np.array([K[0], K[4]], dtype=np.float32)
                principal_point = np.array([K[2], K[5]], dtype=np.float32)

                depth_factor = cam_dict[str_im_id]["depth_scale"]  

                for anno_i, anno in enumerate(gt_dict[str_im_id]):
                    obj_id = anno["obj_id"]

                    if obj_id not in self.category:
                        continue

                    self.obj_ply_path = osp.join(YCBV_DIR, "models/obj_{:06d}.ply".format(obj_id))
                    self.obj_pointcloud = py3d_io.load_ply(self.obj_ply_path)[0].numpy()  

                    # scene_obj_id = f"scene{scene_id}_obj{obj_id}"
                    obj_class = id2obj[obj_id]

                    if obj_id not in self.obj_data.keys():
                        self.obj_data[obj_id] = []

                    R = np.array(anno["cam_R_m2c"], dtype="float32").reshape(3, 3)
                    t = np.array(anno["cam_t_m2c"], dtype="float32") 

                    pose = np.hstack([R, t.reshape(3, 1)])  # 3x4

                    bbox_visib = gt_info_dict[str_im_id][anno_i]["bbox_visib"]
                    bbox_obj = gt_info_dict[str_im_id][anno_i]["bbox_obj"]
                    x1, y1, w, h = bbox_visib

                    if h <= 10 or w <= 10:
                        continue
                    
                    mask_file = osp.join(scene_root, "mask/{:06d}_{:06d}.png".format(int_im_id, anno_i))
                    mask_visib_file = osp.join(scene_root, "mask_visib/{:06d}_{:06d}.png".format(int_im_id, anno_i))
                    assert osp.exists(mask_file), mask_file
                    assert osp.exists(mask_visib_file), mask_visib_file


                    obj_info = {
                        "rgb_path": rgb_path,
                        # "depth_path": depth_path,
                        "mask_path": mask_file,
                        "mask_visib_path": mask_visib_file,    
                        "obj_id": obj_id,  
                        "bbox_visib": bbox_visib,  # x, y, w, h
                        "bbox": bbox_obj,
                        "pose": pose,
                        "T": t,
                        "R": R,
                        "cam_K": K.reshape(3, 3),
                        "pts": self.obj_pointcloud,
                        "focal_length": focal_length,
                        "principal_point": principal_point,
                        "depth_scale": depth_factor,                   
                    }

                    self.obj_data[obj_id].append(obj_info)
                    self.all_data.append(obj_info)

        self.sequence_list = list(self.obj_data.keys())
        print(f"sequence nums: {self.sequence_list}")
        

        if transform is None:
            self.transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Resize((224, 224), antialias=True),
                ]
            )
        else:
            self.transform = transform

        self.ref_images_num = ref_images_num
        self.img_size = img_size
        self.eval_time = eval_time
        self.mask_images = mask_images
        
        print(f"Data size: {len(self)}")

    def __len__(self):
        return len(self.all_data)
    
    def __getitem__(self, idx):
        data_dict = dict()

        query_image = self.all_data[idx]
        cur_obj = query_image['obj_id']
        ref_len = len(self.obj_data[cur_obj])
        ids = np.random.choice(ref_len, self.ref_images_num, replace=False)

        data_dict['query_image'] = {}
        data_dict['ref_images'] = {}

        query_rgb_path = query_image['rgb_path']
        query_bbox = query_image['bbox']
        query_R = query_image['R']
        query_T = query_image['T']
        K = query_image['cam_K']

        # todo: filter some bad images
        # image = Image.open(query_rgb_path).convert("RGB")
        # image = transforms.functional.crop(image, top=query_bbox[1], left=query_bbox[0], height=query_bbox[3], width=query_bbox[2])
        # image = self.transform(image)

        original_img = cv2.imread(query_rgb_path)
        img_h,img_w = original_img.shape[:2]
        x0, y0, w, h = query_bbox
        x1 = x0 + w
        y1 = y0 + h

        if not self.add_detector_noise:
            compact_percent = 0.1
            x0 -= int(w * compact_percent)
            y0 -= int(h * compact_percent)
            x1 += int(w * compact_percent)
            y1 += int(h * compact_percent)
        else:
            compact_percent = 0.1
            offset_percent = np.random.uniform(low=-1*compact_percent, high=1*compact_percent)
            # apply compact noise:
            x0 -= int(w * compact_percent)
            y0 -= int(h * compact_percent)
            x1 += int(w * compact_percent)
            y1 += int(h * compact_percent)
            # apply offset noise:
            x0 += int(w * offset_percent)
            y0 += int(h * offset_percent)
            x1 += int(w * offset_percent)
            y1 += int(h * offset_percent)

        # Crop image by 2D visible bbox, and change K
        box = np.array([x0, y0, x1, y1])
        resize_shape = np.array([y1 - y0, x1 - x0])
        K_crop, K_crop_homo = get_K_crop_resize(box, K, resize_shape)
        image_crop, _ = get_image_crop_resize(original_img, box, resize_shape)

        box_new = np.array([0, 0, x1 - x0, y1 - y0])
        resize_shape = np.array([256, 256])
        K_crop, K_crop_homo = get_K_crop_resize(box_new, K_crop, resize_shape)
        image_crop, _ = get_image_crop_resize(image_crop, box_new, resize_shape)


        query_pose = query_image['pose']

        data_dict['query_image']['image'] = torch.tensor(image_crop)
        data_dict['query_image']['pose'] = torch.tensor(query_pose)
        data_dict['query_image']['R'] = torch.tensor(query_R)
        data_dict['query_image']['T'] = torch.tensor(query_T)

        # pts = query_image['pts']
        # data_dict['pts'] = pts

        ref_images = []
        ref_poses = []
        ref_Rs = []
        ref_Ts = []
        for i in ids:
            ref = self.obj_data[cur_obj][i]
            ref_rgb_path = ref['rgb_path']
            ref_bbox = ref['bbox']
            ref_pose = ref['pose']
            ref_R = ref['R']
            ref_T = ref['T']
            K = ref['cam_K']

            # ref_image = Image.open(ref_rgb_path).convert("RGB")
            # ref_image = transforms.functional.crop(ref_image, top=ref_bbox[1], left=ref_bbox[0], height=ref_bbox[3], width=ref_bbox[2])
            # ref_image = self.transform(ref_image)

            original_img = cv2.imread(ref_rgb_path)
            x0, y0, w, h = ref_bbox
            x1 = x0 + w
            y1 = y0 + h

            if not self.add_detector_noise:
                compact_percent = 0.1
                x0 -= int(w * compact_percent)
                y0 -= int(h * compact_percent)
                x1 += int(w * compact_percent)
                y1 += int(h * compact_percent)
            else:
                compact_percent = 0.1
                offset_percent = np.random.uniform(low=-1*compact_percent, high=1*compact_percent)
                # apply compact noise:
                x0 -= int(w * compact_percent)
                y0 -= int(h * compact_percent)
                x1 += int(w * compact_percent)
                y1 += int(h * compact_percent)
                # apply offset noise:
                x0 += int(w * offset_percent)
                y0 += int(h * offset_percent)
                x1 += int(w * offset_percent)
                y1 += int(h * offset_percent)

            # Crop image by 2D visible bbox, and change K
            box = np.array([x0, y0, x1, y1])
            resize_shape = np.array([y1 - y0, x1 - x0])
            K_crop, K_crop_homo = get_K_crop_resize(box, K, resize_shape)
            image_crop, _ = get_image_crop_resize(original_img, box, resize_shape)

            box_new = np.array([0, 0, x1 - x0, y1 - y0])
            resize_shape = np.array([256, 256])
            K_crop, K_crop_homo = get_K_crop_resize(box_new, K_crop, resize_shape)
            image_crop, _ = get_image_crop_resize(image_crop, box_new, resize_shape)

            ref_images.append(torch.tensor(image_crop)) # 3xHxW
            ref_poses.append(torch.tensor(ref_pose))
            ref_Rs.append(torch.tensor(ref_R)) # 3x3
            ref_Ts.append(torch.tensor(ref_T)) # 3

        data_dict['ref_images']['image'] = torch.stack(ref_images)
        data_dict['ref_images']['pose'] = torch.stack(ref_poses)
        data_dict['ref_images']['R'] = torch.stack(ref_Rs)
        data_dict['ref_images']['T'] = torch.stack(ref_Ts)

        return data_dict


id2obj = {
    1: "002_master_chef_can",  # [1.3360, -0.5000, 3.5105]
    2: "003_cracker_box",  # [0.5575, 1.7005, 4.8050]
    3: "004_sugar_box",  # [-0.9520, 1.4670, 4.3645]
    4: "005_tomato_soup_can",  # [-0.0240, -1.5270, 8.4035]
    5: "006_mustard_bottle",  # [1.2995, 2.4870, -11.8290]
    6: "007_tuna_fish_can",  # [-0.1565, 0.1150, 4.2625]
    7: "008_pudding_box",  # [1.1645, -4.2015, 3.1190]
    8: "009_gelatin_box",  # [1.4460, -0.5915, 3.6085]
    9: "010_potted_meat_can",  # [2.4195, 0.3075, 8.0715]
    10: "011_banana",  # [-18.6730, 12.1915, -1.4635]
    11: "019_pitcher_base",  # [5.3370, 5.8855, 25.6115]
    12: "021_bleach_cleanser",  # [4.9290, -2.4800, -13.2920]
    13: "024_bowl",  # [-0.2270, 0.7950, -2.9675]
    14: "025_mug",  # [-8.4675, -0.6995, -1.6145]
    15: "035_power_drill",  # [9.0710, 20.9360, -2.1190]
    16: "036_wood_block",  # [1.4265, -2.5305, 17.1890]
    17: "037_scissors",  # [7.0535, -28.1320, 0.0420]
    18: "040_large_marker",  # [0.0460, -2.1040, 0.3500]
    19: "051_large_clamp",  # [10.5180, -1.9640, -0.4745]
    20: "052_extra_large_clamp",  # [-0.3950, -10.4130, 0.1620]
    21: "061_foam_brick",  # [-0.0805, 0.0805, -8.2435]
}