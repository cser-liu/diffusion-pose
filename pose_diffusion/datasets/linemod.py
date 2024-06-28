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

import numpy as np
import torch
from PIL import Image, ImageFile
from pytorch3d.renderer import PerspectiveCameras
from pytorch3d import io as py3d_io
from pytorch3d import ops as py3d_ops
from pytorch3d import transforms as py3d_transform

from torch.utils.data import Dataset
from torchvision import transforms

from util.normalize_cameras import normalize_cameras

from multiprocessing import Pool
import tqdm
from util.camera_transform import adjust_camera_to_bbox_crop_, adjust_camera_to_image_scale_, bbox_xyxy_to_xywh

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True

LINEMOID_OBJECTS = {
    'ape': "ape", 
    "benchvise": "benchvise",  
    'cam': "cam", 
    "can": "can",
    "cat": "cat", 
    "driller": "driller", 
    "duck": "duck", 
    "eggbox": "eggbox", 
    "glue": "glue", 
    "holepuncher": "holepuncher", 
    "iron": "iron", 
    "lamp": "lamp", 
    "phone": "phone",
    }


class LINEMOD_Dataset_BOP(torch.utils.data.Dataset):
    def __init__(self, 
                 data_root,  # /scratch/liudan/data/linemod
                 obj_name, 
                 subset_mode='train', 
                 obj_database_dir=None, 
                 load_gt_bbox=False, 
                 use_gt_mask=False,
                 use_binarized_mask=False,
                 num_refer_views=-1, # default -1 means all views
                 load_yolo_det=False, 
                 num_grid_points=4096):

        self.name2classID = {'ape': 1, "benchvise": 2,  
                            'cam': 4, "can": 5,
                            "cat": 6, "driller": 8, 
                            "duck": 9, "eggbox": 10, 
                            "glue": 11, "holepuncher": 12, 
                            "iron": 13, "lamp": 14, "phone": 15,
              }

        assert(subset_mode in ['train', 'test']), f'{subset_mode} is not a valid subset mode [train, test]'
        assert(obj_name in LINEMOID_OBJECTS.keys()), f'{obj_name} is not in the LINEMOD {LINEMOID_OBJECTS}'

        self.obj_name = obj_name
        self.data_root = data_root
        self.subset_mode = subset_mode
        self.num_grid_points = num_grid_points
        self.obj_classID = self.name2classID[self.obj_name]
        self.obj_dir = os.path.join(self.data_root, 'LINEMOD/objects', '{:06}'.format(self.obj_name))

        self.use_gt_mask = use_gt_mask
        self.load_gt_bbox = load_gt_bbox
        self.load_yolo_det = load_yolo_det
        self.obj_database_dir = obj_database_dir
        self.use_binarized_mask = use_binarized_mask
        self.to_meter_scale = 1e-3
        
        self.camK = np.array([
            [572.4114 ,   0.     , 325.2611 ],
            [  0.     , 573.57043, 242.049  ],
            [  0.     ,   0.     ,   1.     ]], dtype=np.float32)
        
        model_dir = os.path.join(self.data_root, 'models/models')
        if not os.path.exists(model_dir):
            model_dir = os.path.join(self.data_root, 'models/models_eval')
        model_file = os.path.join(model_dir, f'models_info.json')
        with open(model_file, 'r') as f:
            self.model_info = json.load(f)

        self.obj_model_info = self.model_info[str(self.obj_classID)]
        self.diameter = self.obj_model_info['diameter'] * self.to_meter_scale # convert to m

        self.is_symmetric = False
        for _key, _val in self.obj_model_info.items():
            if 'symmetries' in _key:
                self.is_symmetric = True

        bbox3d_xyz = np.array([self.obj_model_info["size_x"],
                                self.obj_model_info["size_y"],
                                self.obj_model_info["size_z"],
                            ]) * self.to_meter_scale # convert to m
        self.obj_bbox3d = np.array([
            [-bbox3d_xyz[0], -bbox3d_xyz[0], -bbox3d_xyz[0], -bbox3d_xyz[0],  bbox3d_xyz[0],  bbox3d_xyz[0],  bbox3d_xyz[0],  bbox3d_xyz[0]],
            [-bbox3d_xyz[1], -bbox3d_xyz[1],  bbox3d_xyz[1],  bbox3d_xyz[1], -bbox3d_xyz[1], -bbox3d_xyz[1],  bbox3d_xyz[1],  bbox3d_xyz[1]],
            [-bbox3d_xyz[2],  bbox3d_xyz[2],  bbox3d_xyz[2], -bbox3d_xyz[2], -bbox3d_xyz[2],  bbox3d_xyz[2],  bbox3d_xyz[2], -bbox3d_xyz[2]]
            ]).T / 2 
        self.bbox3d_diameter = np.linalg.norm(bbox3d_xyz)
        
        self.obj_ply_path = os.path.join(model_dir, 'obj_{:06}.ply'.format(self.obj_classID))
        self.obj_pointcloud = py3d_io.load_ply(self.obj_ply_path)[0].numpy() * self.to_meter_scale # convert to m

        # yolo_detection_dir = os.path.join(DATASPACE_DIR, 'bop_dataset/lm_yolo_detection/val')
        # obj_yolo_detect_name = '08{:02d}-lm{}-others'.format(self.obj_classID, self.obj_classID)
        # obj_yolo_label_dir = os.path.join(yolo_detection_dir, obj_yolo_detect_name, 'labels')
        
        self.poses_file = os.path.join(self.obj_dir, 'scene_gt.json')
        with open(self.poses_file, 'r') as f:
            self.poses_info = json.load(f)

        if self.load_gt_bbox:
            self.bboxes_file = os.path.join(self.obj_dir, 'scene_gt_info.json')
            with open(self.bboxes_file, 'r') as f:
                self.bboxes_info = json.load(f)
            self.gt_bboxes = dict()

        self.poses = list()
        self.image_IDs = list()
        self.allo_poses = list()
        self.image_paths = list()
        self.yolo_bboxes = dict()
        image_subset_lists = gs_utils.read_list_data_from_txt(os.path.join(self.obj_dir, f'{self.subset_mode}.txt'))
        for idx, img_inst in enumerate(image_subset_lists):
            image_ID = int(img_inst)
            image_path = os.path.join(self.obj_dir, 'rgb', '{:06d}.png'.format(image_ID))
            pose_RT = self.poses_info[str(image_ID)][0]
            obj_pose = np.eye(4)
            obj_pose[:3, :3] = np.array(pose_RT['cam_R_m2c'], dtype=np.float32).reshape(3, 3)
            obj_pose[:3, 3] = np.array(pose_RT['cam_t_m2c'], dtype=np.float32).reshape(3) * self.to_meter_scale # convert to m

            self.poses.append(obj_pose)
            self.image_paths.append(image_path)

            allo_pose = obj_pose.copy() 
            allo_pose[:3, :3] = gs_utils.egocentric_to_allocentric(allo_pose)[:3, :3]
            self.allo_poses.append(allo_pose)

            self.image_IDs.append(image_ID)

            if self.load_gt_bbox:
                gt_x1, gt_y1, gt_bw, gt_bh = self.bboxes_info[str(image_ID)][0]['bbox_visib']
                gt_x2 = gt_x1 + gt_bw
                gt_y2 = gt_y1 + gt_bh
                self.gt_bboxes[image_ID] = np.array([gt_x1, gt_y1, gt_x2, gt_y2])

            if self.load_yolo_det:
                yolo_bbox_path = os.path.join(obj_yolo_label_dir, '{:06d}.txt'.format(image_ID + 1)) # yolo_results starts from 1
                if os.path.exists(yolo_bbox_path):
                    yolo_box = np.loadtxt(yolo_bbox_path)
                    assert yolo_box.shape[0] != 0, f"img id:{image_ID} no box detected!"
                    if len(yolo_box.shape) == 2:
                        want_id = np.argsort(yolo_box[:,5])[0]
                        yolo_box = yolo_box[want_id]
                    x_c_n, y_c_n, w_n, h_n = yolo_box[1:5]
                    x0_n, y0_n = x_c_n - w_n / 2, y_c_n - h_n / 2
                    x1_n, y1_n = x_c_n + w_n / 2, y_c_n + h_n / 2
                    self.yolo_bboxes[image_ID] = np.array([x0_n, y0_n, x1_n, y1_n])
                # else:
                #     self.yolo_bboxes[image_ID] = np.array([0, 0, 1.0, 1.0])
        
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        data_dict = dict()
        camK = self.camK
        pose = self.poses[idx]
        image_ID = self.image_IDs[idx]
        allo_pose = self.allo_poses[idx]
        image_path = self.image_paths[idx]
        image = np.array(Image.open(image_path), dtype=np.uint8) / 255.0

        data_dict['camK'] = torch.as_tensor(camK, dtype=torch.float32) 
        data_dict['pose'] = torch.as_tensor(pose, dtype=torch.float32)
        data_dict['image'] = torch.as_tensor(image, dtype=torch.float32)    
        data_dict['allo_pose'] = torch.as_tensor(allo_pose, dtype=torch.float32)

        data_dict['image_ID'] = image_ID
        data_dict['image_path'] = image_path

        if self.use_gt_mask:
            mask_path = os.path.join(self.obj_dir, 'mask_visib', f'{image_ID:06d}_000000.png')
            data_dict['gt_mask_path'] = mask_path
        
        if self.obj_database_dir is not None:
            data_dict['coseg_mask_path'] = os.path.join(self.obj_database_dir, 'pred_coseg_mask', '{:06d}.png'.format(image_ID))

        if self.load_yolo_det and self.yolo_bboxes.get(image_ID, None) is not None:
            img_hei, img_wid = image.shape[:2]
            x0_n, y0_n, x1_n, y1_n = self.yolo_bboxes[image_ID]
            x0_n, x1_n = x0_n * img_wid, x1_n * img_wid
            y0_n, y1_n = y0_n * img_hei, y1_n * img_hei
            
            bbox_xyxy = np.array([x0_n, y0_n, x1_n, y1_n])
            data_dict['yolo_bbox'] = torch.as_tensor(bbox_xyxy, dtype=torch.float32)

            bbox_scale = max(x1_n - x0_n, y1_n - y0_n)
            bbox_center = np.array([(x0_n + x1_n) / 2, (y0_n + y1_n) / 2])
            data_dict['bbox_scale'] = torch.as_tensor(bbox_scale, dtype=torch.float32)
            data_dict['bbox_center'] = torch.as_tensor(bbox_center, dtype=torch.float32)
            
        if self.load_gt_bbox:
            x1, y1, x2, y2 = self.gt_bboxes[image_ID]
            bbox_scale = max(x2 - x1, y2 - y1)
            bbox_center = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
            data_dict['gt_bbox_scale'] = torch.as_tensor(bbox_scale, dtype=torch.float32)
            data_dict['gt_bbox_center'] = torch.as_tensor(bbox_center, dtype=torch.float32)


        return data_dict

    def collate_fn(self, batch):
        """
        batchify the data
        """
        new_batch = dict()
        for each_dat in batch:
            for key, val in each_dat.items():
                if key not in new_batch:
                    new_batch[key] = list()
                new_batch[key].append(val)

        for key, val in new_batch.items():
            new_batch[key] = torch.stack(val, dim=0)

        return new_batch