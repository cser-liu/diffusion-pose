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
import os.path as osp
import random

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
        category=("unseen",),
        split="test",
        transform=None,
        debug=False,
        random_aug=True,
        jitter_scale=[0.8, 1.2],
        jitter_trans=[-0.07, 0.07],
        min_num_images=50,
        img_size=224,
        eval_time=False,
        normalize_cameras=False,
        first_camera_transform=True,
        mask_images=False,
        LM_DIR=None,
        LM_ANNOTATION_DIR=None,
        foreground_crop=True,
        center_box=True,
        sort_by_filename=False,
        compute_optical=False,
        color_aug=True,
        erase_aug=False,
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
        if "seen" in category:
            category = TRAINING_CATEGORIES
            
        if "unseen" in category:
            category = TEST_CATEGORIES
        
        if "all" in category:
            category = TRAINING_CATEGORIES + TEST_CATEGORIES
        
        category = sorted(category)
        self.category = category

        if split == "train":
            split_name = "train"
        elif split == "test":
            split_name = "test"

        self.low_quality_translations = []
        self.rotations = {}
        self.category_map = {}

        if LM_DIR == None:
            raise ValueError("LINEMOD_DIR is not specified")

        print(f"LINEMOD_DIR is {LM_DIR}")

        self.LM_DIR = LM_DIR # /scratch/liudan/data/linemod
        self.LM_ANNOTATION_DIR = LM_ANNOTATION_DIR
        self.center_box = center_box
        self.split_name = split_name
        self.min_num_images = min_num_images
        self.foreground_crop = foreground_crop

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

        for c in category:
            # annotation_file = osp.join(self.CO3D_ANNOTATION_DIR, f"{c}_{split_name}.jgz")
            # with gzip.open(annotation_file, "r") as fin:
            #     annotation = json.loads(fin.read())
            obj_id = self.name2classID[c]
            model_path = osp.join(self.LM_DIR, "models/models/obj_{:06d}.ply".format(obj_id))
            self.obj_pointcloud = py3d_io.load_ply(model_path)[0].numpy()
            model_file = osp.join(self.LM_DIR, f'models/models/models_info.json')
            with open(model_file, 'r') as f:
                self.model_info = json.load(f)

            self.obj_model_info = self.model_info[str(obj_id)]
            self.diameter = self.obj_model_info['diameter'] * 1e-3

            obj_path = osp.join(self.LM_DIR, f"LINEMOD/objects/{c}")
            pose_dir = osp.join(obj_path, "pose")
            rgb_dir = osp.join(obj_path, "rgb")
            mask_dir = osp.join(obj_path, "mask")

            counter = 0
            self.rotations[c] = []
            for seq_id in range(1000):
                counter += 1

                pose_path = osp.join(pose_dir, "{:04d}.txt".format(seq_id))
                data = np.loadtxt(pose_path)
            
                seq_pose = np.array(data, dtype=np.float32).reshape(4, 4)
                R = seq_pose[:3, :3]
                T = seq_pose[:3, 3].reshape(3)
                # R = np.array(seq_pose['cam_R_m2c'], dtype=np.float32).reshape(3, 3)
                # T = np.array(seq_pose['cam_t_m2c'], dtype=np.float32).reshape(3)

                rgb_path = osp.join(rgb_dir, "{:04d}.jpg".format(seq_id))
                mask_path = osp.join(mask_dir, "{:04d}.png".format(seq_id))

                seq_info = {
                    "rgb_path": rgb_path,
                    "mask_path": mask_path,  
                    "T": T,
                    "R": R,
                    "cam": self.camK,
                }

                self.rotations[c].append(seq_info)

            print(counter)

        self.sequence_list = list(self.rotations.keys())
                
        self.split = split
        self.debug = debug
        self.sort_by_filename = sort_by_filename

        if transform is None:
            self.transform = transforms.Compose([transforms.ToTensor(), transforms.Resize(img_size, antialias=True)])
        else:
            self.transform = transform

        if random_aug and not eval_time:
            self.jitter_scale = jitter_scale
            self.jitter_trans = jitter_trans
        else:
            self.jitter_scale = [1, 1]
            self.jitter_trans = [0, 0]

        self.img_size = img_size
        self.eval_time = eval_time
        self.normalize_cameras = normalize_cameras
        self.first_camera_transform = first_camera_transform
        self.mask_images = mask_images
        self.compute_optical = compute_optical
        self.color_aug = color_aug
        self.erase_aug = erase_aug

        if self.color_aug:
            self.color_jitter = transforms.Compose(
                [
                    transforms.RandomApply(
                        [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)], p=0.65
                    ),
                    transforms.RandomGrayscale(p=0.15),
                ]
            )
        if self.erase_aug:
            self.rand_erase = transforms.RandomErasing(
                p=0.1, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False
            )

        print(f"Low quality translation sequences, not used: {self.low_quality_translations}")
        print(f"Data size: {len(self)}")

    def __len__(self):
        return len(self.sequence_list)

    def _jitter_bbox(self, bbox):
        # Random aug to bounding box shape

        bbox = square_bbox(bbox.astype(np.float32))
        s = np.random.uniform(self.jitter_scale[0], self.jitter_scale[1])
        tx, ty = np.random.uniform(self.jitter_trans[0], self.jitter_trans[1], size=2)

        side_length = bbox[2] - bbox[0]
        center = (bbox[:2] + bbox[2:]) / 2 + np.array([tx, ty]) * side_length
        extent = side_length / 2 * s

        # Final coordinates need to be integer for cropping.
        ul = (center - extent).round().astype(int)
        lr = ul + np.round(2 * extent).astype(int)
        return np.concatenate((ul, lr))

    def _crop_image(self, image, bbox, white_bg=False):
        if white_bg:
            # Only support PIL Images
            image_crop = Image.new("RGB", (bbox[2] - bbox[0], bbox[3] - bbox[1]), (255, 255, 255))
            image_crop.paste(image, (-bbox[0], -bbox[1]))
        else:
            image_crop = transforms.functional.crop(
                image, top=bbox[1], left=bbox[0], height=bbox[3] - bbox[1], width=bbox[2] - bbox[0]
            )
        return image_crop

    def __getitem__(self, idx_N):
        """Fetch item by index and a dynamic variable n_per_seq."""

        # Different from most pytorch datasets,
        # here we not only get index, but also a dynamic variable n_per_seq
        # supported by DynamicBatchSampler

        index, n_per_seq = idx_N
        sequence_name = self.sequence_list[index]
        metadata = self.rotations[sequence_name]
        ids = np.random.choice(len(metadata), n_per_seq, replace=False)
        return self.get_data(index=index, ids=ids)

    def get_data(self, index=None, sequence_name=None, ids=(0, 1), no_images=False, return_path = False):
        if sequence_name is None:
            sequence_name = self.sequence_list[index]
        metadata = self.rotations[sequence_name]
        # category = self.category_map[sequence_name]

        annos = [metadata[i] for i in ids]

        images = []
        pts = []
        rotations = []
        translations = []
        focal_lengths = []
        principal_points = []
        image_paths = []
        
        for anno in annos:
            image_path = anno["rgb_path"]
            image = Image.open(image_path).convert("RGB")

            if self.mask_images:
                white_image = Image.new("RGB", image.size, (255, 255, 255))

                mask_path = anno["mask_path"]
                mask = Image.open(mask_path).convert("L")

                if mask.size != image.size:
                    mask = mask.resize(image.size)
                mask = Image.fromarray(np.array(mask) > 125)
                image = Image.composite(image, white_image, mask)

            images.append(self.transform(image))
            rotations.append(torch.tensor(anno["R"]))
            translations.append(torch.tensor(anno["T"]))
            # focal_lengths.append(torch.tensor(anno["focal_length"]))
            # principal_points.append(torch.tensor(anno["principal_point"]))
            image_paths.append(image_path)
            # pts.append(anno["pts"])
            
        batch = {}

        batch["R"] = torch.stack(rotations)
        batch["T"] = torch.stack(translations)
        # batch["pts"] = torch.stack(pts)

        if self.transform is not None:
            images = torch.stack(images)

        if self.color_aug and (not self.eval_time):
            images = self.color_jitter(images)
            if self.erase_aug:
                images = self.rand_erase(images)

        batch["image"] = images

        if return_path:
            return batch, image_paths
        
        return batch


def square_bbox(bbox, padding=0.0, astype=None):
    """
    Computes a square bounding box, with optional padding parameters.

    Args:
        bbox: Bounding box in xyxy format (4,).

    Returns:
        square_bbox in xyxy format (4,).
    """
    if astype is None:
        astype = type(bbox[0])
    bbox = np.array(bbox)
    center = (bbox[:2] + bbox[2:]) / 2
    extents = (bbox[2:] - bbox[:2]) / 2
    s = max(extents) * (1 + padding)
    square_bbox = np.array([center[0] - s, center[1] - s, center[0] + s, center[1] + s], dtype=astype)
    return square_bbox


TRAINING_CATEGORIES = [
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

TEST_CATEGORIES = ["ape"]

DEBUG_CATEGORIES = []
