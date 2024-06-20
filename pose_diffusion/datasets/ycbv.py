import gzip
import json
import os.path as osp
import random
import os 

import numpy as np
import torch
from PIL import Image, ImageFile
from pytorch3d.renderer import PerspectiveCameras
from pytorch3d.renderer.cameras import get_ndc_to_screen_transform
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

import matplotlib.pyplot as plt


Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True


class YcbvDataset(Dataset):
    def __init__(
        self,
        split="train",
        transform=None,
        debug=False,
        random_aug=True,
        jitter_scale=[0.8, 1.0],
        jitter_trans=[-0.07, 0.07],
        min_num_images=50,
        img_size=224,
        eval_time=False,
        normalize_cameras=False,
        first_camera_transform=True,
        first_camera_rotation_only=False,
        mask_images=False,
        YCBV_DIR=None,
        center_box=True,
        crop_longest=False,
        sort_by_filename=False,
        compute_optical=False,
        color_aug=True,
        erase_aug=False,
    ):
        
        self.ycbv_DIR = YCBV_DIR
        if YCBV_DIR == None:
            raise NotImplementedError
        print(f"ycbv_DIR is {YCBV_DIR}")

        
        if split == "train":
            split_name = "train_pbr"
            self.scenes = [f"{i:06d}" for i in range(50)]
        elif split == "test":
            split_name = "test"
            self.scenes = [f"{i:06d}" for i in range(48, 60)]

        self.cat_ids = [cat_id for cat_id, obj_name in id2obj.items()]
        self.cat2label = {v: i for i, v in enumerate(self.cat_ids)}  # id_map

        # /scratch/liudan/data/ycbv/...
        self.data_dir = os.path.join(YCBV_DIR, split_name)
        
        self.rotations = {}
        self.category_map = {}

        for scene in tqdm(self.scenes):
            scene_id = int(scene)
            scene_root = osp.join(self.data_dir, scene)
            with open(osp.join(scene_root, "scene_gt.json"), "r") as f:
                gt_dict = json.load(f)
            with open(osp.join(scene_root, "scene_gt_info.json"), "r") as f_info:
                gt_info_dict = json.load(f_info)
            with open(osp.join(scene_root, "scene_camera.json"), "r") as f_cam:
                cam_dict = json.load(f_cam)

            # gt_info_dict = mmcv.load(osp.join(scene_root, "scene_gt_info.json"))
            # cam_dict = mmcv.load(osp.join(scene_root, "scene_camera.json"))

            for str_im_id in tqdm(gt_dict, postfix=f"{scene_id}"):
                int_im_id = int(str_im_id)
                # int_im_id: 000000 - 000999
                if split == "train":
                    rgb_path = osp.join(scene_root, "rgb/{:06d}.jpg").format(int_im_id)
                else:
                    rgb_path = osp.join(scene_root, "rgb/{:06d}.png").format(int_im_id)
                assert osp.exists(rgb_path), rgb_path

                depth_path = osp.join(scene_root, "depth/{:06d}.png".format(int_im_id))

                scene_im_id = f"{scene_id}/{int_im_id}"
                

                K = np.array(cam_dict[str_im_id]["cam_K"], dtype=np.float32)
                focal_length = np.array([K[0], K[4]], dtype=np.float32)
                principal_point = np.array([K[2], K[5]], dtype=np.float32)

                depth_factor = 1000.0 / cam_dict[str_im_id]["depth_scale"]  # 10000
                
                image_info = {
                    "rgb_path": rgb_path,
                    "depth_path": depth_path,
                    "image_id": int_im_id,
                    "scene_im_id": scene_im_id,  # for evaluation
                    "cam": K,
                    "depth_factor": depth_factor,
                }

                for anno_i, anno in enumerate(gt_dict[str_im_id]):
                    obj_id = anno["obj_id"]
                    if obj_id not in self.cat_ids:
                        continue

                    scene_obj_id = f"scene{scene_id}_obj{obj_id}"
                    self.category_map[scene_obj_id] = id2obj[obj_id]

                    if scene_obj_id not in self.rotations.keys():
                        self.rotations[scene_obj_id] = []

                    # cur_label = self.cat2label[obj_id]  # 0-based label

                    R = np.array(anno["cam_R_m2c"], dtype="float32").reshape(3, 3)
                    t = np.array(anno["cam_t_m2c"], dtype="float32") / 1000.0

                    pose = np.hstack([R, t.reshape(3, 1)])  #3x4
                    # quat = mat2quat(R).astype("float32")

                    # proj = (image_info["cam"] @ t.T).T
                    # proj = proj[:2] / proj[2]

                    bbox_visib = gt_info_dict[str_im_id][anno_i]["bbox_visib"]
                    bbox_obj = gt_info_dict[str_im_id][anno_i]["bbox_obj"]
                    x1, y1, w, h = bbox_visib
                    
                    mask_file = osp.join(scene_root, "mask/{:06d}_{:06d}.png".format(int_im_id, anno_i))
                    mask_visib_file = osp.join(scene_root, "mask_visib/{:06d}_{:06d}.png".format(int_im_id, anno_i))
                    assert osp.exists(mask_file), mask_file
                    assert osp.exists(mask_visib_file), mask_visib_file

                    # xyz_path = osp.join(self.xyz_root, f"{scene_id:06d}/{int_im_id:06d}_{anno_i:06d}-xyz.pkl")
                    # assert osp.exists(xyz_path), xyz_path

                    inst = {
                        "rgb_path": rgb_path,
                        "depth_path": depth_path,
                        "mask_path": mask_file,
                        "mask_visib_path": mask_visib_file,    
                        "category_id": obj_id,  
                        "bbox_visib": bbox_visib,  # x, y, w, h
                        "bbox": bbox_obj,
                        "pose": pose,
                        "T": t,
                        "R": R,
                        "cam": K,
                        "focal_length": focal_length,
                        "principal_point": principal_point,
                        "depth_scale": depth_factor,
                        # "centroid_2d": proj,  # absolute (cx, cy)
                        
                    }
                    inst["image"] = image_info

                    self.rotations[scene_obj_id].append(inst)

        self.sequence_list = list(self.rotations.keys())
        # print(f"sequence nums: {self.sequence_list}")
        
        self.center_box = center_box
        self.crop_longest = crop_longest
        self.min_num_images = min_num_images            

        self.debug = debug
        self.sort_by_filename = sort_by_filename

        if transform is None:
            self.transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Resize(img_size, antialias=True),
                ]
            )
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
                    transforms.RandomApply([transforms.ColorJitter(brightness=0.3, contrast=0.4, saturation=0.2, hue=0.1)], p=0.75),
                    transforms.RandomGrayscale(p=0.05),
                    transforms.RandomApply([transforms.GaussianBlur(5, sigma=(0.1, 1.0))], p=0.05),
                ]
            )

        if self.erase_aug:
            self.rand_erase = transforms.RandomErasing(p=0.1, scale=(0.02, 0.05), ratio=(0.3, 3.3), value=0, inplace=False)


        print(f"Data size: {len(self)}")

    def __len__(self):
        return len(self.sequence_list)

    def _jitter_bbox(self, bbox):
        # Random aug to cropping box shape
        
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
            image_crop = transforms.functional.crop(image, top=bbox[1], left=bbox[0], height=bbox[3] - bbox[1], width=bbox[2] - bbox[0])

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
        category = self.category_map[sequence_name]

        annos = [metadata[i] for i in ids]

        if self.sort_by_filename:
            annos = sorted(annos, key=lambda x: x["image"]["rgb_path"])

        images = []
        rotations = []
        translations = []
        focal_lengths = []
        principal_points = []
        image_paths = []
        
        for anno in annos:
            rgb_path = anno["rgb_path"]
            image = Image.open(rgb_path).convert("RGB")

            if self.mask_images:
                white_image = Image.new("RGB", image.size, (255, 255, 255))

                mask_path = anno["mask_path"]
                mask = Image.open(mask_path).convert("L")

                if mask.size != image.size:
                    mask = mask.resize(image.size)
                mask = Image.fromarray(np.array(mask) > 125)
                image = Image.composite(image, white_image, mask)

            images.append(image)
            rotations.append(torch.tensor(anno["R"]))
            translations.append(torch.tensor(anno["T"]))
            focal_lengths.append(torch.tensor(anno["focal_length"]))
            principal_points.append(torch.tensor(anno["principal_point"]))
            image_paths.append(rgb_path)
            
        crop_parameters = []
        images_transformed = []

        new_fls = []
        new_pps = []

        for i, (anno, image) in enumerate(zip(annos, images)):
            w, h = image.width, image.height

            if self.center_box:
                min_dim = min(h, w)
                top = (h - min_dim) // 2
                left = (w - min_dim) // 2
                bbox = np.array([left, top, left + min_dim, top + min_dim])
            else:
                bbox = np.array(anno["bbox"])
                # xywh -> xyxy
                xy = bbox[:2] + bbox[2:]
                bbox = np.concatenate(bbox[:2], xy)

            if not self.eval_time:
                bbox_jitter = self._jitter_bbox(bbox)
            else:
                bbox_jitter = bbox


            bbox_xywh = torch.FloatTensor(bbox_xyxy_to_xywh(bbox_jitter))
            (focal_length_cropped, principal_point_cropped) = adjust_camera_to_bbox_crop_(
                focal_lengths[i], principal_points[i], torch.FloatTensor(image.size), bbox_xywh
            )

            image = self._crop_image(image, bbox_jitter, white_bg=self.mask_images)

            (new_focal_length, new_principal_point) = adjust_camera_to_image_scale_(
                focal_length_cropped,
                principal_point_cropped,
                torch.FloatTensor(image.size),
                torch.FloatTensor([self.img_size, self.img_size]),
            )

            new_fls.append(new_focal_length)
            new_pps.append(new_principal_point)

            images_transformed.append(self.transform(image))
            crop_center = (bbox_jitter[:2] + bbox_jitter[2:]) / 2
            cc = (2 * crop_center / min(h, w)) - 1
            crop_width = 2 * (bbox_jitter[2] - bbox_jitter[0]) / min(h, w)

            crop_parameters.append(torch.tensor([-cc[0], -cc[1], crop_width]).float())

        images = images_transformed

        batch = {"seq_id": sequence_name, "category": category, "n": len(metadata), "ind": torch.tensor(ids)}

        new_fls = torch.stack(new_fls)
        new_pps = torch.stack(new_pps)

        if self.normalize_cameras:
            cameras = PerspectiveCameras(
                focal_length=new_fls.numpy(),
                principal_point=new_pps.numpy(),
                R=[data["R"] for data in annos],
                T=[data["T"] for data in annos],
            )

            normalized_cameras = normalize_cameras(
                cameras, compute_optical=self.compute_optical, first_camera=self.first_camera_transform
            )

            if normalized_cameras == -1:
                print("Error in normalizing cameras: camera scale was 0")
                raise RuntimeError

            batch["R"] = normalized_cameras.R
            batch["T"] = normalized_cameras.T
            batch["crop_params"] = torch.stack(crop_parameters)
            batch["R_original"] = torch.stack([torch.tensor(anno["R"]) for anno in annos])
            batch["T_original"] = torch.stack([torch.tensor(anno["T"]) for anno in annos])

            batch["fl"] = normalized_cameras.focal_length
            batch["pp"] = normalized_cameras.principal_point

            if torch.any(torch.isnan(batch["T"])):
                print(ids)
                print(category)
                print(sequence_name)
                raise RuntimeError

        else:
            batch["R"] = torch.stack(rotations)
            batch["T"] = torch.stack(translations)
            batch["crop_params"] = torch.stack(crop_parameters)
            batch["fl"] = new_fls
            batch["pp"] = new_pps

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