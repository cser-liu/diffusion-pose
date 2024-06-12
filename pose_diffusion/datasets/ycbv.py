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
import tqdm
import mmcv
from transforms3d.quaternions import mat2quat, quat2mat
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
        ycbv_DIR=None,
        ycbv_ANNOTATION_DIR = None, 
        center_box=True,
        crop_longest=False,
        sort_by_filename=False,
        compute_optical=False,
        color_aug=True,
        erase_aug=False,
    ):
        
        self.ycbv_DIR = ycbv_DIR
        if ycbv_DIR == None:
            raise NotImplementedError
        print(f"ycbv_DIR is {ycbv_DIR}")

        
        if split == "train":
            split_name = "train_pbr"
            self.scenes = [f"{i:06d}" for i in range(50)]
        elif split == "test":
            split_name = "test"
            self.scenes = [f"{i:06d}" for i in range(48, 60)]

        self.cat_ids = [cat_id for cat_id, obj_name in id2obj.items()]
        self.cat2label = {v: i for i, v in enumerate(self.cat_ids)}  # id_map

        # /scratch/liudan/data/ycbv/...
        self.data_dir = os.path.join(ycbv_DIR, split_name)
        

        for scene in tqdm(self.scenes):
            scene_id = int(scene)
            scene_root = osp.join(self.data_dir, scene)

            gt_dict = mmcv.load(osp.join(scene_root, "scene_gt.json"))
            gt_info_dict = mmcv.load(osp.join(scene_root, "scene_gt_info.json"))
            cam_dict = mmcv.load(osp.join(scene_root, "scene_camera.json"))

            for str_im_id in tqdm(gt_dict, postfix=f"{scene_id}"):
                int_im_id = int(str_im_id)
                # int_im_id: 000000 - 000999
                rgb_path = osp.join(scene_root, "rgb/{:06d}.jpg").format(int_im_id)
                assert osp.exists(rgb_path), rgb_path

                depth_path = osp.join(scene_root, "depth/{:06d}.png".format(int_im_id))

                scene_im_id = f"{scene_id}/{int_im_id}"

                K = np.array(cam_dict[str_im_id]["cam_K"], dtype=np.float32).reshape(3, 3)
                depth_factor = 1000.0 / cam_dict[str_im_id]["depth_scale"]  # 10000

                record = {
                    "dataset_name": self.name,
                    "file_name": osp.relpath(rgb_path, PROJ_ROOT),
                    "depth_file": osp.relpath(depth_path, PROJ_ROOT),
                    "height": self.height,
                    "width": self.width,
                    "image_id": int_im_id,
                    "scene_im_id": scene_im_id,  # for evaluation
                    "cam": K,
                    "depth_factor": depth_factor,
                    "img_type": "syn_pbr",  # NOTE: has background
                }

                insts = []
                for anno_i, anno in enumerate(gt_dict[str_im_id]):
                    obj_id = anno["obj_id"]
                    if obj_id not in self.cat_ids:
                        continue

                    cur_label = self.cat2label[obj_id]  # 0-based label
                    R = np.array(anno["cam_R_m2c"], dtype="float32").reshape(3, 3)
                    t = np.array(anno["cam_t_m2c"], dtype="float32") / 1000.0
                    pose = np.hstack([R, t.reshape(3, 1)])  #3x4
                    quat = mat2quat(R).astype("float32")

                    proj = (record["cam"] @ t.T).T
                    proj = proj[:2] / proj[2]

                    bbox_visib = gt_info_dict[str_im_id][anno_i]["bbox_visib"]
                    bbox_obj = gt_info_dict[str_im_id][anno_i]["bbox_obj"]
                    x1, y1, w, h = bbox_visib
                    if self.filter_invalid:
                        if h <= 1 or w <= 1:
                            self.num_instances_without_valid_box += 1
                            continue

                    mask_file = osp.join(scene_root, "mask/{:06d}_{:06d}.png".format(int_im_id, anno_i))
                    mask_visib_file = osp.join(scene_root, "mask_visib/{:06d}_{:06d}.png".format(int_im_id, anno_i))
                    assert osp.exists(mask_file), mask_file
                    assert osp.exists(mask_visib_file), mask_visib_file
                    # load mask visib  TODO: load both mask_visib and mask_full
                    mask_single = mmcv.imread(mask_visib_file, "unchanged")
                    area = mask_single.sum()
                    if area < 3:  # filter out too small or nearly invisible instances
                        self.num_instances_without_valid_segmentation += 1
                        continue

                    visib_fract = gt_info_dict[str_im_id][anno_i].get("visib_fract", 1.0)

                    mask_rle = binary_mask_to_rle(mask_single, compressed=True)

                    xyz_path = osp.join(self.xyz_root, f"{scene_id:06d}/{int_im_id:06d}_{anno_i:06d}-xyz.pkl")
                    assert osp.exists(xyz_path), xyz_path
                    
                    inst = {
                        "category_id": cur_label,  # 0-based label
                        "bbox": bbox_visib,  # TODO: load both bbox_obj and bbox_visib
                        "bbox_mode": BoxMode.XYWH_ABS,
                        "pose": pose,
                        "quat": quat,
                        "trans": t,
                        "centroid_2d": proj,  # absolute (cx, cy)
                        "segmentation": mask_rle,
                        "mask_full_file": mask_file,  # TODO: load as mask_full, rle
                        "visib_fract": visib_fract,
                        "xyz_path": xyz_path,
                    }

                    model_info = self.models_info[str(obj_id)]
                    inst["model_info"] = model_info
                    # TODO: using full mask and full xyz
                    for key in ["bbox3d_and_center"]:
                        inst[key] = self.models[cur_label][key]
                    insts.append(inst)
                if len(insts) == 0:  # filter im without anno
                    continue
                record["annotations"] = insts
                dataset_dicts.append(record)
        
        
        
        self.center_box = center_box
        self.crop_longest = crop_longest
        self.min_num_images = min_num_images            

        self.build_dataset()
        
        self.sequence_list = sorted(list(self.wholedata.keys()))
        self.sequence_list_len = len(self.sequence_list)

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

    def build_dataset(self):
        self.wholedata = {}
        
        cached_pkl = os.path.join(os.path.dirname(os.path.dirname(self.scene_info_dir)), "processed.pkl")
        
        if os.path.exists(cached_pkl):
            # if you have processed annos, load it 
            with open(cached_pkl, 'rb') as file: 
                self.wholedata = pickle.load(file)
        else:
            for scene in self.scenes:
                print(scene)
                scene_name = "re10k" + scene
                
                scene_info_name = os.path.join(self.scene_info_dir, os.path.basename(scene)+".txt")
                scene_info = np.loadtxt(scene_info_name, delimiter=' ', dtype=np.float64, skiprows=1)

                filtered_data = []

                for frame_idx in range(len(scene_info)):
                    # using try here because some images may be missing
                    try:
                        raw_line = scene_info[frame_idx]
                        timestamp = raw_line[0]
                        intrinsics = raw_line[1:7]
                        extrinsics = raw_line[7:]

                        imgpath = os.path.join(self.train_dir, scene, '%s'%int(timestamp)+'.png')
                        image_size = Image.open(imgpath).size
                        posemat = extrinsics.reshape(3, 4).astype('float64')
                        focal_length = intrinsics[:2]* image_size
                        principal_point = intrinsics[2:4]* image_size

                        data = {
                                "filepath": imgpath,
                                "R": posemat[:3,:3],
                                "T": posemat[:3,-1],
                                "focal_length": focal_length,
                                "principal_point":principal_point,  
                            }

                        filtered_data.append(data)
                    except:
                        print("this image is missing")
                        
                if len(filtered_data) > self.min_num_images:
                    self.wholedata[scene_name] = filtered_data
                else:
                    print(f"scene {scene_name} does not have enough image nums")
                    
            print("finished")


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
        metadata = self.wholedata[sequence_name]
        ids = np.random.choice(len(metadata), n_per_seq, replace=False)
        return self.get_data(index=index, ids=ids)

    def get_data(self, index=None, sequence_name=None, ids=(0, 1), no_images=False, return_path=False):
        if sequence_name is None:
            sequence_name = self.sequence_list[index]
            
        metadata = self.wholedata[sequence_name]

        assert len(np.unique(ids)) == len(ids)
                
        annos = [metadata[i] for i in ids]

        if self.sort_by_filename:
            annos = sorted(annos, key=lambda x: x["filepath"])

        images = []
        rotations = []
        translations = []
        focal_lengths = []
        principal_points = []
        image_paths = []
        
        for anno in annos:
            filepath = anno["filepath"]
            image_path = osp.join(self.Re10K_DIR, filepath)
            image = Image.open(image_path).convert("RGB")

            images.append(image)
            rotations.append(torch.tensor(anno["R"]))
            translations.append(torch.tensor(anno["T"]))
            image_paths.append(image_path)

            ######## to make the convention of pytorch 3D happy          
            # Raw FL PP. If you want to use them, uncomment here
            # focal_lengths_raw.append(torch.tensor(anno["focal_length"]))
            # principal_points_raw.append(torch.tensor(anno["principal_point"]))
            
            # PT3D FL PP
            original_size_wh = np.array(image.size)
            scale = min(original_size_wh) / 2
            c0 = original_size_wh / 2.0
            focal_pytorch3d = anno["focal_length"] / scale
            # mirrored principal point
            p0_pytorch3d = -(anno["principal_point"] - c0) / scale
            focal_lengths.append(torch.tensor(focal_pytorch3d))
            principal_points.append(torch.tensor(p0_pytorch3d))
            ########


        crop_parameters = []
        images_transformed = []

        new_fls = []
        new_pps = []


        for i, (anno, image) in enumerate(zip(annos, images)):
            w, h = image.width, image.height

            if self.crop_longest:
                crop_dim = max(h, w)
                top = (h - crop_dim) // 2
                left = (w - crop_dim) // 2
                bbox = np.array([left, top, left + crop_dim, top + crop_dim])            
            elif self.center_box:
                crop_dim = min(h, w)
                top = (h - crop_dim) // 2
                left = (w - crop_dim) // 2
                bbox = np.array([left, top, left + crop_dim, top + crop_dim])
            else:
                bbox = np.array(anno["bbox"])

            if self.eval_time:
                bbox_jitter = bbox
            else:
                bbox_jitter = self._jitter_bbox(bbox)

            bbox_xywh = torch.FloatTensor(bbox_xyxy_to_xywh(bbox_jitter))

            ### cropping images
            focal_length_cropped, principal_point_cropped = adjust_camera_to_bbox_crop_(
                focal_lengths[i], principal_points[i], torch.FloatTensor(image.size), bbox_xywh
            )

            image = self._crop_image(image, bbox_jitter, white_bg=self.mask_images)

            ### resizing images
            new_focal_length, new_principal_point = adjust_camera_to_image_scale_(
                focal_length_cropped, 
                principal_point_cropped, 
                torch.FloatTensor(image.size), 
                torch.FloatTensor([self.img_size, self.img_size])
            )
            
            images_transformed.append(self.transform(image))
            new_fls.append(new_focal_length)
            new_pps.append(new_principal_point)

            crop_center = (bbox_jitter[:2] + bbox_jitter[2:]) / 2
            cc = (2 * crop_center / min(h, w)) - 1
            crop_width = 2 * (bbox_jitter[2] - bbox_jitter[0]) / min(h, w)

            crop_parameters.append(torch.tensor([-cc[0], -cc[1], crop_width]).float())

        ################################################################
        images = images_transformed

        new_fls = torch.stack(new_fls)
        new_pps = torch.stack(new_pps)

        batchR = torch.cat([torch.tensor(data["R"][None]) for data in annos])
        batchT = torch.cat([torch.tensor(data["T"][None]) for data in annos])
        
        # From COLMAP to PT3D
        batchR = batchR.clone().permute(0, 2, 1)
        batchR[:, :, :2] *= -1
        batchT[:, :2] *= -1

        cameras = PerspectiveCameras(focal_length=new_fls.float(), 
                                     principal_point=new_pps.float(), 
                                     R=batchR.float(), 
                                     T=batchT.float())

        if self.normalize_cameras:
            ################################################################################################################            
            norm_cameras = normalize_cameras(
                cameras,
                compute_optical=self.compute_optical, 
                first_camera=self.first_camera_transform, 
                normalize_T=True,
            )
            if norm_cameras == -1:
                print("Error in normalizing cameras: camera scale was 0")
                raise RuntimeError
        else:
            raise NotImplementedError("please normalize cameras")
            

        crop_params = torch.stack(crop_parameters)


        batch = {
            "seq_name": sequence_name,
            "frame_num": len(metadata),
        }

        # Add images
        if self.transform is not None:
            images = torch.stack(images)
    
        if self.color_aug and (not self.eval_time):
            for augidx in range(len(images)):
                if self.erase_aug:
                    if random.random() < 0.15:
                        ex, ey, eh, ew, ev = self.rand_erase.get_params(images[augidx], scale=self.rand_erase.scale,ratio=self.rand_erase.ratio, value=[self.rand_erase.value])
                        images[augidx] = transforms.functional.erase(images[augidx], ex, ey, eh, ew, ev, self.rand_erase.inplace)

                images[augidx] = self.color_jitter(images[augidx])

        batch["image"] = images.clamp(0,1)

        batch["R"] = norm_cameras.R
        batch["T"] = norm_cameras.T

        batch["fl"] = norm_cameras.focal_length
        batch["pp"] = norm_cameras.principal_point
        batch["crop_params"] = torch.stack(crop_parameters)
        
        
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