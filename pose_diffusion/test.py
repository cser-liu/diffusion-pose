# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import time
from collections import OrderedDict
from functools import partial
from typing import Dict, List, Optional, Union
from multiprocessing import Pool


import hydra
import torch
import numpy as np
from hydra.utils import instantiate, get_original_cwd
from accelerate import Accelerator
from omegaconf import DictConfig, OmegaConf
from pytorch3d.ops import corresponding_cameras_alignment
from pytorch3d.renderer.cameras import PerspectiveCameras
from pytorch3d.vis.plotly_vis import plot_scene

from datasets.linemod_test import TRAINING_CATEGORIES, TEST_CATEGORIES, DEBUG_CATEGORIES
from util.match_extraction import extract_match
from util.geometry_guided_sampling import geometry_guided_sampling
# from util.metric import camera_to_rel_deg, calculate_auc_np
from util.metric import calc_pose_error, calc_add_metric, calc_projection_2d_error, calc_bbox_IOU, aggregate_metrics
from util.load_img_folder import load_and_preprocess_images
from util.train_util import (
    get_lm_dataset_test,
    get_ycbv_dataset_test,
    set_seed_and_print,
)




@hydra.main(config_path="../cfgs/", config_name="lm_test")
def test_fn(cfg: DictConfig):
    OmegaConf.set_struct(cfg, False)
    accelerator = Accelerator(even_batches=False, device_placement=False)

    # Print configuration and accelerator state
    accelerator.print("Model Config:", OmegaConf.to_yaml(cfg), accelerator.state)

    torch.backends.cudnn.benchmark = cfg.test.cudnnbenchmark if not cfg.debug else False
    if cfg.debug:
        accelerator.print("********DEBUG MODE********")
        torch.backends.cudnn.deterministic = True

    set_seed_and_print(cfg.seed)

    # Model instantiation
    model = instantiate(cfg.MODEL, _recursive_=False)
    model = model.to(accelerator.device)

    # Accelerator setup
    model = accelerator.prepare(model)

    if cfg.test.resume_ckpt:
        accelerator.load_state(cfg.test.resume_ckpt)
        # try:
        #     model.load_state_dict(prefix_with_module(checkpoint), strict=True)
        # except:
        #     model.load_state_dict(checkpoint, strict=True)

        accelerator.print(f"Successfully resumed from {cfg.test.resume_ckpt}")


    categories = cfg.test.category

    # if "seen" in categories:
    #     categories = TRAINING_CATEGORIES
            
    if "unseen" in categories:
        categories = TEST_CATEGORIES
    
    if "debug" in categories:
        categories = DEBUG_CATEGORIES
    
    if "all" in categories:
        categories = TRAINING_CATEGORIES + TEST_CATEGORIES
    
    categories = sorted(categories)

    print("-"*100)
    print(f"Testing on {categories}")
    print("-"*100)
    
    category_dict = {}
    metric_name = ["r_Error", "t_Error", "ADD_metric", "Proj2D"]
    
    for m_name in metric_name:
        category_dict[m_name] = {}
    

    for category in categories:
        print("-"*100)
        print(f"Category {category} Start")

        error_dict = _test_one_category(
            model = model,
            category = category,
            cfg = cfg,
            num_frames = cfg.test.num_frames,
            random_order = cfg.test.random_order, 
            accelerator = accelerator,
        )

        category_dict["r_Error"][category] = error_dict["r_Error"]
        category_dict["t_Error"][category] = error_dict["t_Error"]
        category_dict["ADD_metric"][category] = error_dict["ADD_metric"]
        category_dict["Proj2D"][category] = error_dict["Proj2D"]
        
        print("-"*100)
        print(f"Category {category} Done")
    
    for m_name in metric_name:
        category_dict[m_name]["mean"] = np.mean(list(category_dict[m_name].values()))     

    for c_name in (categories + ["mean"]): 
        print_str = f"{c_name.ljust(20)}: "
        for m_name in metric_name:  
            print_num = np.mean(category_dict[m_name][c_name])
            print_str = print_str + f"{m_name} is {print_num:.3f} | " 
            
        if c_name == "mean":
            print("-"*100)
        print(print_str)
        

    return True

def _test_one_category(model, category, cfg, num_frames, random_order, accelerator):
    model.eval()
    
    print(f"******************************** test on {category} ********************************")

    # Data loading
    test_dataset = get_lm_dataset_test(cfg, category=category)
    
    category_error_dict = {"r_Error":[], "t_Error":[], "ADD_metric":[], "Proj2D":[]}
    
    for seq_name in test_dataset.sequence_list: 
        # print(f"Testing the sequence {seq_name.ljust(15, ' ')} of category {category.ljust(15, ' ')}")
        metadata = test_dataset.rotations[seq_name]
        print(f"length of metadata is {len(metadata)}")
        
        if len(metadata) < num_frames:
            print(f"Skip sequence {seq_name}")
            continue
        
        if random_order:
            ids = np.random.choice(len(metadata), num_frames, replace=False)
        else:
            raise ValueError("Please specify your own sampling strategy")
            
        batch, image_paths = test_dataset.get_data(sequence_name=seq_name, ids=ids, return_path = True)

        # Use load_and_preprocess_images() here instead of using batch["image"] as
        #       images = batch["image"].to(accelerator.device)
        # because we need bboxes_xyxy and resized_scales for GGS
        # TODO combine this into Co3D V2 dataset
        images, image_info = load_and_preprocess_images(image_paths = image_paths, image_size = cfg.test.img_size)
        images = images.to(accelerator.device)
        
        if cfg.GGS.enable:
            kp1, kp2, i12 = extract_match(image_paths = image_paths, image_info = image_info)
            
            if kp1 is not None:
                keys = ["kp1", "kp2", "i12", "img_shape"]
                values = [kp1, kp2, i12, images.shape]
                matches_dict = dict(zip(keys, values))

                cfg.GGS.pose_encoding_type = cfg.MODEL.pose_encoding_type
                GGS_cfg = OmegaConf.to_container(cfg.GGS)

                cond_fn = partial(geometry_guided_sampling, matches_dict=matches_dict, GGS_cfg=GGS_cfg)
                print("[92m=====> Sampling with GGS <=====[0m")
            else:
                cond_fn = None
        else:
            cond_fn = None
            print("[92m=====> Sampling without GGS <=====[0m")

            
        translation = batch["T"].to(accelerator.device)
        rotation = batch["R"].to(accelerator.device)
        # fl = batch["fl"].to(accelerator.device)
        # pp = batch["pp"].to(accelerator.device)
        pts = test_dataset.obj_pointcloud
        camK = test_dataset.camK
        diameter = test_dataset.diameter

        batch_size = len(images)
        # expand to 1 x N x 3 x H x W
        images = images.unsqueeze(0)
        
        print(f"image num is {batch_size}")

        with torch.no_grad():
            predictions = model(images, cond_fn=cond_fn, cond_start_step=cfg.GGS.start_step, training=False)

        pred_pose = predictions["pred_pose"]
        pred_rot = pred_pose["R"]
        pred_tran = pred_pose["T"]
        
        # compute metrics
        # r_error = 0
        # t_error = 0
        for i in range(batch_size):
            pred_RT = np.eye(4)
            pred_RT[:3, :3] = np.array(pred_rot[i].cpu(), dtype=np.float32)
            pred_RT[:3, 3] = np.array(pred_tran[i].cpu(), dtype=np.float32).reshape(3)

            gt_RT = np.eye(4)
            gt_RT[:3, :3] = np.array(rotation[i].cpu(), dtype=np.float32)
            gt_RT[:3, 3] = np.array(translation[i].cpu(), dtype=np.float32).reshape(3)

            print(f"pred pose is {pred_RT}")
            print(f"gt pose is {gt_RT}")

            re, te = calc_pose_error(pred_RT, gt_RT, unit='m')
            add = calc_add_metric(model_3D_pts=pts, diameter=diameter, pose_pred=pred_RT, pose_target=gt_RT, return_error=True)
            proj = calc_projection_2d_error(model_3D_pts=pts, pose_pred=pred_RT, pose_targets=gt_RT, K=camK)

            category_error_dict["r_Error"].append(re)
            category_error_dict["t_Error"].append(te)
            category_error_dict["ADD_metric"].append(add)
            category_error_dict["Proj2D"].append(proj)
    
    return category_error_dict


def prefix_with_module(checkpoint):
    prefixed_checkpoint = OrderedDict()
    for key, value in checkpoint.items():
        prefixed_key = "module." + key
        prefixed_checkpoint[prefixed_key] = value
    return prefixed_checkpoint


if __name__ == "__main__":
    test_fn()
