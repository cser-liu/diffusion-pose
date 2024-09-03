# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import time
import cv2
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

# from datasets.linemod_one import ALL_CATEGORIES, TEST_CATEGORIES
from datasets.linemod_one import lmDataset
from datasets.OnePose_test import OnePoseTestDataset
from util.match_extraction import extract_match
from util.geometry_guided_sampling import geometry_guided_sampling
# from util.metric import camera_to_rel_deg, calculate_auc_np
from util.metric import calc_pose_error, calc_add_metric, calc_projection_2d_error, calc_bbox_IOU, aggregate_metrics
from util.load_img_folder import load_and_preprocess_images
from util.train_util import (
    get_lm_dataset_test,
    set_seed_and_print,
    get_onepose_dataset_test
)




@hydra.main(config_path="../cfgs/", config_name="onepose_test")
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


    if cfg.test.category == "test":
        categories = ["aptamil"]
    else:
        categories = []

    print("-"*100)
    print(f"Testing on {categories}")
    print("-"*100)
    
    category_dict = {}
    metric_name = ["r_Error", "t_Error"]
    
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

        agg_metric = aggregate_metrics(error_dict)
        
        print("-"*100)
        print(f"Category {category} Done")  

        print_str = f"{category}: "
        for m_name in agg_metric.keys():  
            print_num = agg_metric[m_name]
            print_str = print_str + f"{m_name} is {print_num:.3f} | " 
            
        print(print_str)

    return True

def _test_one_category(model, category, cfg, num_frames, random_order, accelerator):
    model.eval()
    
    print(f"******************************** test on {category} ********************************")

    # Data loading
    test_dataset = OnePoseTestDataset(category=category, data_dir=cfg.test.data_dir, split='test')

    dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size = cfg.test.batch_size,
        shuffle = True,
    )    

    # pts = test_dataset.obj_pointcloud[category]
    # camK = test_dataset.camK # original K
    # diameter = test_dataset.diameter


    category_error_dict = {"R_errs":[], "t_errs":[], "ADD_metric":[], "Proj2D":[]}

    for step, batch in enumerate(dataloader):  

        # data preparation
        query_image = batch['query_image']['image'].to(accelerator.device) # bx3xhxw
        query_T = batch['query_image']['T'].to(accelerator.device) # bx3
        query_R = batch['query_image']['R'].to(accelerator.device)# bx3x3

        ref_images = batch['ref_images']['image'].to(accelerator.device) # bxrx3xhxw
        ref_T = batch['ref_images']['T'].to(accelerator.device) # bxrx3
        ref_R = batch['ref_images']['R'].to(accelerator.device) # bxrx3x3


        batch_size = query_image.shape[0]

        query_pose = {
            'R': query_R,
            'T': query_T
        }
        ref_pose = {
            'R': ref_R,
            'T': ref_T
        }

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

        with torch.no_grad():
            predictions = model(query_image, ref_images, query_pose, ref_pose, cond_fn=cond_fn, cond_start_step=cfg.GGS.start_step, training=False)

        pred_pose = predictions["pred_pose"] # object pose, b x (r+1) x 7
    
        pred_rot = pred_pose["R"].reshape(batch_size, -1, 3, 3) # b x (r+1) x 3 x 3
        pred_tran = pred_pose["T"].reshape(batch_size, -1, 3) # b x (r+1) x 3
        
        # compute metrics
        # r_error = 0
        # t_error = 0
        for i in range(batch_size):
            pred_RT = np.eye(4)
            pred_RT[:3, :3] = np.array(pred_rot[i][0].cpu(), dtype=np.float32)
            pred_RT[:3, 3] = np.array(pred_tran[i][0].cpu(), dtype=np.float32)

            gt_RT = np.eye(4)
            gt_RT[:3, :3] = np.array(query_R[i].cpu(), dtype=np.float32)
            gt_RT[:3, 3] = np.array(query_T[i].cpu(), dtype=np.float32)

            if i == 0:
                print(f"pred pose is {pred_RT}")
                print(f"gt pose is {gt_RT}")

            re, te = calc_pose_error(pred_RT, gt_RT, unit='cm')
            # add = calc_add_metric(model_3D_pts=pts, diameter=diameter, pose_pred=pred_RT, pose_target=gt_RT)
            # proj = calc_projection_2d_error(model_3D_pts=pts, pose_pred=pred_RT, pose_targets=gt_RT, K=camK)

            print(f"rotation loss is {re}, translation error is {te}")

            category_error_dict["R_errs"].append(re)
            category_error_dict["t_errs"].append(te)
            # category_error_dict["ADD_metric"].append(add)
            # category_error_dict["Proj2D"].append(proj)
    
    return category_error_dict


def prefix_with_module(checkpoint):
    prefixed_checkpoint = OrderedDict()
    for key, value in checkpoint.items():
        prefixed_key = "module." + key
        prefixed_checkpoint[prefixed_key] = value
    return prefixed_checkpoint


if __name__ == "__main__":
    test_fn()
