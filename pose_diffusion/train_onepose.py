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
import cv2
import numpy as np

import hydra
import torch
from hydra.utils import instantiate, get_original_cwd
from accelerate import Accelerator
from omegaconf import DictConfig, OmegaConf
from pytorch3d.renderer.cameras import PerspectiveCameras
from pytorch3d.vis.plotly_vis import plot_scene
from util.train_util import (
    DynamicBatchSampler,
    VizStats,
    WarmupCosineRestarts,
    get_onepose_dataset,
    plotly_scene_visualization,
    set_seed_and_print,
    view_color_coded_images_for_visdom,
)

# import torch.distributed as dist
#dist.init_process_group(backend='nccl', init_method='env://', timeout=datetime.timedelta(seconds=5400)) 


@hydra.main(config_path="../cfgs/", config_name="onepose_train")
def train_fn(cfg: DictConfig):
    OmegaConf.set_struct(cfg, False)
    accelerator = Accelerator(even_batches=False, device_placement=False)

    # Print configuration and accelerator state
    accelerator.print("Model Config:", OmegaConf.to_yaml(cfg), accelerator.state)

    torch.backends.cudnn.benchmark = cfg.train.cudnnbenchmark if not cfg.debug else False
    if cfg.debug:
        accelerator.print("********DEBUG MODE********")
        torch.backends.cudnn.deterministic = True

    set_seed_and_print(cfg.seed)

    # Visualization setup
    if accelerator.is_main_process:
        try:
            from visdom import Visdom

            viz = Visdom()
            # cams_show = {"ours_pred": pred_cameras, "ours_pred_aligned": pred_cameras_aligned, "gt_cameras": gt_cameras}
            # fig = plot_scene({f"{folder_path}": cams_show})
            # viz.plotlyplot(fig, env="visual", win="cams")
        except:
            print("Warning: please check your visdom connection for visualization")

    # Data loading
    dataset = get_onepose_dataset(cfg)
    print(f"len of dataset is {len(dataset)}")
    # print(f"len of eval_dataset is {len(eval_dataset)}")

    # dataloader = get_dataloader(cfg, dataset)
    # eval_dataloader = get_dataloader(cfg, eval_dataset, is_eval=True)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size = cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
        shuffle = True,
        pin_memory=cfg.train.pin_memory,
        persistent_workers=cfg.train.persistent_workers,
        drop_last=True,
    )

    accelerator.print("length of train dataloader is: ", len(dataloader))
    # accelerator.print("length of eval dataloader is: ", len(eval_dataloader))

    # Model instantiation
    model = instantiate(cfg.MODEL, _recursive_=False)
    model = model.to(accelerator.device)

    # Optimizer and Scheduler
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=cfg.train.lr)

    lr_scheduler = WarmupCosineRestarts(
        optimizer=optimizer, T_0=cfg.train.restart_num, iters_per_epoch=len(dataloader), warmup_ratio=0.1
    )
    # torch.optim.lr_scheduler.OneCycleLR() can achieve similar performance

    # Accelerator setup
    model, dataloader, optimizer, lr_scheduler = accelerator.prepare(model, dataloader, optimizer, lr_scheduler)

    start_epoch = 1
    if cfg.train.resume_ckpt:
        accelerator.load_state(cfg.train.resume_ckpt)
        # checkpoint = torch.load(cfg.train.resume_ckpt)
        # try:
        #     model.load_state_dict(prefix_with_module(checkpoint), strict=True)
        # except:
        #     model.load_state_dict(checkpoint, strict=True)

        accelerator.print(f"Successfully resumed from {cfg.train.resume_ckpt}")

    # metrics to record
    stats = VizStats(("loss", "lr", "sec/it", "Auc_30", "Racc_5", "Racc_15", "Racc_30", "Tacc_5", "Tacc_15", "Tacc_30"))
    num_epochs = cfg.train.epochs

    for epoch in range(start_epoch, num_epochs):
        stats.new_epoch()

        set_seed_and_print(cfg.seed + epoch)

        # Evaluation
        

        # Training
        accelerator.print(f"----------Start to train at epoch {epoch}----------")
        _train_or_eval_fn(
            model, dataloader, cfg, optimizer, stats, accelerator, lr_scheduler, training=True, visualize=False
        )
        accelerator.print(f"----------Finish the train at epoch {epoch}----------")

        if accelerator.is_main_process:
            lr = lr_scheduler.get_last_lr()[0]
            accelerator.print(f"----------LR is {lr}----------")
            accelerator.print(f"----------Saving stats to {cfg.exp_name}----------")
            stats.update({"lr": lr}, stat_set="train")
            stats.plot_stats(viz=viz, visdom_env=cfg.exp_name)
            accelerator.print(f"----------Done----------")

        if epoch % cfg.train.ckpt_interval == 0:
            accelerator.wait_for_everyone()
            ckpt_path = os.path.join(cfg.exp_dir, f"ckpt_{epoch:06}")
            accelerator.print(f"----------Saving the ckpt at epoch {epoch} to {ckpt_path}----------")
            accelerator.save_state(output_dir=ckpt_path, safe_serialization=False)

            if accelerator.is_main_process:
                stats.save(cfg.exp_dir + "stats")

    accelerator.wait_for_everyone()
    accelerator.save_state(output_dir=os.path.join(cfg.exp_dir, f"ckpt_{epoch:06}"), safe_serialization=False)

    return True


def _train_or_eval_fn(
    model, dataloader, cfg, optimizer, stats, accelerator, lr_scheduler, training=True, visualize=False
):
    if training:
        model.train()
    else:
        model.eval()

    time_start = time.time()
    max_it = len(dataloader)

    stat_set = "train" if training else "eval"

    for step, batch in enumerate(dataloader):  
        # data preparation
        query_image = batch['query_image']['image'].to(accelerator.device) # bx3xhxw
        query_T = batch['query_image']['T'].to(accelerator.device) # bx3
        query_R = batch['query_image']['R'].to(accelerator.device)# bx3x3

        ref_images = batch['ref_images']['image'].to(accelerator.device) # bxrx3xhxw
        ref_T = batch['ref_images']['T'].to(accelerator.device) # bxrx3
        ref_R = batch['ref_images']['R'].to(accelerator.device) # bxrx3x3

        query_pose = {
            'R': query_R,
            'T': query_T
        }
        ref_pose = {
            'R': ref_R,
            'T': ref_T
        }

        # for j in range(query_image.shape[0]):
        #     cv2.imwrite(f"/scratch/liudan/PoseDiffusion/new_images/{j}_query.jpg", np.array(query_image[j].cpu()))
        #     for i in range(ref_images.shape[1]):
        #         cv2.imwrite(f"/scratch/liudan/PoseDiffusion/new_images/{j}_ref_{i}.jpg", np.array(ref_images[j][i].cpu()))

        if training:
            predictions = model(query_image, ref_images, query_pose, ref_pose, training=True)
            predictions["loss"] = predictions["loss"].mean()
            loss = predictions["loss"]
        else:
            with torch.no_grad():
                predictions = model(query_image, ref_images, query_pose, ref_pose, training=False)

        pred_pose = predictions["pred_pose"] # object pose

        stats.update(predictions, time_start=time_start, stat_set=stat_set)
        if step % cfg.train.print_interval == 0:
            accelerator.print(stats.get_status_string(stat_set=stat_set, max_it=max_it))

        if training:
            optimizer.zero_grad()
            accelerator.backward(loss)
            if cfg.train.clip_grad > 0 and accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), cfg.train.clip_grad)
            optimizer.step()
            lr_scheduler.step()

    return True


def prefix_with_module(checkpoint):
    prefixed_checkpoint = OrderedDict()
    for key, value in checkpoint.items():
        prefixed_key = "module." + key
        prefixed_checkpoint[prefixed_key] = value
    return prefixed_checkpoint


if __name__ == "__main__":
    train_fn()
