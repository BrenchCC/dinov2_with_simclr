# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import logging
import os
import random
import subprocess
import shutil
from urllib.parse import urlparse
import re

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

import yaml


logger = logging.getLogger("dinov2")


def rewrite_pretrained_ckpt_keys(state_dict):
    """
    meta released pretrained weights have different param keys with DinoVisionTransformer, change for loading.
    """
    new_chkpt = {}
    for k in state_dict.keys():
        new_k = ""
        if k.startswith("blocks."):
            match_str = re.match(r'blocks\.(\d+)\.', k)
            if match_str:
                block_id = match_str.group(1)
                if 0 <= int(block_id) <= 9:
                    new_k = k[:6] + ".0." + k[7:]
                elif 10 <= int(block_id) <= 19:
                    new_k = k[:6] + ".1." + k[7:]
                elif 20 <= int(block_id) <= 29:
                    new_k = k[:6] + ".2." + k[7:]
                elif 30 <= int(block_id) <= 39:
                    new_k = k[:6] + ".3." + k[7:]
        else:
            new_k = k
        new_chkpt[new_k] = state_dict[k]
    return new_chkpt


def load_pretrained_weights(model, pretrained_weights, checkpoint_key):
    if urlparse(pretrained_weights).scheme:  # If it looks like an URL
        state_dict = torch.hub.load_state_dict_from_url(pretrained_weights, map_location="cpu")
    else:
        state_dict = torch.load(pretrained_weights, map_location="cpu")
        # rewrite pretrained ckpt keys if loading meta released pretrained ckpt.
        if pretrained_weights.split("/")[-1].endswith("pretrain.pth"):
            state_dict = rewrite_pretrained_ckpt_keys(state_dict)
    if checkpoint_key is not None and checkpoint_key in state_dict:
        logger.info(f"Take key {checkpoint_key} in provided checkpoint dict")
        state_dict = state_dict[checkpoint_key]
    # remove `module.` prefix
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    # remove `backbone.` prefix induced by multicrop wrapper
    state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}

    # 对位置编码进行插值以匹配模型尺寸
    if "pos_embed" in state_dict and state_dict["pos_embed"].shape != model.pos_embed.shape:
        logger.info("Positional embedding shape mismatch. Attempting to interpolate.")
        pos_embed_checkpoint = state_dict["pos_embed"]
        
        # 从当前学生模型获取期望的 patch 数量
        num_patches = model.pos_embed.shape[1] - 1
        # 从预训练权重获取 patch 数量
        num_patches_pretrain = pos_embed_checkpoint.shape[1] - 1

        global_patch_size_pretrain = int(num_patches_pretrain**0.5)
        global_patch_size_curr = int(num_patches**0.5)
        assert global_patch_size_pretrain ** 2 == num_patches_pretrain, f"pos emb shape for pretrained model is not perfect square"
        assert global_patch_size_curr ** 2 == num_patches, f"pos emb shape for current model is not perfect square"

        logger.info(f"Interpolating positional embedding from {global_patch_size_pretrain}x{global_patch_size_pretrain} to {global_patch_size_curr}x{global_patch_size_curr}.")
        
        # 分离 CLS token 的位置编码
        pos_embed_cls = pos_embed_checkpoint[:, 0:1, :]
        
        # 获取 patch 的位置编码 (除去 CLS token)
        patch_pos_embed = pos_embed_checkpoint[:, 1:, :]
        
        # 重塑为 2D 网格以进行插值
        dim = pos_embed_checkpoint.shape[-1]
        h_pre = w_pre = int(num_patches_pretrain**0.5)
        patch_pos_embed = patch_pos_embed.reshape(1, h_pre, w_pre, dim).permute(0, 3, 1, 2)
        
        h_new = w_new = int(num_patches**0.5)
        
        # 执行双三次插值
        patch_pos_embed = F.interpolate(
            patch_pos_embed,
            size=(h_new, w_new),
            mode="bicubic",
            align_corners=False,
        )
        
        # 重塑并与 CLS token 的位置编码拼接
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).reshape(1, -1, dim)
        new_pos_embed = torch.cat((pos_embed_cls, patch_pos_embed), dim=1)
        
        # 更新 checkpoint 中的位置编码
        state_dict["pos_embed"] = new_pos_embed
    
    msg = model.load_state_dict(state_dict, strict=False)
    logger.info("Pretrained weights found at {} and loaded with msg: {}".format(pretrained_weights, msg))


def fix_random_seeds(seed=31):
    """
    Fix random seeds.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def get_sha():
    cwd = os.path.dirname(os.path.abspath(__file__))

    def _run(command):
        return subprocess.check_output(command, cwd=cwd).decode("ascii").strip()

    sha = "N/A"
    diff = "clean"
    branch = "N/A"
    try:
        sha = _run(["git", "rev-parse", "HEAD"])
        subprocess.check_output(["git", "diff"], cwd=cwd)
        diff = _run(["git", "diff-index", "HEAD"])
        diff = "has uncommitted changes" if diff else "clean"
        branch = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    except Exception:
        pass
    message = f"sha: {sha}, status: {diff}, branch: {branch}"
    return message


class CosineScheduler(object):
    def __init__(self, base_value, final_value, total_iters, warmup_iters=0, start_warmup_value=0, freeze_iters=0):
        super().__init__()
        self.final_value = final_value
        self.total_iters = total_iters

        freeze_schedule = np.zeros((freeze_iters))

        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

        iters = np.arange(total_iters - warmup_iters - freeze_iters)
        schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))
        self.schedule = np.concatenate((freeze_schedule, warmup_schedule, schedule))

        assert len(self.schedule) == self.total_iters

    def __getitem__(self, it):
        if it >= self.total_iters:
            return self.final_value
        else:
            return self.schedule[it]


def has_batchnorms(model):
    bn_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)
    for name, module in model.named_modules():
        if isinstance(module, bn_types):
            return True
    return False


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


def save_config_file(model_checkpoints_folder, args):
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
        with open(os.path.join(model_checkpoints_folder, 'config.yml'), 'w') as outfile:
            yaml.dump(args, outfile, default_flow_style=False)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
