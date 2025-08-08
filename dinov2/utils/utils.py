# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import logging
import os
import random
import subprocess
from urllib.parse import urlparse
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

logger = logging.getLogger("dinov2")


def load_pretrained_weights(model, pretrained_weights, checkpoint_key):
    if urlparse(pretrained_weights).scheme:  # If it looks like an URL
        state_dict = torch.hub.load_state_dict_from_url(pretrained_weights, map_location="cpu")
    else:
        state_dict = torch.load(pretrained_weights, map_location="cpu")
    if checkpoint_key is not None and checkpoint_key in state_dict:
        logger.info(f"Take key {checkpoint_key} in provided checkpoint dict")
        state_dict = state_dict[checkpoint_key]
    # remove `module.` prefix
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    # remove `backbone.` prefix induced by multicrop wrapper
    state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}

    # Interpolate positional encodings to match model size
    if "pos_embed" in state_dict and state_dict["pos_embed"].shape != model.pos_embed.shape:
        logger.info("Positional embedding shape mismatch. Attempting to interpolate.")
        pos_embed_checkpoint = state_dict["pos_embed"]

        # Get the expected number of patches from the current student model
        num_patches = model.pos_embed.shape[1] - 1
        # Get the number of patches from the pretrain model
        num_patches_pretrain = pos_embed_checkpoint.shape[1] - 1
        # from 1369(37*37) patches interpolate to 256(16*16) patches
        global_patch_size_pretrain = int(num_patches_pretrain ** 0.5)
        global_patch_size_curr = int(num_patches ** 0.5)
        assert global_patch_size_pretrain ** 2 == num_patches_pretrain, f"pos emb shape for pretrained model is not perfect square"
        assert global_patch_size_curr ** 2 == num_patches, f"pos emb shape for current model is not perfect square"
        logger.info(
            f"Interpolating positional embedding from {global_patch_size_pretrain}x{global_patch_size_pretrain} to {global_patch_size_curr}x{global_patch_size_curr}.")

        # Separate the positional embedding of the cls token
        pos_embed_cls = pos_embed_checkpoint[:, 0:1, :]

        # Get the patch positional embedding(expect cls)
        patch_pos_embed = pos_embed_checkpoint[:, 1:, :]

        # Reshape into 2d network for interpolation
        dim = pos_embed_checkpoint.shape[-1]
        h_pre = w_pre = int(num_patches_pretrain ** 0.5)
        patch_pos_embed = patch_pos_embed.reshape(1, h_pre, w_pre, dim).permute(0, 3, 1, 2)

        h_new = w_new = int(num_patches ** 0.5)

        # Perform bicubic interpolation
        patch_pos_embed = F.interpolate(
            patch_pos_embed,
            size=(h_new, w_new),
            mode="bicubic",
            align_corners=False,
        )

        # Reshape and concatenate with the positional encoding of the cls token
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).reshape(1, -1, dim)
        new_pos_embed = torch.cat((pos_embed_cls, patch_pos_embed), dim=1)

        # Update the positional embedding in checkpoint
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
