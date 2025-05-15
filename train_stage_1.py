# -*- coding: utf-8 -*

import math
import os
import sys
import warnings
import torch
import torch.utils.checkpoint
from accelerate.logging import get_logger
from diffusers.utils import check_min_version

from models.champ_model_1d import ChampModel_TS
from models.unet_1d import UNet1DConditionModel
from models.mutual_self_attention import ReferenceAttentionControl

warnings.filterwarnings("ignore")
check_min_version("0.10.0.dev0")
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
logger = get_logger(__name__, log_level="INFO")


class Logger(object):
    def __init__(self, fileN="Default.log"):
        self.terminal = sys.stdout
        self.log = open(fileN, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


def model_instance(gradient_checkpointing, is_train, cfg, in_len, out_len, type='pretrain'):
    token_num_out = math.ceil(out_len // cfg.model.phase_dim)
    token_num_in = 32

    reference_unet = UNet1DConditionModel(
        in_channels=token_num_in,
        out_channels=token_num_in,
        block_out_channels=cfg.model.block_out_channels,
        cross_attention_dim=cfg.model.cross_attention_dim,
        layers_per_block=cfg.model.layers_per_block,
        use_motion_module=False,
        unet_use_temporal_attention=False,
        with_transfer_module=False,
    )
    denoising_unet = UNet1DConditionModel(
        in_channels=token_num_out,
        out_channels=token_num_out,
        block_out_channels=cfg.model.block_out_channels,
        cross_attention_dim=cfg.model.cross_attention_dim,
        layers_per_block=cfg.model.layers_per_block,
        use_motion_module=False,
        unet_use_temporal_attention=False,
        with_transfer_module=True,
    )

    if is_train:
        reference_unet.requires_grad_(True)
        denoising_unet.requires_grad_(True)
    else:
        reference_unet.requires_grad_(False)
        denoising_unet.requires_grad_(False)

    reference_control_writer = ReferenceAttentionControl(
        reference_unet,
        do_classifier_free_guidance=False,
        mode="write",
        fusion_blocks="full",
    )
    reference_control_reader = ReferenceAttentionControl(
        denoising_unet,
        do_classifier_free_guidance=False,
        mode="read",
        fusion_blocks="full",
    )

    model = ChampModel_TS(
        reference_unet,
        denoising_unet,
        reference_control_writer,
        reference_control_reader,
        in_len, out_len, cfg.model.phase_dim,
        type=type,
    )
    if is_train:
        model.requires_grad_(True)
    else:
        model.requires_grad_(False)

    for name, param in model.named_parameters():
        if param.requires_grad and (param.dtype == torch.float16):
            print(name)

    if is_train and gradient_checkpointing:
        reference_unet.enable_gradient_checkpointing()
        denoising_unet.enable_gradient_checkpointing()
    return model, reference_control_writer, reference_control_reader