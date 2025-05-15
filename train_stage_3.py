# -*- coding: utf-8 -*

import argparse
import logging
import math
import os
import sys
import warnings
import shutil
import diffusers
import numpy as np
import torch
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs
from diffusers import DDIMScheduler
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from omegaconf import OmegaConf
from tqdm.auto import tqdm

from train_stage_1 import model_instance

from utils.tb_tracker import TbTracker
from utils.util import seed_everything

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


def main(cfg):
    current_bs = cfg.data.batch_size * 8
    current_lr = cfg.solver.learning_rate * 10
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    tb_tracker = TbTracker(project_name, cfg.output_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.solver.gradient_accumulation_steps,
        mixed_precision=cfg.solver.mixed_precision,
        log_with=tb_tracker,
        project_dir=f'{cfg.output_dir}/{project_name}',
        kwargs_handlers=[kwargs],
    )
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=True)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()
    if cfg.seed is not None:
        seed_everything(cfg.seed)
    sched_kwargs = OmegaConf.to_container(cfg.noise_scheduler_kwargs)
    if cfg.enable_zero_snr:
        sched_kwargs.update(
            rescale_betas_zero_snr=True,
            timestep_spacing="trailing",
            prediction_type="v_prediction",
        )
    sched_kwargs.update({"beta_schedule": "scaled_linear"})
    train_noise_scheduler = DDIMScheduler(**sched_kwargs)
    if cfg.data.data == 'Concat':
        from data_3_forecast_concat import getConcatDatsset
        dataset = getConcatDatsset(
            input_len=cfg.data.in_len,
            pred_len=cfg.data.out_len,
            flag='test',
        )
    else:
        from data_1_forecast import getTimeSeriesDataset
        dataset = getTimeSeriesDataset(
            data=cfg.data.data,
            input_len=cfg.data.in_len,
            pred_len=cfg.data.out_len,
            down_scale=cfg.data.down_scale,
            step=cfg.data.step,
            flag='test'
        )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=current_bs,
        shuffle=True,
        num_workers=cfg.data.num_works,
        drop_last=False,
    )

    num_update_steps_per_epoch = len(dataset) // current_bs
    num_update_steps_per_epoch = num_update_steps_per_epoch if num_update_steps_per_epoch != 0 else 1
    num_train_epochs = math.ceil(cfg.max_train_steps_stage3 / num_update_steps_per_epoch)

    model, reference_control_writer, reference_control_reader = model_instance(
        gradient_checkpointing=cfg.solver.gradient_checkpointing, is_train=False, cfg=cfg,
        in_len=cfg.data.in_len, out_len=cfg.data.out_len, type='forecasting'
    )
    pth_name = f"{cfg.output_dir}/{cfg.exp_stage2_name}_{args.mode}/saved_models/Champ_RNA-{cfg.bset_batch_idx_stage2}.pth"
    pth = torch.load(pth_name)
    model.load_state_dict(pth, strict=False)
    for name, module in model.named_modules():
        if "forecasting" in name:
            for params in module.parameters():
                params.requires_grad = True
    generator = torch.Generator(device="cuda")
    generator.manual_seed(cfg.seed)

    if cfg.solver.scale_lr:
        learning_rate = (
            current_lr
            * cfg.solver.gradient_accumulation_steps
            * current_bs
            * accelerator.num_processes
        )
    else:
        learning_rate = current_lr

    if cfg.solver.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError("Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`")
        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    trainable_params = list(filter(lambda p: p.requires_grad, model.parameters()))
    logger.info(f"Total trainable params {len(trainable_params)}")

    optimizer = optimizer_cls(
        trainable_params,
        lr=learning_rate,
        betas=(cfg.solver.adam_beta1, cfg.solver.adam_beta2),
        weight_decay=cfg.solver.adam_weight_decay,
        eps=cfg.solver.adam_epsilon,
    )

    lr_scheduler = get_scheduler(
        cfg.solver.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=cfg.solver.lr_warmup_steps
        * cfg.solver.gradient_accumulation_steps,
        num_training_steps=cfg.max_train_steps_stage3
        * cfg.solver.gradient_accumulation_steps,
    )

    if cfg.resume_from_checkpoint:
        if cfg.resume_from_checkpoint != "latest":
            resume_dir = cfg.resume_from_checkpoint
        else:
            resume_dir = f"{cfg.output_dir}/{project_name}/saved_models"
        dirs = os.listdir(resume_dir)
        dirs = sorted(dirs, key=lambda x: int(x.split("-")[1].split(".")[0]))
        path = dirs[-1]
        model.load_state_dict(torch.load(os.path.join(resume_dir, path)))
        global_step = int(path.split("-")[1].split(".")[0])
        first_epoch = global_step // num_update_steps_per_epoch
    else:
        global_step, first_epoch = 0, 0

    logger.info("Start training ...")
    logger.info(f"Dataset_len: {len(dataset)}")                     # 8209
    logger.info(f"Batch_size: {current_bs}")                        # 32
    logger.info(f"Batch_num: {num_update_steps_per_epoch}")         # 256
    logger.info(f"Total_batch: {cfg.max_train_steps_stage3}")       # 10000
    logger.info(f"Start_batch: {global_step}")                      # 0
    logger.info(f"Total_epoch: {num_train_epochs}")                 # 40
    logger.info(f"Start_epoch: {first_epoch}")                      # 0
    model, optimizer, dataloader, lr_scheduler = accelerator.prepare(model, optimizer, dataloader, lr_scheduler)

    progress_bar = tqdm(range(global_step, cfg.max_train_steps_stage3), disable=not accelerator.is_local_main_process,)
    progress_bar.set_description("Steps")

    criterion_mse = torch.nn.MSELoss()
    criterion_mae = torch.nn.L1Loss()
    all_train_loss_ = []
    for epoch in range(first_epoch, num_train_epochs):
        train_loss = 0.
        for _, batch in enumerate(dataloader):
            with accelerator.accumulate(model):
                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                model = model.to(device)
                lookback, forecast = batch
                lookback, forecast = lookback.to(dtype=torch.float32).to(device), forecast.to(dtype=torch.float32).to(device)
                lookback, forecast = lookback.permute(0, 2, 1), forecast.permute(0, 2, 1)
                model.updateReferenceAttentionControl(
                    guidance_scale=cfg.model.guidance_scale,
                    batch_size=lookback.shape[0]
                )
                pred_image = model.forecasting(
                    lookback=lookback,
                    num_inference_steps=cfg.model.num_inference_steps,
                    guidance_scale=cfg.model.guidance_scale,
                    generator=generator,
                    scheduler=train_noise_scheduler,
                )
                true_image = forecast
                loss_mse = criterion_mse(pred_image, true_image)
                loss_mae = criterion_mae(pred_image, true_image)
                avg_loss = loss_mse.repeat(current_bs)
                avg_loss = accelerator.gather(avg_loss).mean()
                train_loss += avg_loss.item() / cfg.solver.gradient_accumulation_steps
                all_train_loss_.append(train_loss)
                accelerator.backward(loss_mse)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainable_params, cfg.solver.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            print('Step:{}; TrainMSELoss:{}; TrainMAELoss:{};'.format(global_step, loss_mse, loss_mae))

            if accelerator.sync_gradients:
                reference_control_reader.clear()
                reference_control_writer.clear()
                progress_bar.update(1)
                global_step += 1
                tb_tracker.add_scalar(tag='pretrain loss', scalar_value=train_loss, global_step=global_step)

                if accelerator.is_main_process:
                    if global_step % cfg.savemodel_steps_stage3 == 0 or global_step == 1:
                        unwrap_model = accelerator.unwrap_model(model)
                        save_path = f"{save_dir}/saved_models/Champ_RNA-{global_step}.pth"
                        state_dict = unwrap_model.state_dict()
                        torch.save(state_dict, save_path)

                train_loss = 0.

            logs = {
                "step_loss": loss_mse.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
                "stage": 3,
            }
            progress_bar.set_postfix(**logs)
            if global_step >= cfg.max_train_steps_stage3:
                break

    accelerator.wait_for_everyone()
    accelerator.end_training()
    np.save(file=f"{save_dir}/validation/all_train_loss.npy", arr=np.array(all_train_loss_))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="scratch", help="scratch or finetune")
    parser.add_argument("--config", type=str, default="./configs/pretrain/stage.yaml")
    args = parser.parse_args()
    if args.config[-5:] == ".yaml":
        cfg = OmegaConf.load(args.config)
    else:
        raise ValueError("Do not support this format config file")
    project_name = f'{cfg.exp_stage3_name}_{args.mode}'
    save_dir = os.path.join(cfg.output_dir, project_name)
    os.makedirs(save_dir, exist_ok=True)
    if os.path.exists(f'./{save_dir}/train_stage_3.txt'):
        os.remove(f'./{save_dir}/train_stage_3.txt')
    sys.stdout = Logger(f'./{save_dir}/train_stage_3.txt')
    os.makedirs(os.path.join(save_dir, 'sanity_check'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'saved_models'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'validation'), exist_ok=True)
    shutil.copy(args.config, os.path.join(save_dir, 'sanity_check', f'{cfg.exp_stage3_name}.yaml'))
    shutil.copy(os.path.abspath(__file__), os.path.join(save_dir, 'sanity_check'))
    main(cfg)
    