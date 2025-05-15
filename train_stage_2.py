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

torch.cuda.current_device()
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
            flag='train',
        )
    else:
        from data_1_forecast import getTimeSeriesDataset
        dataset = getTimeSeriesDataset(
            data=cfg.data.data,
            input_len=cfg.data.in_len,
            pred_len=cfg.data.out_len,
            down_scale=cfg.data.down_scale,
            step=cfg.data.step,
            flag='train',
        )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.data.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_works,
        drop_last=True,
    )
    num_update_steps_per_epoch = len(dataset) // cfg.data.batch_size
    num_train_epochs = math.ceil(cfg.max_train_steps_stage2 / num_update_steps_per_epoch)
    if 'scratch' in project_name:
        model, reference_control_writer, reference_control_reader = model_instance(
            gradient_checkpointing=cfg.solver.gradient_checkpointing, is_train=True, cfg=cfg,
            in_len=cfg.data.in_len, out_len=cfg.data.out_len,
        )
    elif 'finetune' in project_name:
        model, reference_control_writer, reference_control_reader = model_instance(
            gradient_checkpointing=cfg.solver.gradient_checkpointing, is_train=False, cfg=cfg,
            in_len=cfg.data.in_len, out_len=cfg.data.out_len,
        )
        pth_name = f"{cfg.output_dir}/{cfg.exp_stage1_name}/saved_models/Champ_RNA-{cfg.bset_batch_idx_stage1}.pth"
        pth = torch.load(pth_name)
        for name in list(pth.keys()):
            if "transfer_module" in name:
                pth.pop(name)
        model.load_state_dict(pth, strict=False)
        for name, module in model.named_modules():
            if "transfer_module" in name:
                for params in module.parameters():
                    params.requires_grad = True
    else:
        raise ValueError(f'unknown projection information: {project_name}')

    if cfg.solver.scale_lr:
        learning_rate = (
            cfg.solver.learning_rate
            * cfg.solver.gradient_accumulation_steps
            * cfg.data.batch_size
            * accelerator.num_processes
        )
    else:
        learning_rate = cfg.solver.learning_rate

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
        num_training_steps=cfg.max_train_steps_stage2
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

    # 5.2、打印训练信息
    logger.info("Start training ...")
    logger.info(f"Dataset_len: {len(dataset)}")                     # 8209
    logger.info(f"Batch_size: {cfg.data.batch_size}")               # 32
    logger.info(f"Batch_num: {num_update_steps_per_epoch}")         # 256
    logger.info(f"Total_batch: {cfg.max_train_steps_stage2}")       # 10000
    logger.info(f"Start_batch: {global_step}")                      # 0
    logger.info(f"Total_epoch: {num_train_epochs}")                 # 40
    logger.info(f"Start_epoch: {first_epoch}")                      # 0
    model, optimizer, dataloader, lr_scheduler = accelerator.prepare(model, optimizer, dataloader, lr_scheduler)

    progress_bar = tqdm(range(global_step, cfg.max_train_steps_stage2), disable=not accelerator.is_local_main_process,)
    progress_bar.set_description("Steps")

    save_dir = f"{cfg.output_dir}/{project_name}"
    criterion_mse = torch.nn.MSELoss()
    all_train_loss_ = []
    all_valid_loss_ = []
    for epoch in range(first_epoch, num_train_epochs):
        train_loss = 0.
        for batch_idx, batch in enumerate(dataloader):
            with accelerator.accumulate(model):
                device = torch.device("cuda:0")
                model = model.to(device)
                lookback, forecast = batch
                lookback, forecast = lookback.to(dtype=torch.float32).to(device), forecast.to(dtype=torch.float32).to(device)
                lookback, forecast = lookback.permute(0, 2, 1), forecast.permute(0, 2, 1)
                timesteps = torch.randint(
                    0,
                    train_noise_scheduler.num_train_timesteps,
                    (forecast.shape[0]*forecast.shape[1],),
                    device=lookback.device,
                )
                timesteps = timesteps.long()
                loss_snr, pred_latent, true_latent = model(
                    lookback=lookback,
                    forecast=forecast,
                    train_noise_scheduler=train_noise_scheduler,
                    noise_offset=cfg.noise_offset,
                    snr_gamma=cfg.snr_gamma,
                    timesteps=timesteps,
                )
                loss_latent = criterion_mse(pred_latent, true_latent)
                loss_latent = loss_latent.repeat(cfg.data.batch_size)
                avg_loss = accelerator.gather(loss_latent).mean()
                train_loss += avg_loss.item() / cfg.solver.gradient_accumulation_steps
                all_train_loss_.append(train_loss)
                accelerator.backward(loss_snr)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainable_params, cfg.solver.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            print('Step:{}; TrainLoss:{};'.format(global_step, train_loss))

            if accelerator.sync_gradients:
                reference_control_reader.clear()
                reference_control_writer.clear()
                progress_bar.update(1)
                global_step += 1
                tb_tracker.add_scalar(tag='pretrain loss', scalar_value=train_loss, global_step=global_step)
                if accelerator.is_main_process:
                    if global_step % cfg.savemodel_steps_stage2 == 0 or global_step == 1:
                        unwrap_model = accelerator.unwrap_model(model)
                        save_path = f"{save_dir}/saved_models/Champ_RNA-{global_step}.pth"
                        state_dict = unwrap_model.state_dict()
                        torch.save(state_dict, save_path)

                train_loss = 0.

            logs = {
                "step_loss": loss_snr.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
                "stage": 2,
            }
            progress_bar.set_postfix(**logs)
            if global_step >= cfg.max_train_steps_stage2:
                break

    accelerator.wait_for_everyone()
    accelerator.end_training()
    np.save(file=f"{save_dir}/validation/all_train_loss.npy", arr=np.array(all_train_loss_))
    np.save(file=f"{save_dir}/validation/all_valid_loss.npy", arr=np.array(all_valid_loss_))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="scratch", help="scratch or finetune")
    parser.add_argument("--config", type=str, default="./configs/pretrain/stage.yaml")
    args = parser.parse_args()
    if args.config[-5:] == ".yaml":
        cfg = OmegaConf.load(args.config)
    else:
        raise ValueError("Do not support this format config file")
    project_name = f'{cfg.exp_stage2_name}_{args.mode}'
    save_dir = os.path.join(cfg.output_dir, project_name)
    os.makedirs(save_dir, exist_ok=True)
    if os.path.exists(f'./{save_dir}/train_stage_2.txt'):
        os.remove(f'./{save_dir}/train_stage_2.txt')
    sys.stdout = Logger(f'./{save_dir}/train_stage_2.txt')
    os.makedirs(os.path.join(save_dir, 'sanity_check'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'saved_models'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'validation'), exist_ok=True)
    shutil.copy(args.config, os.path.join(save_dir, 'sanity_check', f'{cfg.exp_stage2_name}.yaml'))
    shutil.copy(os.path.abspath(__file__), os.path.join(save_dir, 'sanity_check'))
    main(cfg)
    