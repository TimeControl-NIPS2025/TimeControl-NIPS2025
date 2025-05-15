# -*- coding: utf-8 -*

import argparse
import logging
import os
import sys
import numpy as np
import torch
import torch.utils.checkpoint
from diffusers import DDIMScheduler
from omegaconf import OmegaConf

from data_1_forecast import getTimeSeriesDataset
from train_stage_1 import model_instance
from metrics_utils.loss_function import metric


class Logger(object):
    def __init__(self, fileN="Default.log"):
        self.terminal = sys.stdout
        self.log = open(fileN, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


def main(cfg, n_resample):
    current_bs = cfg.data.batch_size * 8
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logging.info(f"Running inference ...")
    sched_kwargs = OmegaConf.to_container(cfg.noise_scheduler_kwargs)
    if cfg.enable_zero_snr:
        sched_kwargs.update(
            rescale_betas_zero_snr=True,
            timestep_spacing="trailing",
            prediction_type="v_prediction",
        )
    scheduler = DDIMScheduler(**sched_kwargs)
    dataset = getTimeSeriesDataset(
        data=cfg.data.data,
        input_len=cfg.data.in_len,
        pred_len=cfg.data.out_len,
        down_scale=cfg.data.down_scale,
        step=cfg.data.step,
        flag='test'
    )
    current_bs = current_bs if len(dataset) > current_bs else len(dataset)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=current_bs,
        shuffle=False,
        num_workers=cfg.data.num_works,
        drop_last=True,
    )
    model, _, _ = model_instance(
        gradient_checkpointing=cfg.solver.gradient_checkpointing, is_train=False, cfg=cfg,
        in_len=cfg.data.in_len, out_len=cfg.data.out_len, type='forecasting'
    )
    pth_name = f"{cfg.output_dir}/{cfg.exp_stage3_name}_{args.mode}/saved_models/Champ_RNA-{cfg.bset_batch_idx_stage3}.pth"
    pth = torch.load(pth_name)
    model.load_state_dict(pth, strict=True)
    model_show(model)
    generator = torch.Generator(device="cuda")
    generator.manual_seed(cfg.seed)
    for n_iter in range(n_resample):
        criterion_mse = torch.nn.MSELoss()
        criterion_mae = torch.nn.L1Loss()
        look_image_ = []
        pred_image_ = []
        true_image_ = []
        model.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
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
                    scheduler=scheduler,
                )
                true_image = forecast
                loss_image_mse = criterion_mse(pred_image, true_image)
                loss_image_mae = criterion_mae(pred_image, true_image)
                look_image_.append(lookback.detach().cpu())
                pred_image_.append(pred_image.detach().cpu())
                true_image_.append(true_image.detach().cpu())
                print('Step:{}; TrainMSELoss:{}; TrainMAELoss:{};'.format(batch_idx, loss_image_mse, loss_image_mae))

        look_image_ = np.concatenate(look_image_, axis=0)
        pred_image_ = np.concatenate(pred_image_, axis=0)
        true_image_ = np.concatenate(true_image_, axis=0)
        np.save(file=f"{save_dir}/validation/valid_look_{n_iter}.npy", arr=look_image_)
        np.save(file=f"{save_dir}/validation/valid_pred_{n_iter}.npy", arr=pred_image_)
        np.save(file=f"{save_dir}/validation/valid_true_{n_iter}.npy", arr=true_image_)
        mae, mse, rmse, mape, mspe = metric(pred_image_, true_image_)
        print('LatentSpace: mse:{}, mae:{}, rmse:{}, mape:{}, mspe:{}'.format(mse, mae, rmse, mape, mspe))
        f = open(f"{cfg.output_dir}/inference_forcasting.txt", 'a')
        f.write(f'Inference_Stage_3_{n_iter}: {args.mode} in {args.config}' + '\n')
        f.write(f'mse:{mse}, mae:{mae}' + '\n')
        f.write('\scalebox{0.78}{' + str(round(mse, 3)) + '} &\scalebox{0.78}{' + str(round(mae, 3)) + '} &' + '\n' + '\n')
        f.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="scratch", help="scratch or finetune")
    parser.add_argument("--config", type=str, default="./configs/inference.yaml")
    args = parser.parse_args()

    n_resample = 50
    if args.config[-5:] == ".yaml":
        cfg = OmegaConf.load(args.config)
    else:
        raise ValueError("Do not support this format config file")
    if 'zero_shot' in args.config:
        project_name = f'{cfg.data.data}_{cfg.data.in_len}_{cfg.data.out_len}_zeroshot'
    else:
        project_name = f'{cfg.exp_stage3_name}_{args.mode}'
    save_dir = os.path.join(cfg.output_dir, project_name, f'inference_forecasting_resample{n_resample}')
    os.makedirs(save_dir, exist_ok=True)
    if os.path.exists(f'./{save_dir}/inference_stage_3.txt'):
        os.remove(f'./{save_dir}/inference_stage_3.txt')
    sys.stdout = Logger(f'./{save_dir}/inference_stage_3.txt')
    os.makedirs(os.path.join(save_dir, 'validation'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'show'), exist_ok=True)
    main(cfg, n_resample)


