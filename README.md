# TimeControl

Official code for "TimeControl: Diffusion-based Controllable Generalization for Cross-Domain Time Series Forecasting" submitted to NIPS2025.

You can execute the following commands to quickly run our code:

## Pretrain

accelerate launch --num_processes=1 train_stage_2.py --mode scratch --config configs/forecasting/Concat/336_96.yaml

accelerate launch --num_processes=1 train_stage_3.py --mode scratch --config configs/forecasting/Concat/336_96.yaml

accelerate launch --num_processes=1 train_stage_2.py --mode scratch --config configs/forecasting/Concat/336_192.yaml

accelerate launch --num_processes=1 train_stage_3.py --mode scratch --config configs/forecasting/Concat/336_192.yaml

accelerate launch --num_processes=1 train_stage_2.py --mode scratch --config configs/forecasting/Concat/336_336.yaml

accelerate launch --num_processes=1 train_stage_3.py --mode scratch --config configs/forecasting/Concat/336_336.yaml

accelerate launch --num_processes=1 train_stage_2.py --mode scratch --config configs/forecasting/Concat/336_720.yaml

accelerate launch --num_processes=1 train_stage_3.py --mode scratch --config configs/forecasting/Concat/336_720.yaml

## Scratch-ETTh1

accelerate launch --num_processes=1 train_stage_2.py --mode scratch --config configs/forecasting/ETTh1/336_96.yaml

accelerate launch --num_processes=1 train_stage_3.py --mode scratch --config configs/forecasting/ETTh1/336_96.yaml

accelerate launch --num_processes=1 train_stage_2.py --mode scratch --config configs/forecasting/ETTh1/336_192.yaml

accelerate launch --num_processes=1 train_stage_3.py --mode scratch --config configs/forecasting/ETTh1/336_192.yaml

accelerate launch --num_processes=1 train_stage_2.py --mode scratch --config configs/forecasting/ETTh1/336_336.yaml

accelerate launch --num_processes=1 train_stage_3.py --mode scratch --config configs/forecasting/ETTh1/336_336.yaml

accelerate launch --num_processes=1 train_stage_2.py --mode scratch --config configs/forecasting/ETTh1/336_720.yaml

accelerate launch --num_processes=1 train_stage_3.py --mode scratch --config configs/forecasting/ETTh1/336_720.yaml

python -W ignore inference_stage_3.py --mode scratch --config configs/forecasting/ETTh1/336_96.yaml

python -W ignore inference_stage_3.py --mode scratch --config configs/forecasting/ETTh1/336_192.yaml

python -W ignore inference_stage_3.py --mode scratch --config configs/forecasting/ETTh1/336_336.yaml

python -W ignore inference_stage_3.py --mode scratch --config configs/forecasting/ETTh1/336_720.yaml

python -W ignore inference_stage_4.py --mode scratch --config configs/forecasting/ETTh1/336_336.yaml

## ZeroShot-ETTh1

python -W ignore inference_stage_3.py --mode scratch --config configs/zero_shot/ETTh1/336_96.yaml

python -W ignore inference_stage_3.py --mode scratch --config configs/zero_shot/ETTh1/336_192.yaml

python -W ignore inference_stage_3.py --mode scratch --config configs/zero_shot/ETTh1/336_336.yaml

python -W ignore inference_stage_3.py --mode scratch --config configs/zero_shot/ETTh1/336_720.yaml

python -W ignore inference_stage_4.py --mode scratch --config configs/zero_shot/ETTh1/336_336.yaml

## Resample

python -W ignore inference_stage_3_resample.py --mode scratch --config configs/forecasting/ETTh1/336_96.yaml

python -W ignore inference_stage_3_resample.py --mode scratch --config configs/forecasting/ETTh2/336_96.yaml

python -W ignore inference_stage_3_resample.py --mode scratch --config configs/forecasting/ETTm1/336_96.yaml

python -W ignore inference_stage_3_resample.py --mode scratch --config configs/forecasting/ETTm2/336_96.yaml

python -W ignore inference_stage_3_resample.py --mode scratch --config configs/forecasting/Exchange/336_96.yaml

python -W ignore inference_stage_3_resample.py --mode scratch --config configs/forecasting/Weather/336_96.yaml

python -W ignore inference_stage_3_resample.py --mode scratch --config configs/forecasting/ECL/336_96.yaml

python -W ignore inference_stage_3_resample.py --mode scratch --config configs/forecasting/Traffic/336_96.yaml
