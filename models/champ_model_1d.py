import inspect
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm
from utils.util import compute_snr
from diffusers.utils.torch_utils import randn_tensor

from models.unet_1d import UNet1DConditionModel
from models.resnet_1d import InflatedConv1d


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model, kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        d_model_raw = d_model
        if (d_model_raw % 2) != 0:
            d_model = d_model + 1
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        if (d_model_raw % 2) != 0:
            pe = pe[:, :, :d_model_raw]
        self.register_buffer('pe', pe)

    def forward(self, x):
        output = self.pe[:, :x.shape[1]]
        return output


class PatchEmbedding(nn.Module):
    def __init__(self, d_model, patch_len, stride, dropout=0.1):
        super(PatchEmbedding, self).__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch_layer = nn.ReplicationPad1d((0, stride))
        self.value_embedding = TokenEmbedding(patch_len, d_model)
        self.position_embedding = PositionalEmbedding(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        n_channel = x.shape[1]
        x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        ve = self.value_embedding(x)
        pe = self.position_embedding(x)
        x_embed = ve + pe
        x_embed = self.dropout(x_embed)
        return x_embed, n_channel


class PhaseInduced(nn.Module):
    def __init__(self, phase_dim):
        super(PhaseInduced, self).__init__()
        self.phase_dim = phase_dim

    def forward(self, x):

        if x.shape[2] % self.phase_dim != 0:
            seq_len_padding = ((x.shape[2] // self.phase_dim) + 1) * self.phase_dim
            padding = torch.zeros([x.shape[0], x.shape[1], (seq_len_padding - x.shape[2])]).to(x.device)
            x = torch.cat([x, padding], dim=2)
        else:
            seq_len_padding = x.shape[2]

        n_channel = x.shape[1]
        x = x.reshape(x.shape[0], x.shape[1], seq_len_padding // self.phase_dim, self.phase_dim)
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        return x, n_channel


class PhaseInducedEmbedding(nn.Module):
    def __init__(self, phase_dim, in_channel, out_channel, dropout=0.1):
        super(PhaseInducedEmbedding, self).__init__()
        self.phase_dim = phase_dim
        self.value_embedding = TokenEmbedding(phase_dim, phase_dim)
        self.channel_embedding = InflatedConv1d(in_channel, out_channel, kernel_size=3, padding=1)
        self.position_embedding = PositionalEmbedding(phase_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        if x.shape[2] % self.phase_dim != 0:
            seq_len_padding = ((x.shape[2] // self.phase_dim) + 1) * self.phase_dim
            padding = torch.zeros([x.shape[0], x.shape[1], (seq_len_padding - x.shape[2])]).to(x.device)
            x = torch.cat([x, padding], dim=2)
        else:
            seq_len_padding = x.shape[2]

        n_channel = x.shape[1]
        x = x.reshape(x.shape[0], x.shape[1], seq_len_padding // self.phase_dim, self.phase_dim)
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))

        ve = self.value_embedding(x)
        ve = self.channel_embedding(ve)
        pe = self.position_embedding(ve)
        x_embed = ve + pe
        x_embed = self.dropout(x_embed)
        return x_embed, n_channel


class MLP(nn.Module):
    def __init__(self, in_size, hid_size, out_size, dropout_rate=0.1):
        super(MLP, self).__init__()

        self.layer1 = nn.Linear(in_size, hid_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.layer2 = nn.Linear(hid_size, out_size)

        self.initialize_weight(self.layer1)
        self.initialize_weight(self.layer2)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.layer2(x)
        return x

    def initialize_weight(self, x):
        nn.init.xavier_uniform_(x.weight)
        if x.bias is not None:
            nn.init.constant_(x.bias, 0)


class Flatten_Head(nn.Module):
    def __init__(self, input_seq_len, output_seq_len, dropout=0.1):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(input_seq_len, output_seq_len)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class OutputLayer(nn.Module):
    def __init__(self, token_dim, token_num_out, dropout=0.1):
        super().__init__()
        self.token_dim = token_dim
        self.token_num_out = token_num_out
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(token_dim*token_num_out, token_dim*token_num_out)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear(x)
        x = torch.reshape(x, (x.shape[0], self.token_num_out, self.token_dim))
        return x


class moving_avg(nn.Module):
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp_multi(nn.Module):
    def __init__(self, kernel_size=(13, 17)):
        super(series_decomp_multi, self).__init__()
        self.kernel_size = kernel_size
        self.moving_avg = [moving_avg(kernel, stride=1) for kernel in kernel_size]

    def forward(self, x):
        moving_mean = []
        res = []
        for func in self.moving_avg:
            moving_avg = func(x)
            moving_mean.append(moving_avg)
            sea = x - moving_avg
            res.append(sea)

        sea = sum(res) / len(res)
        moving_mean = sum(moving_mean) / len(moving_mean)
        return sea, moving_mean


class ChampModel_TS(nn.Module):
    def __init__(
        self,
        reference_unet: UNet1DConditionModel,
        denoising_unet: UNet1DConditionModel,
        reference_control_writer,
        reference_control_reader,
        in_len, out_len, phase_dim,
        type='pretrain'
    ):
        super().__init__()
        self.dtype = torch.float32
        self.token_dim = phase_dim
        self.token_num_in = math.ceil(in_len // phase_dim)
        self.token_num_hid = 32
        self.token_num_out = math.ceil(out_len // phase_dim)
        self.decomp_multi = series_decomp_multi()
        self.prompt_embedding = MLP(in_size=in_len, hid_size=in_len*2, out_size=768)
        self.pi = PhaseInduced(phase_dim=phase_dim)
        self.pi_embedding = PhaseInducedEmbedding(phase_dim=phase_dim, in_channel=self.token_num_in, out_channel=self.token_num_hid)
        self.reference_unet = reference_unet
        self.denoising_unet = denoising_unet
        self.reference_control_writer = reference_control_writer
        self.reference_control_reader = reference_control_reader

        if type in ['forecasting', 'sampling']:
            self.forecasting_flatten = nn.Flatten(start_dim=-2)
            self.forecasting_head = MLP(in_size=out_len, hid_size=4*out_len, out_size=out_len)

    def forward(self, lookback, forecast, train_noise_scheduler, noise_offset, snr_gamma, timesteps):
        lookback_embed, lookback_channel = self.pi_embedding(lookback)
        forecast_embed, forecast_channel = self.pi(forecast)
        assert lookback_channel == forecast_channel, "The channel of lookback and forecast should be same!"
        seasonal_part, trend_part = self.decomp_multi(lookback.permute(0, 2, 1))
        trend_part = trend_part.permute(0, 2, 1)
        series_prompt_embed = torch.reshape(trend_part, (lookback.shape[0] * lookback.shape[1], lookback.shape[2]))
        series_prompt_embed = series_prompt_embed.unsqueeze(1)
        series_prompt_embed = self.prompt_embedding(series_prompt_embed)

        noise = torch.randn_like(forecast_embed)

        if noise_offset > 0.0:
            tmp = torch.randn((noise.shape[0], noise.shape[1], 1), device=noise.device)
            noise += noise_offset * tmp

        noisy_latents = train_noise_scheduler.add_noise(forecast_embed, noise, timesteps)
        if train_noise_scheduler.prediction_type == "epsilon":
            target_latents = noise
        elif train_noise_scheduler.prediction_type == "v_prediction":
            target_latents = train_noise_scheduler.get_velocity(forecast_embed, noise, timesteps)
        else:
            raise ValueError(f"Unknown prediction type {train_noise_scheduler.prediction_type}")

        ref_timesteps = torch.zeros_like(timesteps)
        self.reference_unet(
            lookback_embed,
            ref_timesteps,
            encoder_hidden_states=series_prompt_embed,
        )

        self.reference_control_reader.update(self.reference_control_writer)

        model_pred = self.denoising_unet(
            noisy_latents,
            timesteps,
            encoder_hidden_states=series_prompt_embed,
        )

        if snr_gamma == 0:
            loss_diff = F.mse_loss(model_pred.float(), target_latents.float(), reduction="mean")
        else:
            snr = compute_snr(train_noise_scheduler, timesteps)
            if train_noise_scheduler.config.prediction_type == "v_prediction":
                snr = snr + 1
            tmp = torch.ones_like(timesteps)
            mse_loss_weights = torch.stack([snr, snr_gamma * tmp], dim=1).min(dim=1)[0] / snr
            loss_diff = F.mse_loss(model_pred.float(), target_latents.float(), reduction="none")
            loss_diff = loss_diff.mean(dim=list(range(1, len(loss_diff.shape)))) * mse_loss_weights
            loss_diff = loss_diff.mean()
        return loss_diff, model_pred.detach(), target_latents.detach()

    def updateReferenceAttentionControl(self, guidance_scale, batch_size):
        do_classifier_free_guidance = guidance_scale > 1.0
        if do_classifier_free_guidance:
            from models.mutual_self_attention import ReferenceAttentionControl
            self.reference_control_writer = ReferenceAttentionControl(
                unet=self.reference_unet,
                mode="write",
                do_classifier_free_guidance=do_classifier_free_guidance,
                batch_size=batch_size,
                fusion_blocks="full",
            )
            self.reference_control_reader = ReferenceAttentionControl(
                unet=self.denoising_unet,
                mode="read",
                do_classifier_free_guidance=do_classifier_free_guidance,
                batch_size=batch_size,
                fusion_blocks="full",
            )

    def prepare_extra_step_kwargs(self, generator, eta, scheduler):
        accepts_eta = "eta" in set(inspect.signature(scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta
        accepts_generator = "generator" in set(inspect.signature(scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def progress_bar(self, iterable=None, total=None):
        if not hasattr(self, "_progress_bar_config"):
            self._progress_bar_config = {}
        elif not isinstance(self._progress_bar_config, dict):
            raise ValueError(
                f"`self._progress_bar_config` should be of type `dict`, but is {type(self._progress_bar_config)}."
            )

        if iterable is not None:
            return tqdm(iterable, **self._progress_bar_config)
        elif total is not None:
            return tqdm(total=total, **self._progress_bar_config)
        else:
            raise ValueError("Either `total` or `iterable` has to be defined.")

    def forecasting(self, lookback, num_inference_steps, guidance_scale, generator, scheduler, eta=0.0, callback=None, callback_steps=1):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        lookback = lookback.to(device)
        scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = scheduler.timesteps
        lookback_embed, lookback_channel = self.pi_embedding(lookback)
        series_prompt_embed = torch.reshape(lookback, (lookback.shape[0] * lookback.shape[1], lookback.shape[2]))
        series_prompt_embed = series_prompt_embed.unsqueeze(1)
        series_prompt_embed = self.prompt_embedding(series_prompt_embed)
        uncond_latent_embeds = torch.zeros_like(series_prompt_embed)

        do_classifier_free_guidance = guidance_scale > 1.0
        if do_classifier_free_guidance:
            encoder_hidden_states = torch.cat([uncond_latent_embeds, series_prompt_embed], dim=0)
        else:
            encoder_hidden_states = series_prompt_embed

        shape = (lookback_embed.shape[0], self.token_num_out, self.token_dim)
        latents = randn_tensor(shape, generator=generator, device=device, dtype=encoder_hidden_states.dtype)
        latents = latents * scheduler.init_noise_sigma

        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta, scheduler)
        num_warmup_steps = len(timesteps) - num_inference_steps * scheduler.order

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if i == 0:
                    tmp_lookback_embed = lookback_embed.repeat((2 if do_classifier_free_guidance else 1), 1, 1)
                    tmp_t = torch.zeros_like(t)
                    self.reference_unet(
                        tmp_lookback_embed,
                        tmp_t,
                        encoder_hidden_states=encoder_hidden_states,
                    )
                    self.reference_control_reader.update(self.reference_control_writer)

                latent_model_input = latents.to(device).repeat((2 if do_classifier_free_guidance else 1), 1, 1)
                latent_model_input = scheduler.scale_model_input(latent_model_input, t)
                noise_pred = self.denoising_unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=encoder_hidden_states,
                )
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                latents = scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample
                if i == len(timesteps)-1 or ((i+1) > num_warmup_steps and (i+1) % scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(scheduler, "order", 1)
                        callback(step_idx, t, latents)

            self.reference_control_reader.clear()
            self.reference_control_writer.clear()

        latents = torch.reshape(latents, (-1, lookback_channel, latents.shape[-2], latents.shape[-1]))
        latents = self.forecasting_flatten(latents)
        latents = self.forecasting_head(latents)
        return latents

    def sampling(self, num_inference_steps, guidance_scale, generator, scheduler, gen_shape=(32, 1, 336), eta=0.0, callback=None, callback_steps=1):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        lookback = torch.zeros(gen_shape).to(device)

        scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = scheduler.timesteps
        lookback_embed, lookback_channel = self.pi_embedding(lookback)
        series_prompt_embed = torch.reshape(lookback, (lookback.shape[0] * lookback.shape[1], lookback.shape[2]))
        series_prompt_embed = series_prompt_embed.unsqueeze(1)
        series_prompt_embed = self.prompt_embedding(series_prompt_embed)
        uncond_latent_embeds = torch.zeros_like(series_prompt_embed)

        do_classifier_free_guidance = guidance_scale > 1.0
        if do_classifier_free_guidance:
            encoder_hidden_states = torch.cat([uncond_latent_embeds, series_prompt_embed], dim=0)
        else:
            encoder_hidden_states = series_prompt_embed

        shape = (lookback_embed.shape[0], self.token_num_out, self.token_dim)
        latents = randn_tensor(shape, generator=generator, device=device, dtype=encoder_hidden_states.dtype)
        latents = latents * scheduler.init_noise_sigma

        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta, scheduler)
        num_warmup_steps = len(timesteps) - num_inference_steps * scheduler.order

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if i == 0:
                    tmp_lookback_embed = lookback_embed.repeat((2 if do_classifier_free_guidance else 1), 1, 1)
                    tmp_t = torch.zeros_like(t)
                    self.reference_unet(
                        tmp_lookback_embed,
                        tmp_t,
                        encoder_hidden_states=encoder_hidden_states,
                    )
                    self.reference_control_reader.update(self.reference_control_writer)

                latent_model_input = latents.to(device).repeat((2 if do_classifier_free_guidance else 1), 1, 1)
                latent_model_input = scheduler.scale_model_input(latent_model_input, t)
                noise_pred = self.denoising_unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=encoder_hidden_states,
                )
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                latents = scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(scheduler, "order", 1)
                        callback(step_idx, t, latents)

            self.reference_control_reader.clear()
            self.reference_control_writer.clear()

        latents = torch.reshape(latents, (-1, lookback_channel, latents.shape[-2], latents.shape[-1]))
        latents = self.forecasting_flatten(latents)
        latents = self.forecasting_head(latents)
        return latents



