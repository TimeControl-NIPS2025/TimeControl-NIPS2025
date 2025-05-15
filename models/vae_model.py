import math
import torch
from torch import nn


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


class Encoder(nn.Module):
    def __init__(self, d_model, patch_len, stride, dropout=0.2, hidden_dim=(1024, 2048, 2048, 1024)):
        super(Encoder, self).__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch_layer = nn.ReplicationPad1d((0, stride))
        self.value_embedding = TokenEmbedding(patch_len, d_model)
        self.position_embedding = PositionalEmbedding(d_model)
        self.dropout = nn.Dropout(dropout)
        self.mlp = nn.ModuleList()
        for i in range(len(hidden_dim)):
            if i == 0:
                self.mlp.append(
                    nn.Sequential(
                        nn.Dropout(p=dropout),
                        nn.Linear(d_model, hidden_dim[i]),
                        nn.LayerNorm(hidden_dim[i]),
                        nn.PReLU(),
                    )
                )
            else:
                self.mlp.append(
                    nn.Sequential(
                        nn.Dropout(p=dropout),
                        nn.Linear(hidden_dim[i-1], hidden_dim[i]),
                        nn.LayerNorm(hidden_dim[i]),
                        nn.PReLU(),
                    )
                )
        self.mlp.append(nn.Linear(hidden_dim[-1], d_model))

    def forward(self, x):
        n_channel = x.shape[1]
        x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        ve = self.value_embedding(x)
        pe = self.position_embedding(x)
        x_embed = ve + pe
        x_embed = self.dropout(x_embed)
        for i, layer in enumerate(self.mlp):
            x_embed = layer(x_embed)
        return x_embed, n_channel


class Decoder(nn.Module):
    def __init__(self, input_seq_len, output_seq_len, dropout=0.2, hidden_dim=(1024, 2048, 1024)):
        super(Decoder, self).__init__()
        self.flatten = nn.Flatten(start_dim=-2)
        self.mlp = nn.ModuleList()
        for i in range(len(hidden_dim)):
            if i == 0:
                self.mlp.append(
                    nn.Sequential(
                        nn.Dropout(p=dropout),
                        nn.Linear(input_seq_len, hidden_dim[i]),
                        nn.LayerNorm(hidden_dim[i]),
                        nn.PReLU(),
                    )
                )
            else:
                self.mlp.append(
                    nn.Sequential(
                        nn.Dropout(p=dropout),
                        nn.Linear(hidden_dim[i-1], hidden_dim[i]),
                        nn.LayerNorm(hidden_dim[i]),
                        nn.PReLU(),
                    )
                )
        self.mlp.append(nn.Linear(hidden_dim[-1], output_seq_len))

    def forward(self, x):
        x = x.permute(0, 1, 3, 2)
        x = self.flatten(x)
        for i, layer in enumerate(self.mlp):
            x = layer(x)
        return x


class VAE(torch.nn.Module):
    def __init__(self, d_model, patch_len, stride, input_seq_len, output_seq_len, device="cuda"):
        super(VAE, self).__init__()
        self.device = device
        self.encoder = Encoder(
            d_model=d_model,
            patch_len=patch_len,
            stride=stride,
        )
        self.decoder = Decoder(
            input_seq_len=input_seq_len,
            output_seq_len=output_seq_len,
        )
        self.hparams = {
            "autoencoder_wd": 0.01,
            "autoencoder_lr": 1.0e-5,
        }
        get_params = lambda model, cond: list(model.parameters()) if cond else []
        self.optimizer_autoencoder = torch.optim.AdamW(
            params=(get_params(self.encoder, True)+get_params(self.decoder, True)),
            lr=self.hparams["autoencoder_lr"],
            weight_decay=self.hparams["autoencoder_wd"],
        )
        self.loss_autoencoder = nn.MSELoss(reduction='mean')
        self.iteration = 0

    def forward(self, inputs, only_encoder=False, only_decoder=False):
        if only_decoder:
            reconstructions = self.decoder(inputs)
            reconstructions = nn.ReLU()(reconstructions)
            return reconstructions

        latents, n_channel = self.encoder(inputs)
        if only_encoder:
            return latents

        latents = torch.reshape(latents, (-1, n_channel, latents.shape[-2], latents.shape[-1]))
        reconstructions = self.decoder(latents)
        return reconstructions

    def pretrain(self, inputs):
        inputs = inputs.to(self.device)
        reconstructions = self.forward(inputs)
        recon_loss = self.loss_autoencoder(reconstructions, inputs)
        self.optimizer_autoencoder.zero_grad()
        recon_loss.backward()
        self.optimizer_autoencoder.step()
        self.iteration += 1
        return recon_loss.item()
