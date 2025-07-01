import torch
import torch.nn as nn
from transformers import T5Tokenizer, T5EncoderModel
from diffusers import UNet2DConditionModel


class AudioEncoder(nn.Module):
    def __init__(self, latent_ch: int = 32):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, 4, 2, 1), nn.GELU(),
            nn.Conv1d(32, 64, 4, 2, 1), nn.GELU(),
            nn.Conv1d(64, 128, 4, 2, 1), nn.GELU(),
        )
        self.mu = nn.Conv1d(128, latent_ch, 1)
        self.logvar = nn.Conv1d(128, latent_ch, 1)

    def forward(self, wav: torch.Tensor):
        h = self.conv(wav)
        return self.mu(h), self.logvar(h)
    

class AudioDecoder(nn.Module):
    def __init__(self, latent_ch: int = 32, target_len: int = 160_000):
        super().__init__()
        self.pre = nn.Conv1d(latent_ch, 128, 1)
        self.deconv = nn.Sequential(
            nn.ConvTranspose1d(128, 64, 4, 2, 1), nn.GELU(),
            nn.ConvTranspose1d(64 , 32, 4, 2, 1), nn.GELU(),
            nn.ConvTranspose1d(32 , 1 , 4, 2, 1),
        )
        self.target_len = target_len

    def forward(self, z: torch.Tensor):
        h = self.pre(z)
        wav = self.deconv(h)
        return wav[..., : self.target_len]
    

class PromptEncoder(nn.Module):
    """T5 encoder + mean pooling -> projection"""
    def __init__(self, model_name="t5-base", proj_dim=256):
        super().__init__()
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.encoder = T5EncoderModel.from_pretrained(model_name)
        self.proj = nn.Linear(self.encoder.config.d_model, proj_dim, bias=False)

    @torch.no_grad()
    def forward(self, prompts: list[str], device):
        tok = self.tokenizer(
            prompts, padding=True, truncation=True, return_tensors="pt"
        ).to(device)
        hid = self.encoder(**tok).last_hidden_state
        pooled = hid.mean(dim=1)
        return self.proj(pooled)
    

class CondUNet(nn.Module):
    def __init__(
        self,
        latent_ch: int,
        cond_dim: int,
        latent_steps: int,
        block_channels = (128, 256, 512),
        layers_per_block = 2,
        xattn_heads: int = 8,
    ):
        super().__init__()
        self.unet = UNet2DConditionModel(
            sample_size = (latent_steps, 1),
            in_channels = latent_ch,
            out_channels = latent_ch,
            cross_attention_dim = cond_dim,
            block_out_channels = block_channels,
            layers_per_block = layers_per_block,
            transformer_layers_per_block = layers_per_block,
            attention_head_dim = latent_ch // xattn_heads,
            down_block_types = (
                "CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "DownBlock2D",
            ),
            up_block_types = (
                "UpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D",
            ),
        )

    def forward(self, noisy_lat, timesteps, prompt_embed):
        h = noisy_lat.unsqueeze(-1)
        enc_hid = prompt_embed.unsqueeze(1)

        eps = self.unet(
            sample=h,
            timestep=timesteps,
            encoder_hidden_states=enc_hid,
        ).sample.squeeze(-1)

        return eps
