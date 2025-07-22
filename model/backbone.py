import torch
import torch.nn as nn
from transformers import T5Tokenizer, T5EncoderModel, AutoModel, AutoTokenizer
from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor
from diffusers import UNet2DConditionModel
import torch.nn.functional as F
import xformers


class AudioEncoder(nn.Module):
    def __init__(self, latent_ch: int = 64,
                 w2v_ckpt: str = "facebook/wav2vec2-base",
                 trainable: bool = False):
        super().__init__()
        self.backbone = Wav2Vec2Model.from_pretrained(w2v_ckpt)
        if not trainable:
            self.backbone.eval()
            for p in self.backbone.parameters():
                p.requires_grad_(False)
        hd = self.backbone.config.hidden_size
        self.mu = nn.Conv1d(hd, latent_ch, 1)
        self.logvar = nn.Conv1d(hd, latent_ch, 1)

    def forward(self, wav: torch.Tensor):
        inp = wav.squeeze(1)
        hid = self.backbone(input_values=inp).last_hidden_state
        hid = hid.transpose(1, 2)
        return self.mu(hid), self.logvar(hid)


from torch.nn.utils import spectral_norm as sn

def upsample_block(c_in, c_out, k=5):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="nearest"),
        sn(nn.Conv1d(c_in, c_out, k, padding=k//2)),
        nn.GELU(),
    )

class ResStack(nn.Module):
    def __init__(self, ch, k=3):
        super().__init__()
        dilations = (1, 3, 9)
        self.convs = nn.ModuleList([
            sn(nn.Conv1d(ch, ch, k, padding=d, dilation=d))
            for d in dilations
        ])

    def forward(self, x):
        for conv in self.convs:
            x = x + nn.GELU()(conv(x))
        return x

class AudioDecoder(nn.Module):
    def __init__(self, latent_ch=96, target_len=160_000):
        super().__init__()
        self.pre = sn(nn.Conv1d(latent_ch, 512, 1))

        self.up = nn.Sequential(
            upsample_block(512, 256),
            ResStack(256), ResStack(256),
            upsample_block(256, 128),
            ResStack(128), ResStack(128),
            upsample_block(128, 64),
            ResStack(64), ResStack(64),
            upsample_block(64, 32),
            ResStack(32), ResStack(32),
            nn.Upsample(scale_factor=5, mode="nearest"),
            sn(nn.Conv1d(32, 16, 5, padding=2)),
            nn.GELU(),
        )
        self.post = sn(nn.Conv1d(16, 1, 7, padding=3))
        self.tanh = nn.Tanh()
        self.target_len = target_len

    def forward(self, z, target_len: int | None = None):
        h   = self.pre(z)
        h   = self.up(h)
        wav = self.tanh(self.post(h))
        if target_len is not None:
            wav = F.interpolate(wav, size=target_len, mode="nearest")
        return wav


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
            prompts, padding=True, truncation=True, return_tensors="pt", max_length=512
        ).to(device)
        hid = self.encoder(**tok).last_hidden_state
        pooled = hid.mean(dim=1)
        return self.proj(pooled)
    

MODEL_PRESETS = {
    "mini":  "sentence-transformers/all-MiniLM-L6-v2",
    "base":  "sentence-transformers/paraphrase-MiniLM-L12-v2",
    "clip":  "openai/clip-vit-base-patch32",
}

class PromptEncoderv2(nn.Module):
    """
    Small & fast sentence encoder with mean-pooling.
    ----
    Args
    ----
    preset      : "mini" | "base" | "clip" | HF model_id
    proj_dim    : int, projection dimension (default: 256)
    trainable   : bool, whether to train the text encoder (default: True)
    """
    def __init__(self, preset="mini", proj_dim=256, trainable=True):
        super().__init__()
        model_name = MODEL_PRESETS.get(preset, preset)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder   = AutoModel.from_pretrained(model_name)
        enc_dim = self.encoder.config.hidden_size
        self.proj = nn.Linear(enc_dim, proj_dim, bias=False) \
                    if proj_dim != enc_dim else nn.Identity()
        if not trainable:
            self.encoder.eval()
            for p in self.encoder.parameters():
                p.requires_grad_(False)

    @staticmethod
    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def forward(self, prompts: list[str], device):
        tok = self.tokenizer(
            prompts, padding=True, truncation=True, return_tensors="pt"
        ).to(device)

        out = self.encoder(**tok)
        emb = self.mean_pooling(out, tok['attention_mask'])
        emb = F.normalize(emb, p=2, dim=1)
        return self.proj(emb)


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
        self.unet.enable_xformers_memory_efficient_attention()

    def forward(self, noisy_lat, timesteps, prompt_embed):
        h = noisy_lat.unsqueeze(-1)
        enc_hid = prompt_embed.unsqueeze(1)

        eps = self.unet(
            sample=h,
            timestep=timesteps,
            encoder_hidden_states=enc_hid,
        ).sample.squeeze(-1)

        return eps


class VRFMCondUNet(nn.Module):
    def __init__(
        self,
        latent_ch: int,
        cond_dim: int,
        latent_steps: int,
        extra_cond_dim: int = 0,
        block_channels=(128, 256, 512),
        layers_per_block=2,
        xattn_heads: int = 8,
    ):
        super().__init__()

        in_ch = latent_ch + extra_cond_dim if extra_cond_dim else latent_ch
        self.use_extra = extra_cond_dim > 0

        if self.use_extra and extra_cond_dim != latent_ch:
            self.z_proj = nn.Conv1d(extra_cond_dim, extra_cond_dim, 1)
        else:
            self.z_proj = nn.Identity()

        self.unet = UNet2DConditionModel(
            sample_size=(latent_steps, 1),
            in_channels=in_ch,
            out_channels=latent_ch,
            cross_attention_dim=cond_dim,
            block_out_channels=block_channels,
            layers_per_block=layers_per_block,
            transformer_layers_per_block=layers_per_block,
            attention_head_dim=latent_ch // xattn_heads,
            down_block_types=(
                "CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "DownBlock2D",
            ),
            up_block_types=(
                "UpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D",
            ),
        )

    def forward(
        self,
        noisy_lat: torch.Tensor,          # (B, C, T)
        timesteps: torch.Tensor,          # (B,)
        prompt_embed: torch.Tensor,       # (B, D)
        z_lat: torch.Tensor | None = None # (B, C_ex, T) or None
    ) -> torch.Tensor:
        if self.use_extra:
            if z_lat is None:
                raise ValueError("z_lat must be provided when extra_cond_dim>0")
            z_feat = self.z_proj(z_lat) # (B, C_ex, T)
            x = torch.cat([noisy_lat, z_feat], dim=1)
        else:
            x = noisy_lat

        h = x.unsqueeze(-1) # (B,C,T,1)
        enc_hid = prompt_embed.unsqueeze(1) # (B,1,D)

        out = self.unet(
            sample=h,
            timestep=timesteps,
            encoder_hidden_states=enc_hid,
        ).sample.squeeze(-1) # -> (B,C,T)

        return out