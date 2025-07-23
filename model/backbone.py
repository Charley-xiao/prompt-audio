import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from transformers import Wav2Vec2Model
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


def upsample_block(c_in, c_out, k=5):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="nearest"),
        nn.Conv1d(c_in, c_out, k, padding=k//2),
        nn.GELU(),
    )

class ResStack(nn.Module):
    def __init__(self, ch, k=3):
        super().__init__()
        dilations = (1, 3, 9)
        self.convs = nn.ModuleList([
            nn.Conv1d(ch, ch, k, padding=d, dilation=d)
            for d in dilations
        ])

    def forward(self, x):
        for conv in self.convs:
            x = x + nn.GELU()(conv(x))
        return x

class AudioDecoder(nn.Module):
    def __init__(
        self,
        latent_ch: int = 64,
        d_model: int = 768,
        nhead: int = 12,
        num_layers: int = 12,
        target_len: int = 160_000,
    ):
        super().__init__()
        self.prenet = nn.Conv1d(latent_ch, d_model, kernel_size=1)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.proj = nn.Conv1d(d_model, 512, kernel_size=1)
        self.up = nn.Sequential(
            upsample_block(512, 512),
            ResStack(512), ResStack(512),
            upsample_block(512, 256),
            ResStack(256), ResStack(256),
            upsample_block(256, 128),
            ResStack(128), ResStack(128),
            upsample_block(128, 64),
            ResStack(64), ResStack(64),
            nn.Upsample(scale_factor=5, mode="nearest"),
            nn.Conv1d(64, 16, kernel_size=5, padding=2),
            nn.GELU(),
        )
        self.post = nn.Conv1d(16, 1, kernel_size=7, padding=3)
        self.tanh = nn.Tanh()
        self.target_len = target_len

    def forward(self, z: torch.Tensor, target_len: int | None = None):
        h = self.prenet(z) # (B, d_model, T_latent)
        h = h.transpose(1, 2) # (B, T_latent, d_model) for transformer
        h = self.transformer(h)
        h = h.transpose(1, 2) # back to (B, d_model, T_latent)
        h = self.proj(h) # (B, 512, T_latent)
        h = self.up(h) # upsampled feature map
        wav = self.tanh(self.post(h)) # (B, 1, ~target_len)
        if target_len is not None:
            wav = F.interpolate(wav, size=target_len, mode="nearest")
        return wav
    

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

    def forward(self, prompts: list[str]):
        device = self.proj.weight.device
        tok_cpu = self.tokenizer(
            prompts, padding=True, truncation=True, return_tensors="pt"
        )
        tok = {k: v.to(device) for k, v in tok_cpu.items()}
        if next(self.encoder.parameters()).device != device:
            self.encoder.to(device)
        out = self.encoder(**tok)
        emb = self.mean_pooling(out, tok["attention_mask"])
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
