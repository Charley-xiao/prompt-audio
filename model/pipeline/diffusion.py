import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from functools import lru_cache
from torcheval.metrics import FrechetAudioDistance
from torchmetrics.aggregation import MeanMetric
from diffusers import DDPMScheduler, DDIMScheduler, UNet2DConditionModel
from model.backbone import PromptEncoderv2
from model.clap_module import CLAPAudioEmbedding
from pytorch_lightning.utilities import rank_zero_only
from matplotlib import pyplot as plt


class DiffusionPipeline(pl.LightningModule):
    def __init__(
        self,
        sample_length: int = 16_000,
        unet_base_channels: tuple[int, ...] = (64, 128, 256, 512),
        num_steps: int = 1_000,
        lr: float = 2e-4,
        cfg_drop_prob: float = 0.1,
        disable_text_enc: bool = False,
        val_interval: int = 1,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.unet = UNet2DConditionModel(
            sample_size=(sample_length, 1),  # (H, W)
            in_channels=1,
            out_channels=1,
            cross_attention_dim=128,
            block_out_channels=list(unet_base_channels),
            layers_per_block=2,
            transformer_layers_per_block=2,
        )
        self.unet.enable_xformers_memory_efficient_attention()

        self.disable_text_enc = disable_text_enc
        if disable_text_enc:
            cfg_drop_prob = 0.0  # ensure no CFG if encoder off
        self.cfg_drop_prob = cfg_drop_prob
        if self.cfg_drop_prob > 0:
            print(f"CFG drop probability set to {self.cfg_drop_prob:.2f}")
        else:
            print("IMPORTANT: CFG drop probability is 0, no classifier-free guidance will be applied!")

        self.textenc = None if disable_text_enc else PromptEncoderv2(proj_dim=128, preset="mini", trainable=False)

        self.sched_train = DDPMScheduler(num_train_timesteps=num_steps)
        self.sched_eval = DDIMScheduler.from_config(self.sched_train.config, clip_sample=False)

        self.val_interval = val_interval
        self.fad = None
        self.clap = None
        self.clap_sim = MeanMetric()

    def setup(self, stage: str | None = None):
        print(f"Setting up DiffusionPipeline on {self.device}")
        self._sched_to_device(self.sched_train, self.device)
        self._sched_to_device(self.sched_eval, self.device)
        torch.backends.cudnn.benchmark = True
        if self.fad is None:
            self.fad = FrechetAudioDistance.with_vggish(device="cpu").to(self.device)
        if self.clap is None:
            self.clap = CLAPAudioEmbedding(device=self.device)

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=6, min_lr=1e-6)
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": sched, "monitor": "val_FAD", "frequency": 1, "interval": "epoch"},
        }

    def training_step(self, batch, batch_idx):
        wav, prompts = batch  # (B,1,T)
        bsz, _, T = wav.shape
        device = wav.device

        # --- text condition & CFG mask
        if self.disable_text_enc:
            cond = torch.zeros((bsz, 128), device=device)
        else:
            cond = self.textenc(prompts)
        keep_mask = (torch.rand(bsz, 1, device=device) >= self.cfg_drop_prob).float()
        cond_noised = cond * keep_mask

        # --- diffusion objective
        t = torch.randint(0, self.sched_train.config.num_train_timesteps, (bsz,), device=device, dtype=torch.long)
        noise = torch.randn_like(wav)
        noisy_wav = self.sched_train.add_noise(wav, noise, t)

        # UNet expects (B, 1, H, W) → expand last dim
        noise_pred = self.unet(noisy_wav.unsqueeze(-1), t, encoder_hidden_states=cond_noised).sample.squeeze(-1)
        loss = F.mse_loss(noise_pred, noise)
        self.log("loss", loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        wav_gt, prompts = batch
        wav_gen = self.generate(
            prompts, num_steps=50, guidance_scale=0.3 if self.cfg_drop_prob > 0 else None, to_cpu=False
        )
        self._update_fad(wav_gen.squeeze(1), wav_gt.squeeze(1))
        self._update_clap(wav_gen, prompts)
        if batch_idx == 0:
            self._plot_wavs(wav_gen, wav_gt, batch_idx)

    def on_validation_epoch_end(self):
        if (self.current_epoch + 1) % self.val_interval:
            return
        fad_value = self.fad.compute()
        self.log("val_FAD", fad_value, prog_bar=True, sync_dist=True)
        self.fad.reset()

        score = self.clap_sim.compute()
        self.log("val_CLAPSim", score, prog_bar=True, sync_dist=True)
        self.clap_sim.reset()

    @torch.no_grad()
    def generate(self, prompts: list[str], num_steps: int = 50, guidance_scale: float | None = 0.3, to_cpu: bool = True):
        self.eval()
        device = self.device

        # Encode prompts
        if self.disable_text_enc:
            cond = torch.zeros((len(prompts), 128), device=device)
        else:
            cond = torch.cat([self._cached_text(p) for p in prompts], dim=0)

        if guidance_scale is None:
            def eps_fn(x, t):
                return self.unet(x.unsqueeze(-1), t, encoder_hidden_states=cond).sample.squeeze(-1)
        else:
            uncond = torch.zeros_like(cond)

            def eps_fn(x, t):
                eps_uc = self.unet(x.unsqueeze(-1), t, encoder_hidden_states=uncond).sample.squeeze(-1)
                eps_c = self.unet(x.unsqueeze(-1), t, encoder_hidden_states=cond).sample.squeeze(-1)
                return eps_c + guidance_scale * (eps_c - eps_uc)

        wav = torch.randn(len(prompts), 1, self.hparams.sample_length, device=device)
        self.sched_eval.set_timesteps(num_steps, device=device)
        for t in self.sched_eval.timesteps:
            wav = self.sched_eval.step(eps_fn(wav, t), t, wav).prev_sample
        return wav.cpu() if to_cpu else wav

    def _update_fad(self, pred, target):
        self.fad.update(pred, target)

    def _update_clap(self, pred, prompts):
        a_emb = self.clap(pred)
        t_emb = self.clap.text_embed(prompts)
        sim = F.cosine_similarity(a_emb, t_emb)
        self.clap_sim.update(sim)

    @rank_zero_only
    def _plot_wavs(self, wav_gen, wav_gt, batch_idx):
        for i in range(min(3, wav_gen.size(0))):
            fig = plt.figure(figsize=(10, 4))
            plt.plot(wav_gen[i].cpu().squeeze().numpy(), label="Generated")
            plt.plot(wav_gt[i].cpu().squeeze().numpy(), alpha=0.5, label="Ground Truth")
            plt.legend()
            plt.title(f"Waveform Comparison {i+1}")
            plt.xlabel("Time")
            plt.ylabel("Amplitude")
            plt.savefig(f"samples/wav_{self.current_epoch}_{i}.png")
            plt.close(fig)

    @staticmethod
    def _sched_to_device(sched, device):
        for k, v in sched.__dict__.items():
            if torch.is_tensor(v):
                setattr(sched, k, v.to(device))
            elif isinstance(v, dict):
                for kk, vv in v.items():
                    if torch.is_tensor(vv):
                        v[kk] = vv.to(device)

    @lru_cache(maxsize=1024)
    def _cached_text(self, txt: str):
        return self.textenc([txt])
