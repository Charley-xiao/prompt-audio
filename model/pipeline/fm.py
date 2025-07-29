import torch
import pytorch_lightning as pl
import torch.nn.functional as F
from model.backbone import CondUNet, PromptEncoderv2
from torcheval.metrics import FrechetAudioDistance
from torchmetrics.aggregation import MeanMetric
from model.clap_module import CLAPAudioEmbedding
from pytorch_lightning.utilities import rank_zero_only
from functools import lru_cache
from matplotlib import pyplot as plt


class FlowMatchingPipeline(pl.LightningModule):
    def __init__(
        self,
        lr=2e-4,
        sample_length=16000,
        n_val_epochs=1,
        cfg_drop_prob=0.1,
        disable_text_enc=False,
        rho=2.0,
        alpha_t=1.0,
        beta_t=1.0,
        loss_gamma=0.0, # w(t)=t^gamma * (1-t)^delta
        loss_delta=0.0,
        noise_std=1.0,
        solver="heun", # "heun" | "euler"
    ):
        super().__init__()
        self.save_hyperparameters()
        if not disable_text_enc:
            self.textenc = PromptEncoderv2(proj_dim=128, preset="mini", trainable=False)
        else:
            cfg_drop_prob = 0.0
        self.unet = CondUNet(
            in_ch=1,
            cond_dim=128,
            latent_steps=sample_length,
            block_channels=(192, 384, 768)
        )

        self.fad = None
        self.clap = None
        self.clap_sim = MeanMetric()
        self.val_interval = n_val_epochs
        self.cfg_drop_prob = cfg_drop_prob

        if self.cfg_drop_prob > 0:
            print(f"CFG drop probability set to {self.cfg_drop_prob:.2f}")
        else:
            print("IMPORTANT: CFG drop probability is 0, no classifier-free guidance will be applied!")

    def setup(self, stage=None):
        print(f"Setting up FlowMatchingPipeline on {self.device}")
        torch.backends.cudnn.benchmark = True
        if self.fad is None:
            # Uses VGGish; keep on self.device as before
            self.fad = FrechetAudioDistance.with_vggish(device="cpu").to(self.device)
        if self.clap is None:
            self.clap = CLAPAudioEmbedding(device=self.device)

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="min", factor=0.5, patience=6, min_lr=1e-6
        )
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": sched, "monitor": "val_FAD", "frequency": 1, "interval": "epoch"},
        }

    # ------------------------------ Helpers ------------------------------

    @lru_cache(maxsize=2048)
    def _cached_text(self, txt: str):
        return self.textenc([txt])

    @staticmethod
    def _beta_sample(shape, alpha, beta, device):
        # torch Beta: sample via Gamma
        g1 = torch.distributions.Gamma(alpha, 1.0).sample(shape).to(device)
        g2 = torch.distributions.Gamma(beta, 1.0).sample(shape).to(device)
        return g1 / (g1 + g2 + 1e-12)

    def _make_time(self, bsz):
        if self.hparams.alpha_t == 1.0 and self.hparams.beta_t == 1.0:
            t = torch.rand(bsz, 1, 1, device=self.device)
        else:
            t = self._beta_sample((bsz, 1, 1), self.hparams.alpha_t, self.hparams.beta_t, self.device)
        # avoid exact 0 when rho<1; here rho>=1 by default
        return t.clamp_(1e-6, 1 - 1e-6)

    def _interpolant(self, x, z, t):
        rho = self.hparams.rho
        a = t.pow(rho)                              # (B,1,1)
        x_t = (1.0 - a) * z + a * x
        v_star = rho * t.pow(rho - 1.0) * (x - z)   # (B,1,1) * (B,1,T)
        return x_t, v_star

    def _loss_weight(self, t):
        """w(t)=t^gamma * (1-t)^delta."""
        g, d = self.hparams.loss_gamma, self.hparams.loss_delta
        if g == 0.0 and d == 0.0:
            return 1.0
        return (t.pow(g) * (1.0 - t).pow(d)).detach()

    # ------------------------------ Training ------------------------------

    def forward(self, wav, prompt):
        """returns (x_t, v_star, cond, t_flat)."""
        bsz = wav.size(0)
        # Conditioning (CFG train: drop some)
        if self.hparams.disable_text_enc:
            cond = torch.zeros((bsz, 128), device=self.device)
        else:
            cond = self.textenc(prompt)
            keep_mask = (torch.rand(bsz, 1, device=cond.device) >= self.cfg_drop_prob).float()
            cond = cond * keep_mask

        # z ~ N(0, noise_std^2)
        z = self.hparams.noise_std * torch.randn_like(wav)

        # t ~ Beta(α,β)
        t = self._make_time(bsz)  # (B,1,1)
        x_t, v_star = self._interpolant(wav, z, t)
        return x_t, v_star, cond, t.view(-1)  # t flattened for UNet

    def training_step(self, batch, _):
        x, prompt = batch                       # x: [B,1,T] waveform in [-1,1]
        x_t, v_star, cond, t_flat = self(x, prompt)

        # Model predicts velocity at (t, x_t, cond)
        v_pred = self.unet(x_t, t_flat, cond)   # expect shape [B,1,T]

        # Optional loss weighting over t
        w = self._loss_weight(t_flat.view(-1, 1, 1))
        loss_fm = (w * (v_pred - v_star).pow(2)).mean()

        self.log_dict(
            {
                "loss": loss_fm,
                "t_mean": t_flat.mean(),
                "t_std": t_flat.std(unbiased=False),
            },
            prog_bar=True, sync_dist=True, on_step=True, on_epoch=True
        )
        return loss_fm

    # ------------------------------ Validation / Metrics ------------------------------

    def _update_fad(self, pred, target):
        self.fad.update(pred, target)

    def _update_clap(self, pred, target_texts):
        a_emb = self.clap(pred)
        t_emb = self.clap.text_embed(target_texts)
        sim = F.cosine_similarity(a_emb, t_emb)
        self.clap_sim.update(sim)

    def validation_step(self, batch, batch_idx):
        wav_gt, prompts = batch
        wav_gen = self.generate(
            prompts,
            num_steps=100,
            to_cpu=False,
            guidance_scale=0.3 if self.cfg_drop_prob > 0 else None
        )
        self._update_fad(wav_gen.squeeze(1), wav_gt.squeeze(1))
        self._update_clap(wav_gen, prompts)
        if batch_idx == 0:
            self._plot_wavs(wav_gen, wav_gt, batch_idx)

    @rank_zero_only
    def _plot_wavs(self, wav_gen, wav_gt, batch_idx):
        # Specgram
        for i in range(min(3, wav_gen.size(0))):
            fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
            gen_sig = wav_gen[i].detach().cpu().float().squeeze().numpy()
            gt_sig  = wav_gt[i].detach().cpu().float().squeeze().numpy()
            axs[0].specgram(gen_sig, NFFT=256, Fs=16000, noverlap=128, scale='dB', mode='magnitude')
            axs[0].set_title(f"Generated Audio {i+1}")
            axs[1].specgram(gt_sig,  NFFT=256, Fs=16000, noverlap=128, scale='dB', mode='magnitude')
            axs[1].set_title(f"Ground Truth Audio {i+1}")
            plt.tight_layout()
            plt.savefig(f"samples/spec_{self.current_epoch}_{i}.png")
            plt.close(fig)
        # Waveform
        for i in range(min(3, wav_gen.size(0))):
            fig = plt.figure(figsize=(10, 4))
            gen_sig = wav_gen[i].detach().cpu().float().squeeze().numpy()
            gt_sig  = wav_gt[i].detach().cpu().float().squeeze().numpy()
            plt.plot(gen_sig, label="Generated")
            plt.plot(gt_sig, alpha=0.5, label="Ground Truth")
            plt.legend()
            plt.title(f"Waveform Comparison {i+1}")
            plt.xlabel("Time")
            plt.ylabel("Amplitude")
            plt.savefig(f"samples/wav_{self.current_epoch}_{i}.png")
            plt.close(fig)

    def on_validation_epoch_end(self):
        if (self.current_epoch + 1) % self.val_interval:
            return
        fad_value = self.fad.compute()
        self.log("val_FAD", fad_value, prog_bar=True, sync_dist=True)
        self.fad.reset()

        score = self.clap_sim.compute()
        self.log("val_CLAPSim", score, prog_bar=True, sync_dist=True)
        self.clap_sim.reset()

    # ------------------------------ Sampling (ODE integration) ------------------------------

    @torch.no_grad()
    def generate(
        self,
        prompts: list[str],
        num_steps: int = 50,
        guidance_scale: float | None = 0.3,
        to_cpu: bool = True,
    ):
        self.eval()
        device = self.device

        # Conditioning
        if self.hparams.disable_text_enc:
            cond = torch.zeros((len(prompts), 128), device=device)
        else:
            cond = torch.cat([self._cached_text(t) for t in prompts], dim=0)

        # Initial noise (z at t=0)
        x = self.hparams.noise_std * torch.randn(
            len(prompts), 1, self.hparams.sample_length, device=device
        )

        # Time grid 0..1
        t_grid = torch.linspace(0.0, 1.0, steps=num_steps + 1, device=device)
        solver = self.hparams.solver.lower()

        def v_pred_fn(x_in, t_scalar, cond_vec):
            if guidance_scale is None or guidance_scale == 0.0 or self.cfg_drop_prob <= 0.0:
                return self.unet(x_in, t_scalar, cond_vec)
            else:
                v_uc = self.unet(x_in, t_scalar, torch.zeros_like(cond_vec))
                v_c  = self.unet(x_in, t_scalar, cond_vec)
                return v_c + guidance_scale * (v_c - v_uc)

        # Integrate dx/dt = v_theta(t, x_t, cond) from t=0→1
        for i in range(num_steps):
            t0 = t_grid[i].expand(x.size(0))
            dt = float(t_grid[i + 1] - t_grid[i])

            if solver == "euler":
                v1 = v_pred_fn(x, t0, cond)
                x  = x + v1 * dt
            else:  # Heun (RK2)
                v1 = v_pred_fn(x, t0, cond)
                x_euler = x + v1 * dt
                t1 = t_grid[i + 1].expand(x.size(0))
                v2 = v_pred_fn(x_euler, t1, cond)
                x  = x + 0.5 * (v1 + v2) * dt

        wav = x  # model operates directly on waveform
        return wav.cpu() if to_cpu else wav
