import torch
import pytorch_lightning as pl
import torch.nn.functional as F
from model.backbone import AudioEncoder, AudioDecoder, CondUNet, PromptEncoderv2
from diffusers import DDPMScheduler, DDIMScheduler
from torcheval.metrics import FrechetAudioDistance
from torchmetrics.aggregation import MeanMetric
from model.clap_module import CLAPAudioEmbedding
from pytorch_lightning.utilities import rank_zero_only
from functools import lru_cache
from matplotlib import pyplot as plt


class DiffusionVAEPipeline(pl.LightningModule):
    def __init__(
        self,
        latent_ch=64,
        lr=2e-4,
        beta_kl=0.0001,
        beta_rec=100.0,
        sample_length=16000,
        noise_steps=1000,
        n_val_epochs=1,
        cfg_drop_prob=0.1,
        disable_text_enc=False
    ):
        super().__init__()
        self.save_hyperparameters()

        self.encoder = AudioEncoder(latent_ch, trainable=False)
        for n,p in self.encoder.backbone.named_parameters():
            if n.startswith("encoder.layers.11") or n.startswith("project_q"):
                print(f"Making {n} trainable")
                p.requires_grad_(True)
        self.decoder = AudioDecoder(latent_ch, target_len=sample_length)
        if not disable_text_enc:
            self.textenc = PromptEncoderv2(proj_dim=128, preset="mini", trainable=False)
        else:
            cfg_drop_prob = 0
        latent_steps = sample_length // 320
        self.unet = CondUNet(latent_ch, 
                             cond_dim=128, 
                             latent_steps=latent_steps, 
                             block_channels=(192,384,768))
        self.sched_train = DDPMScheduler(num_train_timesteps=noise_steps)
        self.sched_eval  = DDIMScheduler.from_config(self.sched_train.config, clip_sample=False)

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
        print(f"Setting up DiffusionVAEPipeline on {self.device}")
        self.scheduler_to_device(self.sched_train, self.device)
        self.scheduler_to_device(self.sched_eval, self.device)
        torch.backends.cudnn.benchmark = True
        if self.fad is None:
            self.fad = FrechetAudioDistance.with_vggish(device="cpu").to(self.device)
        if self.clap is None:
            self.clap = CLAPAudioEmbedding(device=self.device)

    @staticmethod
    def reparameterize(mu, logvar):
        std, eps = (0.5 * logvar).exp(), torch.randn_like(mu)
        return mu + std * eps

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt,
            mode="min",
            factor=0.5,
            patience=6,
            min_lr=1e-6,
        )
        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": sched,
                "monitor":   "val_FAD",
                "frequency": 1,
                "interval":  "epoch",
            },
        }

    def forward(self, wav, prompt):
        mu, logvar = self.encoder(wav)
        z = self.reparameterize(mu, logvar)
        if self.hparams.disable_text_enc:
            prompt_e = torch.zeros((wav.size(0), 128), device=self.device) # 128: proj_dim
        else:
            prompt_e = self.textenc(prompt)
        return z, mu, logvar, prompt_e

    def training_step(self, batch, _):
        wav, prompt = batch
        z, mu, logvar, prompt_e = self(wav, prompt)
        keep_mask = (torch.rand(prompt_e.size(0), 1,
                                device=prompt_e.device) >= self.cfg_drop_prob).float()
        prompt_e = prompt_e * keep_mask
        bsz = z.size(0)
        t = torch.randint(
            0, self.sched_train.config.num_train_timesteps,
            (bsz,), device=self.device, dtype=torch.long
        )
        noise  = torch.randn_like(z)
        noisy_z = self.sched_train.add_noise(z, noise, t)
        noise_pred = self.unet(noisy_z, t, prompt_e)
        loss_diff  = F.mse_loss(noise_pred, noise)
        kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        dec_wav = self.decoder(z, target_len=wav.size(-1))
        rec = F.l1_loss(dec_wav, wav)
        sigma = (0.5 * logvar).exp()
        loss = loss_diff + self.hparams.beta_kl * kld + self.hparams.beta_rec * rec
        self.log_dict(
            {
                "loss": loss,
                "L_diff": loss_diff,
                "L_KL": kld,
                "L_rec": rec,
                "mu_mean": torch.mean(mu),
                "sigma_mean": torch.mean(sigma),
            },
            prog_bar=True, sync_dist=True, on_step=True, on_epoch=True
        )
        return loss
    
    def _update_fad(self, pred, target):
        self.fad.update(pred, target)

    def _update_clap(self, pred, target):
        a_emb = self.clap(pred)
        t_emb = self.clap.text_embed(target)
        sim = F.cosine_similarity(a_emb, t_emb)
        self.clap_sim.update(sim)
    
    def validation_step(self, batch, batch_idx):
        wav_gt, prompts = batch
        wav_gen = self.generate(prompts, num_steps=100, to_cpu=False, guidance_scale=0.3 if self.cfg_drop_prob > 0 else None)
        self._update_fad(wav_gen.squeeze(1), wav_gt.squeeze(1))
        self._update_clap(wav_gen, prompts)
        if batch_idx == 0:
            self._plot_wavs(wav_gen, wav_gt, batch_idx)

    @rank_zero_only
    def _plot_wavs(self, wav_gen, wav_gt, batch_idx):
        # Specgram
        for i in range(min(3, wav_gen.size(0))):
            fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
            gen_sig = wav_gen[i].cpu().float().squeeze().numpy()
            gt_sig  = wav_gt[i].cpu().float().squeeze().numpy()
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
            gen_sig = wav_gen[i].cpu().float().squeeze().numpy()
            gt_sig = wav_gt[i].cpu().float().squeeze().numpy()
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

    @staticmethod
    def scheduler_to_device(sched, device):
        for k, v in sched.__dict__.items():
            if torch.is_tensor(v):
                setattr(sched, k, v.to(device))
            elif isinstance(v, dict):
                for kk, vv in v.items():
                    if torch.is_tensor(vv):
                        v[kk] = vv.to(device)

    @lru_cache(maxsize=2048)
    def _cached_text(self, txt: str):
        return self.textenc([txt])

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
        if self.hparams.disable_text_enc:
            cond = torch.zeros((len(prompts), 128), device=device)  # 128: proj_dim
        else:
            cond = torch.cat([self._cached_text(t) for t in prompts], dim=0)
        if not guidance_scale:
            eps_fn = lambda lat, t: self.unet(lat, t, cond)
        else:
            uncond = torch.zeros_like(cond)
            def eps_fn(lat, t):
                eps_uc = self.unet(lat, t, uncond)
                eps_c  = self.unet(lat, t, cond)
                return eps_c + guidance_scale * (eps_c - eps_uc)
        lat = torch.randn(
            len(prompts), self.hparams.latent_ch,
            self.hparams.sample_length // 8, device=device
        )
        self.sched_eval.set_timesteps(num_steps, device=device)
        for t in self.sched_eval.timesteps:
            lat = self.sched_eval.step(eps_fn(lat, t), t, lat).prev_sample
        wav = self.decoder(lat, target_len=self.hparams.sample_length)
        return wav.cpu() if to_cpu else wav
