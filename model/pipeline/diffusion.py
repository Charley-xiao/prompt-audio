import torch
import pytorch_lightning as pl
import torch.nn.functional as F
from model.backbone import AudioEncoder, AudioDecoder, PromptEncoder, CondUNet, PromptEncoderv2
from diffusers import DDPMScheduler, DDIMScheduler
from torcheval.metrics import FrechetAudioDistance
from torchmetrics.aggregation import MeanMetric
from model.clap_module import CLAPAudioEmbedding
import contextlib, time
from pytorch_lightning.utilities import rank_zero_only
from functools import lru_cache
from matplotlib import pyplot as plt


@contextlib.contextmanager
def _timer(name, times_dict):
    start = time.time()
    yield
    times_dict[name] = time.time() - start


class DiffusionVAEPipeline(pl.LightningModule):
    def __init__(
        self,
        latent_ch=64,
        lr=2e-4,
        beta_kl=0.01,
        beta_rec=1.0,
        sample_length=16000,
        noise_steps=1000,
        n_val_epochs=1,
        cfg_drop_prob=0.1,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.encoder = AudioEncoder(latent_ch)
        self.decoder = AudioDecoder(latent_ch, target_len=sample_length)
        self.textenc = PromptEncoderv2(proj_dim=128, preset="mini", trainable=False)
        latent_steps = sample_length // 8
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

    def setup(self, stage=None):
        self.scheduler_to_device(self.sched_train, self.device)
        self.scheduler_to_device(self.sched_eval, self.device)
        torch.backends.cudnn.benchmark = True
        if self.fad is None:
            self.fad  = FrechetAudioDistance.with_vggish(device="cpu").to(self.device)
            self.clap = CLAPAudioEmbedding(device=self.device)

    @staticmethod
    def reparameterize(mu, logvar):
        std, eps = (0.5 * logvar).exp(), torch.randn_like(mu)
        return mu + std * eps

    def configure_optimizers(self):
        # return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
        opt = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt,
            mode="min",
            factor=0.5,
            patience=4,
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
        prompt_e = self.textenc(prompt, device=self.device)
        return z, mu, logvar, prompt_e

    def training_step(self, batch, _):
        wav, prompt = batch
        z, mu, logvar, prompt_e = self(wav, prompt)
        if torch.rand(1, device=self.device) < self.cfg_drop_prob:
            prompt_e = torch.zeros_like(prompt_e)
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
        rec = F.l1_loss(self.decoder(z), wav)
        loss = loss_diff + self.hparams.beta_kl * kld + self.hparams.beta_rec * rec
        self.log_dict(
            {"loss": loss, "L_diff": loss_diff, "L_KL": kld, "L_rec": rec},
            prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        wav_gt, prompts = batch
        times = {}
        with _timer("gen", times):
            wav_gen = self.generate(prompts, num_steps=100, to_cpu=False)
        with _timer("fad", times):
            self.fad.update(wav_gen.squeeze(1), wav_gt.squeeze(1))
        with _timer("clap", times):
            a_emb = self.clap(wav_gen)
            t_emb = self.clap.text_embed(prompts)
            sim   = F.cosine_similarity(a_emb, t_emb)
            self.clap_sim.update(sim)
        if batch_idx == 0:
            self._log_times(times)
            for i in range(min(3, wav_gen.size(0))):
                # Create two subplots: one for generated and one for ground truth
                fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
                axs[0].specgram(wav_gen[i].cpu().numpy(), NFFT=256, Fs=16000, Fc=0, noverlap=128)
                axs[0].set_title(f"Generated Audio {i+1}")
                axs[1].specgram(wav_gt[i].cpu().numpy(), NFFT=256, Fs=16000, Fc=0, noverlap=128)
                axs[1].set_title(f"Ground Truth Audio {i+1}")
                plt.tight_layout()
                plt.savefig(f"samples/audio_comp_{self.current_epoch}_{batch_idx}_{i}.png")
                plt.close(fig)

    @rank_zero_only
    def _log_times(self, times):
        print(
            f"┌─ Validation timings\n"
            f"│  Generation : {times['gen'] :.2f} s\n"
            f"│  FAD update : {times['fad'] :.2f} s\n"
            f"│  CLAP sim   : {times['clap']:.2f} s\n"
            f"└─────────────────────"
        )

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
        return self.textenc([txt], device=self.device)

    @torch.no_grad()
    def generate(
        self, 
        prompts: list[str], 
        num_steps: int = 50, 
        to_cpu: bool = True,
        guidance_scale: float = 3.0,
    ):
        self.eval()
        device = self.device
        cond = torch.cat([self._cached_text(t) for t in prompts], dim=0)
        uncond = torch.zeros_like(cond)
        lat = torch.randn(
            cond.size(0),
            self.hparams.latent_ch,
            self.hparams.sample_length // 8,
            device=device,
        )
        self.sched_eval.set_timesteps(num_steps, device=device)
        for t in self.sched_eval.timesteps:
            eps_uc = self.unet(lat, t, uncond)
            eps_c  = self.unet(lat, t, cond)
            eps = eps_uc + guidance_scale * (eps_c - eps_uc)
            lat = self.sched_eval.step(eps, t, lat).prev_sample
        wav = self.decoder(lat)
        return wav.cpu() if to_cpu else wav
