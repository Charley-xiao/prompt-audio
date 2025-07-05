import torch
import pytorch_lightning as pl
import torch.nn.functional as F
from model.backbone import AudioEncoder, AudioDecoder, PromptEncoder, CondUNet, PromptEncoderv2
from diffusers import DDPMScheduler, DDIMScheduler
from torcheval.metrics import FrechetAudioDistance
from torchmetrics.aggregation import MeanMetric
from model.clap_module import CLAPAudioEmbedding


class DiffusionVAEPipeline(pl.LightningModule):
    def __init__(
        self,
        latent_ch=64,
        lr=2e-4,
        beta_kl=0.001,
        beta_rec=10.0,
        sample_length=16000,
        noise_steps=1000,
        n_val_epochs=1,
        scheduler_type="ddpm",
    ):
        super().__init__()
        self.save_hyperparameters()

        self.encoder = AudioEncoder(latent_ch)
        self.decoder = AudioDecoder(latent_ch, target_len=sample_length)
        self.textenc = PromptEncoderv2(proj_dim=128, preset="mini", trainable=False)
        latent_steps = sample_length // 8
        self.unet = CondUNet(latent_ch, cond_dim=128, latent_steps=latent_steps)
        if scheduler_type == "ddpm":
            self.scheduler = DDPMScheduler(num_train_timesteps=noise_steps)
        elif scheduler_type == "ddim":
            self.scheduler = DDIMScheduler(
                num_train_timesteps=noise_steps,
                beta_schedule="squaredcos_cap_v2",
                clip_sample=False,
            )

        self.fad = FrechetAudioDistance.with_vggish(device=self.device)
        self.clap = CLAPAudioEmbedding(device=self.device)
        self.clap_sim = MeanMetric()
        self.val_interval = n_val_epochs

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
            patience=2,
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

        bsz = z.size(0)
        t = torch.randint(
            0, self.scheduler.config.num_train_timesteps,
            (bsz,), device=self.device, dtype=torch.long
        )
        noise  = torch.randn_like(z)
        noisy_z = self.scheduler.add_noise(z, noise, t)

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
        wav_gt, prompt = batch
        wav_gen = self.generate(prompt, num_steps=250)
        self.fad.update(wav_gen.squeeze(1), wav_gt.squeeze(1))

        a_emb = self.clap(wav_gen)
        t_emb = self.clap.text_embed(prompt)
        sim = F.cosine_similarity(a_emb, t_emb)
        self.clap_sim.update(sim)

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
    def generate(self, prompts: list[str], num_steps: int = 50):
        def _scheduler_to_device(sched, device):
            for k, v in sched.__dict__.items():
                if torch.is_tensor(v):
                    setattr(sched, k, v.to(device))
                elif isinstance(v, dict):
                    for kk, vv in v.items():
                        if torch.is_tensor(vv):
                            v[kk] = vv.to(device)
        self.eval()
        device = self.device
        _scheduler_to_device(self.scheduler, device)
        cond = self.textenc(prompts, device=device)
        lat = torch.randn(
            cond.size(0),
            self.hparams.latent_ch,
            self.hparams.sample_length // 8,
            device=device,
        )
        self.scheduler.set_timesteps(num_steps, device=device)
        for t in self.scheduler.timesteps:
            noise_pred = self.unet(lat, t, cond)
            lat = self.scheduler.step(noise_pred, t, lat).prev_sample
        wav = self.decoder(lat)
        return wav.cpu()
