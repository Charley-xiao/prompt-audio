import torch
import pytorch_lightning as pl
import torch.nn.functional as F
from model.backbone import AudioEncoder, AudioDecoder, PromptEncoder, CondUNet
from diffusers import DDPMScheduler


class DiffusionVAEPipeline(pl.LightningModule):
    def __init__(
        self,
        latent_ch=32,
        lr=2e-4,
        beta_kl=0.001,
        beta_rec=10.0,
        sample_length=16000,
        noise_steps=1000
    ):
        super().__init__()
        self.save_hyperparameters()

        self.encoder = AudioEncoder(latent_ch)
        self.decoder = AudioDecoder(latent_ch, target_len=sample_length)
        self.textenc = PromptEncoder(proj_dim=128)
        latent_steps = sample_length // 8
        self.unet = CondUNet(latent_ch, cond_dim=128, latent_steps=latent_steps)
        self.scheduler = DDPMScheduler(num_train_timesteps=noise_steps)

    @staticmethod
    def reparameterize(mu, logvar):
        std, eps = (0.5 * logvar).exp(), torch.randn_like(mu)
        return mu + std * eps

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)

    def forward(self, wav, prompt):
        mu, logvar = self.encoder(wav)
        z = self.reparameterize(mu, logvar)
        prompt_e = self.textenc(prompt, device=self.device)
        return z, mu, logvar, prompt_e

    def training_step(self, batch, _):
        wav, prompt = batch
        z, mu, logvar, prompt_e = self(wav, prompt)

        print(f"Batch size: {wav.size(0)}, Latent shape: {z.shape}, Prompt shape: {prompt_e.shape}")

        # Noise scheduling
        bsz = z.size(0)
        t = torch.randint(
            0, self.scheduler.num_train_timesteps,
            (bsz,), device=self.device, dtype=torch.long
        )
        print(f"Time steps: {t}, shape: {t.shape}")
        noise  = torch.randn_like(z)
        noisy_z = self.scheduler.add_noise(z, noise, t)
        print(f"Noisy latent shape: {noisy_z.shape}, Noise shape: {noise.shape}")

        # Predict noise
        noise_pred = self.unet(noisy_z, t, prompt_e)
        loss_diff  = F.mse_loss(noise_pred, noise)

        # VAE loss
        kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        rec = F.l1_loss(self.decoder(z), wav)

        loss = loss_diff + self.hparams.beta_kl * kld + self.hparams.beta_rec * rec
        self.log_dict(
            {"loss": loss, "L_diff": loss_diff, "L_KL": kld, "L_rec": rec},
            prog_bar=True)
        return loss

    @torch.no_grad()
    def generate(self, prompts: list[str], num_steps: int = 50):
        self.eval()
        cond = self.textenc(prompts, device=self.device)
        bsz  = cond.size(0)
        # Sample initial noise
        lat  = torch.randn(bsz, self.hparams.latent_ch,
                           self.hparams.sample_length // 8,
                           device=self.device)
        self.scheduler.set_timesteps(num_steps, device=self.device)
        for t in self.scheduler.timesteps:
            noise_pred = self.unet(lat, t, cond)
            lat = self.scheduler.step(noise_pred, t, lat).prev_sample
        wav = self.decoder(lat)
        return wav.cpu()
