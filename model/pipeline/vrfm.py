import torch, torch.nn as nn, torch.nn.functional as F
import pytorch_lightning as pl
from model.backbone import AudioEncoder, AudioDecoder, PromptEncoder, CondUNet
from model.clap_module import CLAPAudioEmbedding
from torcheval.metrics import FrechetAudioDistance
from torchmetrics.aggregation import MeanMetric
from model.pipeline.fm import FlowScheduler


class PosteriorEncoder(nn.Module):
    def __init__(self, latent_ch: int):
        super().__init__()
        in_ch = latent_ch * 3 + 1
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, 256, 1), nn.GELU(),
            nn.Conv1d(256, 256, 1),   nn.GELU(),
            nn.Conv1d(256, latent_ch * 2, 1),
        )

    def forward(self, x0, x1, xt, t):
        t_embed = t.view(-1, 1, 1).expand(-1, 1, xt.size(-1))
        cat = torch.cat([x0, x1, xt, t_embed], dim=1)
        out = self.net(cat)
        mu, logvar = out.chunk(2, dim=1)
        return mu, logvar


class VRFMVAEPipeline(pl.LightningModule):
    """Reference: https://arxiv.org/pdf/2502.09616"""
    def __init__(
        self,
        latent_ch=32,
        lr=2e-4,
        beta_kl_latent=1e-3,
        beta_kl_vae=1e-3,
        beta_rec=10.0,
        sample_length=16000,
        noise_steps=250,
        n_val_epochs=1,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.encoder = AudioEncoder(latent_ch)
        self.decoder = AudioDecoder(latent_ch, target_len=sample_length)
        self.textenc = PromptEncoder(proj_dim=128)
        latent_steps = sample_length // 8
        self.unet = CondUNet(
            latent_ch, cond_dim=128, latent_steps=latent_steps,
            extra_cond_dim=latent_ch,
        )
        self.posterior = PosteriorEncoder(latent_ch)
        self.flow = FlowScheduler(num_train_timesteps=noise_steps)

        self.fad  = FrechetAudioDistance.with_vggish(device=self.device)
        self.clap = CLAPAudioEmbedding(device=self.device)
        self.clap_sim = MeanMetric()
        self.val_interval = n_val_epochs

        self.z_proj = nn.Conv1d(latent_ch, 128, 1)

    @staticmethod
    def reparameterize(mu, logvar):
        std, eps = (0.5 * logvar).exp(), torch.randn_like(mu)
        return mu + std * eps

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="min", patience=2, factor=0.5, min_lr=1e-6
        )
        return {"optimizer": opt,
                "lr_scheduler": {"scheduler": sched,
                                 "monitor": "val_FAD"}}

    def encode_latent(self, wav, prompt):
        mu, logvar = self.encoder(wav)
        z0 = self.reparameterize(mu, logvar)
        prompt_e = self.textenc(prompt, device=self.device)
        return z0, mu, logvar, prompt_e

    def training_step(self, batch, _):
        wav, prompt = batch
        device = self.device

        z0, mu0, logvar0, prompt_e = self.encode_latent(wav, prompt)

        eps = torch.randn_like(z0)
        t   = torch.rand(z0.size(0), device=device)
        xt  = (1.0 - t.view(-1,1,1)) * z0 + t.view(-1,1,1) * eps
        v_target = eps - z0

        mu_z, logvar_z = self.posterior(z0, eps, xt, t)
        z_lat = self.reparameterize(mu_z, logvar_z)

        z_cond = self.z_proj(z_lat).mean(dim=-1)
        cond_all = prompt_e + z_cond
        v_pred = self.unet(xt, t, cond_all, z_lat)

        loss_flow = F.mse_loss(v_pred, v_target)

        kl_vae = -0.5 * torch.mean(1 + logvar0 - mu0.pow(2) - logvar0.exp())
        kl_lat = -0.5 * torch.mean(1 + logvar_z - mu_z.pow(2) - logvar_z.exp())
        rec    = F.l1_loss(self.decoder(z0), wav)

        loss = (loss_flow +
                self.hparams.beta_kl_vae * kl_vae +
                self.hparams.beta_kl_latent * kl_lat +
                self.hparams.beta_rec * rec)

        self.log_dict({"loss": loss, "L_flow": loss_flow,
                       "KL_vae": kl_vae, "KL_z": kl_lat, "L_rec": rec},
                      prog_bar=True)
        return loss

    def validation_step(self, batch, _):
        wav_gt, prompts = batch
        wav_gen = self.generate(prompts, num_steps=100)
        self.fad.update(wav_gen.squeeze(1), wav_gt.squeeze(1))
        a_emb = self.clap(wav_gen)
        t_emb = self.clap.text_embed(prompts)
        self.clap_sim.update(F.cosine_similarity(a_emb, t_emb))

    def on_validation_epoch_end(self):
        if (self.current_epoch + 1) % self.val_interval:
            return
        self.log("val_FAD", self.fad.compute(), prog_bar=True, sync_dist=True)
        self.fad.reset()
        self.log("val_CLAPSim", self.clap_sim.compute(), prog_bar=True, sync_dist=True)
        self.clap_sim.reset()

    @torch.no_grad()
    def generate(self, prompts: list[str], num_steps: int = 250):
        self.eval()
        device = self.device

        prompt_e = self.textenc(prompts, device=device)
        z_lat = torch.randn(len(prompts), self.hparams.latent_ch,
                            self.hparams.sample_length // 8, device=device)
        z_cond = self.z_proj(z_lat).mean(dim=-1)
        cond_all = prompt_e + z_cond

        lat = self.flow.latent_prior(
            (len(prompts), self.hparams.latent_ch,
             self.hparams.sample_length // 8), device)

        if num_steps != self.flow.num_steps:
            self.flow.set_steps(num_steps)
        lat = self.flow.reverse_flow(self.unet, lat, cond_all, z_lat, device)
        return self.decoder(lat).cpu()
