import torch, torch.nn.functional as F, pytorch_lightning as pl
from model.backbone import AudioEncoder, AudioDecoder, PromptEncoder, CondUNet
from torcheval.metrics import FrechetAudioDistance
from torchmetrics.aggregation import MeanMetric
from model.clap_module import CLAPAudioEmbedding


class FlowScheduler:
    def __init__(self, num_train_timesteps=1000):
        self.num_steps = num_train_timesteps
        self.timesteps = torch.linspace(0.0, 1.0, num_train_timesteps + 1)

    def latent_prior(self, shape, device):
        return torch.randn(shape, device=device)

    def reverse_flow(self, unet, lat, cond, device):
        dt = -1.0 / self.num_steps
        for step in reversed(range(self.num_steps)):
            t = self.timesteps[step + 1].to(device)
            t_prev = t + dt
            v1 = unet(lat, t.expand(lat.size(0)), cond)
            lat_euler = lat + dt * v1
            v2 = unet(lat_euler, t_prev.expand(lat.size(0)), cond)
            lat = lat + dt * 0.5 * (v1 + v2)
        return lat


class FlowVAEPipeline(pl.LightningModule):
    def __init__(
        self,
        latent_ch=32,
        lr=2e-4,
        beta_kl=1e-3,
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
        self.unet = CondUNet(latent_ch, cond_dim=128, latent_steps=latent_steps)

        self.flow = FlowScheduler(num_train_timesteps=noise_steps)

        self.fad = FrechetAudioDistance.with_vggish(device=self.device)
        self.clap = CLAPAudioEmbedding(device=self.device)
        self.clap_sim = MeanMetric()
        self.val_interval = n_val_epochs

    @staticmethod
    def reparameterize(mu, logvar):
        std, eps = (0.5 * logvar).exp(), torch.randn_like(mu)
        return mu + std * eps

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="min", factor=0.5, patience=2, min_lr=1e-6
        )
        return {"optimizer": opt,
                "lr_scheduler": {"scheduler": sched,
                                 "monitor": "val_FAD",
                                 "interval": "epoch"}}

    def encode_latent(self, wav, prompt):
        mu, logvar = self.encoder(wav)
        z = self.reparameterize(mu, logvar)
        prompt_e = self.textenc(prompt, device=self.device)
        return z, mu, logvar, prompt_e

    def training_step(self, batch, _):
        wav, prompt = batch
        z0, mu, logvar, cond = self.encode_latent(wav, prompt)

        eps = torch.randn_like(z0)
        t   = torch.rand(z0.size(0), device=self.device)
        xt  = (1.0 - t.view(-1, 1, 1)) * z0 + t.view(-1, 1, 1) * eps
        v_target = eps - z0
        v_pred = self.unet(xt, t, cond)
        loss_fm = F.mse_loss(v_pred, v_target)

        kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        rec = F.l1_loss(self.decoder(z0), wav)

        loss = loss_fm + self.hparams.beta_kl * kld + self.hparams.beta_rec * rec
        self.log_dict({"loss": loss, "L_flow": loss_fm,
                       "L_KL": kld, "L_rec": rec},
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
        self.log("val_FAD", self.fad.compute(),
                 prog_bar=True, sync_dist=True)
        self.fad.reset()
        self.log("val_CLAPSim", self.clap_sim.compute(),
                 prog_bar=True, sync_dist=True)
        self.clap_sim.reset()

    @torch.no_grad()
    def generate(self, prompts: list[str], num_steps: int | None = 100):
        self.eval()
        device = self.device
        cond = self.textenc(prompts, device=device)
        lat = self.flow.latent_prior(
            (len(prompts), self.hparams.latent_ch,
             self.hparams.sample_length // 8),
            device=device,
        )
        if num_steps is not None:
            self.flow.num_steps = num_steps
            self.flow.timesteps = torch.linspace(
                0.0, 1.0, num_steps + 1, device=device)
        lat = self.flow.reverse_flow(self.unet, lat, cond, device)
        wav = self.decoder(lat)
        return wav.cpu()
