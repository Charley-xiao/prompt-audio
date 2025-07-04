import torch
from transformers import ClapModel, ClapProcessor
import torchaudio


class CLAPAudioEmbedding:
    def __init__(self, model_name="laion/clap-htsat-fused", device="cpu"):
        self.processor = ClapProcessor.from_pretrained(model_name)
        self.model = ClapModel.from_pretrained(model_name).to(device)
        self.device = device
        self.target_sr = 48_000  # CLAP expects 48kHz audio

    @torch.no_grad()
    def __call__(self, wav16: torch.Tensor) -> torch.Tensor:
        if wav16.dim() == 3:
            wav16 = wav16.squeeze(1)
        if self.target_sr != 16_000:
            wav48 = torchaudio.functional.resample(
                wav16, orig_freq=16_000, new_freq=self.target_sr
            )
        else:
            wav48 = wav16
        inputs = self.processor(
            audios=wav48,
            sampling_rate=self.target_sr,
            return_tensors="pt",
            padding=True,
        ).to(self.device)
        emb = self.model.get_audio_features(**inputs)
        return torch.nn.functional.normalize(emb, dim=-1)

    @torch.no_grad()
    def text_embed(self, texts: list[str]) -> torch.Tensor:
        inputs = self.processor(
            text=texts,
            return_tensors="pt",
            padding=True,
        ).to(self.device)

        emb = self.model.get_text_features(**inputs)
        return torch.nn.functional.normalize(emb, dim=-1)
