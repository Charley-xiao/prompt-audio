import torch
from transformers import ClapModel, ClapProcessor


class CLAPAudioEmbedding:
    def __init__(self, model_name="laion/clap-htsat-fused", device="cpu"):
        self.processor = ClapProcessor.from_pretrained(model_name)
        self.model = ClapModel.from_pretrained(model_name).to(device)
        self.device = device

    @torch.no_grad()
    def __call__(self, wav_16khz: torch.Tensor) -> torch.Tensor:
        if wav_16khz.dim() == 3:
            wav_16khz = wav_16khz.squeeze(1)
        inputs = self.processor(
            audios=wav_16khz,
            sampling_rate=16_000,
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
