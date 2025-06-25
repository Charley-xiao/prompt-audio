import random, torch, torchaudio
import torch.nn.functional as F
from datasets import load_dataset, Audio
from torch.utils.data import IterableDataset, DataLoader
import pytorch_lightning as pl

class LAIONAudioIterable(IterableDataset):
    def __init__(
        self,
        split: str = "train",
        sample_rate: int = 16_000,
        segment_ms: int = 1_000,
        laion_path: str = "laion/LAION-Audio-300M",
        max_rows: int | None = None, # reserved for debugging
    ):
        super().__init__()
        self.seglen = sample_rate * segment_ms // 1000
        ds = load_dataset(
            laion_path, split=split, streaming=True
        )
        ds = ds.cast_column("audio.mp3", Audio(sampling_rate=sample_rate))
        self.ds = ds if max_rows is None else ds.take(max_rows)

    def __iter__(self):
        for ex in self.ds:
            wav = ex["audio.mp3"]["array"]
            if wav.ndim == 1:
                wav = wav[None, :]
            L = wav.shape[-1]
            if L >= self.seglen:
                start = random.randint(0, L - self.seglen)
                wav = wav[:, start : start + self.seglen]
            else:
                pad = self.seglen - L
                wav = F.pad(torch.from_numpy(wav), (0, pad)).numpy()
            caption = ex["metadata.json"]["caption"]
            yield torch.from_numpy(wav).float(), caption


class LAIONAudioDataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int = 32,
        num_workers: int = 4,
        **kwargs,
    ):
        super().__init__()
        self.bs, self.nw, self.kw = batch_size, num_workers, kwargs

    @staticmethod
    def _collate(b):
        wavs, caps = zip(*b)
        return torch.stack(wavs), list(caps)

    def train_dataloader(self):
        ds = LAIONAudioIterable(**self.kw)
        return DataLoader(
            ds, batch_size=self.bs, num_workers=self.nw,
            collate_fn=self._collate, pin_memory=True
        )
