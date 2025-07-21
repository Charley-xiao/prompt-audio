import random, torch, torchaudio
import torch.nn.functional as F
from datasets import load_dataset, Audio
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only


class LAIONClipDataset(Dataset):
    def __init__(self, hf_ds, segment_len, sample_rate):
        self.ds = hf_ds
        self.seglen = segment_len
        self.sr = sample_rate

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        ex = self.ds[idx]
        wav = torch.from_numpy(ex["audio.mp3"]["array"]).float()
        if wav.ndim == 1:
            wav = wav[None, :]
        L = wav.shape[-1]
        if L >= self.seglen:
            start = random.randint(0, L - self.seglen)
            wav = wav[:, start : start + self.seglen]
        else:
            pad = self.seglen - L
            wav = F.pad(wav, (0, pad))
        caption = ex["metadata.json"]["caption"]
        return wav, caption


class LAIONAudioDataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int = 32,
        num_workers: int = 4,
        sample_rate: int = 16_000,
        segment_ms: int = 1_000,
        val_pct: float = 0.00005,
        laion_path: str = "laion/LAION-Audio-300M",
        split: str = "train",
        max_rows: int | None = None,
        data_files: str | list[str] | None = None,
    ):
        super().__init__()
        self.bs, self.nw = batch_size, num_workers
        self.sr = sample_rate
        self.seglen = sample_rate * segment_ms // 1000
        self.val_pct = val_pct
        self.hf_kwargs = dict(
            path = laion_path,
            split = split,
            streaming = False,
            data_files = data_files,
        )
        self.max_rows = max_rows

    @staticmethod
    def _collate(batch):
        wavs, caps = zip(*batch)
        return torch.stack(wavs), list(caps)
    
    @rank_zero_only
    def prepare_data(self):
        load_dataset(**self.hf_kwargs)

    def setup(self, stage: str | None = None):
        if stage not in (None, "fit"):
            return

        ds = load_dataset(**self.hf_kwargs)
        ds = ds.cast_column("audio.mp3", Audio(sampling_rate=self.sr))
        if self.max_rows is not None:
            ds = ds.select(range(self.max_rows))

        split_ds = ds.train_test_split(test_size=self.val_pct, seed=42, shuffle=True)
        train_hf, val_hf = split_ds["train"], split_ds["test"]

        self.train_ds = LAIONClipDataset(train_hf, self.seglen, self.sr)
        self.val_ds = LAIONClipDataset(val_hf,   self.seglen, self.sr)

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.bs,
            shuffle=False,
            num_workers=self.nw,
            pin_memory=True,
            persistent_workers=self.nw > 0,
            drop_last=True,
            collate_fn=self._collate,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.bs,
            shuffle=False,
            num_workers=self.nw,
            pin_memory=True,
            persistent_workers=self.nw > 0,
            collate_fn=self._collate,
        )
