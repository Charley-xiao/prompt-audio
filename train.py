import argparse, pytorch_lightning as pl, torch
from model.pipeline import DiffusionVAEPipeline
from datamodule.laion import LAIONAudioDataModule
import torchaudio, os

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--gpus", type=int, default=1)
    p.add_argument("--segment_ms", type=int, default=10_000)
    p.add_argument("--max_rows",    type=int, default=None,
                   help="for debugging, set to None for full dataset")
    args = p.parse_args()

    dm = LAIONAudioDataModule(
        batch_size=args.batch_size,
        segment_ms=args.segment_ms,
        max_rows=args.max_rows,
    )

    sample_len = args.segment_ms * 16  # 16 kHz
    model = DiffusionVAEPipeline(sample_length=sample_len)

    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=args.gpus,
        max_epochs=args.epochs,
        precision=32,
        log_every_n_steps=10,
    )

    trainer.fit(model, dm)

    wav = model.generate(["A melancholic piano solo"], num_steps=50)
    os.makedirs("samples", exist_ok=True)
    torchaudio.save("samples/demo.wav", wav, 16_000)
    print("Sample saved to samples/demo.wav")
