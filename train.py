import argparse, pytorch_lightning as pl, torch
from model.pipeline import DiffusionVAEPipeline
from datamodule.laion import LAIONAudioDataModule
import torchaudio, os
from pathlib import Path
import glob

def inference(model: DiffusionVAEPipeline, out_dir="samples"):
    model.eval().cuda() if torch.cuda.is_available() else model.cpu()
    wav = model.generate(["A melancholic piano solo"], num_steps=50).squeeze(0)
    Path(out_dir).mkdir(exist_ok=True, parents=True)
    path = Path(out_dir) / "demo.wav"
    torchaudio.save(str(path), wav, 16_000)
    print(f"Sample saved to {path.resolve()}")

def latest_ckpt(ckpt_dir: str) -> str | None:
    ckpts = glob.glob(os.path.join(ckpt_dir, "*.ckpt"))
    return max(ckpts, key=os.path.getmtime) if ckpts else None

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--gpus", type=int, default=1)
    p.add_argument("--segment_ms", type=int, default=10_000)
    p.add_argument("--max_rows",    type=int, default=None,
                   help="for debugging, set to None for full dataset")
    p.add_argument("--data_files", type=str, default=None,
                   help="for debugging, specify a file or list of files to use")
    args = p.parse_args()

    if args.data_files is not None:
        print(f"IMPORTANT: Using data_files={args.data_files} will disable streaming mode.")

    dm = LAIONAudioDataModule(
        batch_size=args.batch_size,
        segment_ms=args.segment_ms,
        max_rows=args.max_rows,
        data_files=args.data_files
    )

    sample_len = args.segment_ms * 16  # 16 kHz
    model = DiffusionVAEPipeline(latent_ch=32, sample_length=sample_len)

    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=args.gpus,
        max_epochs=args.epochs,
        precision=32,
        log_every_n_steps=10,
        val_check_interval=1.0,
        callbacks=[
            pl.callbacks.ModelCheckpoint(
                dirpath="checkpoints",
                filename="model-{epoch:02d}-{val_FAD:.4f}",
                monitor="val_FAD",
                mode="min",
                save_top_k=3,
                save_last=True,
                every_n_epochs=1,
            )
        ]
    )

    try:
        trainer.fit(model, dm)
        inference(model)
    except KeyboardInterrupt:
        print("\nCaught KeyboardInterrupt! Loading last checkpoint...")
        last = latest_ckpt("checkpoints")
        if last is None:
            print("No checkpoint found, aborting.")
            exit(1)
        print(f"Resuming from {last}")
        model = DiffusionVAEPipeline.load_from_checkpoint(last)
        inference(model)
