import os
os.environ.setdefault("NCCL_ASYNC_ERROR_HANDLING", "1")
os.environ.setdefault("TORCH_ENABLE_MPS_FALLBACK", "1")
import argparse, pytorch_lightning as pl, torch
from lightning.pytorch.profilers import SimpleProfiler
from model.pipeline.diffusion import DiffusionVAEPipeline
from model.pipeline.fm import FlowVAEPipeline
from model.pipeline.vrfm import VRFMVAEPipeline
from datamodule.laion import LAIONAudioDataModule
import torchaudio
from pathlib import Path
import glob

torch.set_float32_matmul_precision('medium')
INFERENCE_PROMPT = "A melancholic piano melody plays, characterized by a slow tempo and a minor key. The recording quality suggests a home studio setup, with a slightly warm and intimate sound. The piece evokes feelings of wistful longing."

def inference(model: DiffusionVAEPipeline | FlowVAEPipeline | VRFMVAEPipeline, out_dir="samples"):
    model.eval().cuda() if torch.cuda.is_available() else model.cpu()
    wav = model.generate([INFERENCE_PROMPT], num_steps=500).squeeze(0)
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
    p.add_argument("--model", type=str, default="diffusion",
                   choices=["diffusion", "flow", "vrfm"],
                   help="Choose the model type: 'diffusion', 'flow' or 'vrfm'")
    p.add_argument("--ckpt_dir", type=str, default="checkpoints",
                   help="Directory to save model checkpoints")
    p.add_argument("--profile", action="store_true",
                   help="Enable profiling of the training process")
    p.add_argument("--cfg_drop_prob", type=float, default=0.1,
                   help="Classifier-free guidance drop probability, set to 0 for no CFG")
    args = p.parse_args()

    dm = LAIONAudioDataModule(
        batch_size=args.batch_size,
        segment_ms=args.segment_ms,
        max_rows=args.max_rows,
        data_files=args.data_files
    )

    sample_len = args.segment_ms * 16  # 16 kHz
    if args.model == "diffusion":
        model = DiffusionVAEPipeline(latent_ch=96, sample_length=sample_len, cfg_drop_prob=args.cfg_drop_prob)
    elif args.model == "flow":
        model = FlowVAEPipeline(latent_ch=32, sample_length=sample_len)
    elif args.model == "vrfm":
        model = VRFMVAEPipeline(latent_ch=32, sample_length=sample_len)

    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=args.gpus,
        strategy="ddp",
        sync_batchnorm=True,
        max_epochs=args.epochs,
        precision="bf16-mixed",
        log_every_n_steps=10,
        val_check_interval=1.0,
        callbacks=[
            pl.callbacks.ModelCheckpoint(
                dirpath=f"{args.ckpt_dir}/{args.model}",
                filename=f"model-{{epoch:02d}}-{{val_FAD:.4f}}",
                monitor="val_FAD",
                mode="min",
                save_top_k=3,
                save_last=True,
                every_n_epochs=1,
            )
        ],
        profiler=SimpleProfiler(
            dirpath="profile",
            filename=f"profiler-{args.model}.txt",
        ) if args.profile else None,
    )

    try:
        trainer.fit(model, dm)
        if trainer.is_global_zero:
            inference(model)
    except KeyboardInterrupt:
        print("\nCaught KeyboardInterrupt! Loading last checkpoint...")
        last = latest_ckpt(f"{args.ckpt_dir}/{args.model}")
        if last is None:
            print("No checkpoint found, aborting.")
            exit(1)
        print(f"Resuming from {last}")
        if args.model == "flow":
            model = FlowVAEPipeline.load_from_checkpoint(last)
        elif args.model == "vrfm":
            model = VRFMVAEPipeline.load_from_checkpoint(last)
        else:
            model = DiffusionVAEPipeline.load_from_checkpoint(last)
        inference(model)
