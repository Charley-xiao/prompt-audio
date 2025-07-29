import os
import argparse, pytorch_lightning as pl, torch
from lightning.pytorch.profilers import SimpleProfiler
from model.pipeline.diffusion import DiffusionVAEPipeline
from model.pipeline.fm import FlowMatchingPipeline
from datamodule.laion import LAIONAudioDataModule
import torchaudio
from pathlib import Path
import glob

torch.set_float32_matmul_precision('highest')
INFERENCE_PROMPT = "A melancholic piano melody plays, characterized by a slow tempo and a minor key. The recording quality suggests a home studio setup, with a slightly warm and intimate sound. The piece evokes feelings of wistful longing."

def inference(model: DiffusionVAEPipeline, out_dir="samples"):
    model.eval().cuda() if torch.cuda.is_available() else model.cpu()
    wav = model.generate([INFERENCE_PROMPT], num_steps=100).squeeze(0)
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
                   choices=["diffusion", "flow"],
                   help="Choose the model type: 'diffusion', 'flow' or 'vrfm'")
    p.add_argument("--ckpt_dir", type=str, default="checkpoints",
                   help="Directory to save model checkpoints")
    p.add_argument("--profile", action="store_true",
                   help="Enable profiling of the training process")
    p.add_argument("--cfg_drop_prob", type=float, default=0.1,
                   help="p_uncond in CFG, set to 0 for no CFG")
    p.add_argument("--no_save_ckpt", action="store_true")
    p.add_argument("--disable_text_enc", action="store_true",
                   help="Disable text encoder, useful for debugging")
    p.add_argument("--resume_from", type=str, default=None,
                   help="Path to a checkpoint to resume training from")
    args = p.parse_args()

    dm = LAIONAudioDataModule(
        batch_size=args.batch_size,
        segment_ms=args.segment_ms,
        max_rows=args.max_rows,
        data_files=args.data_files
    )

    sample_len = args.segment_ms * 16  # 16 kHz
    if args.model == "diffusion":
        model = DiffusionVAEPipeline(
            latent_ch=96, 
            sample_length=sample_len, 
            cfg_drop_prob=args.cfg_drop_prob,
            disable_text_enc=args.disable_text_enc
        )
    elif args.model == "flow":
        model = FlowMatchingPipeline(
            latent_ch=96, 
            sample_length=sample_len, 
            cfg_drop_prob=args.cfg_drop_prob,
            disable_text_enc=args.disable_text_enc
        )
    else:
        raise NotImplementedError(f"Model {args.model} is not implemented yet.")

    os.makedirs("samples/", exist_ok=True)
    if args.resume_from:
        print(f"Resuming from {args.resume_from}")
        model = model.load_from_checkpoint(args.resume_from)
    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=args.gpus,
        strategy="ddp",
        sync_batchnorm=True,
        max_epochs=args.epochs,
        precision="32",
        log_every_n_steps=10,
        val_check_interval=1.0,
        enable_checkpointing=not args.no_save_ckpt,
        callbacks=[
            pl.callbacks.ModelCheckpoint(
                dirpath=f"{args.ckpt_dir}/{args.model}",
                filename=f"model-{{epoch:02d}}-{{val_FAD:.4f}}",
                monitor="val_FAD",
                mode="min",
                save_top_k=1,
                save_last=False,
                every_n_epochs=1,
            )
        ] if not args.no_save_ckpt else None,
        profiler=SimpleProfiler(
            dirpath="profile",
            filename=f"profiler-{args.model}.txt",
        ) if args.profile else None,
    )

    trainer.fit(model, dm)
    if trainer.is_global_zero:
        model = model.load_from_checkpoint(
            latest_ckpt(f"{args.ckpt_dir}/{args.model}")
        ) if not args.no_save_ckpt else model
        inference(model)