export CUDA_VISIBLE_DEVICES=5
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/ext_disk/xqw
export TOKENIZERS_PARALLELISM=false
huggingface-cli download --repo-type dataset laion/LAION-Audio-300M --include flash_15_2_random_snippets_0.tar
python train.py --max_rows 10000 --batch_size 12 --segment_ms 2000 --data_files flash_15_2_random_snippets_0.tar --epochs 50 --model diffusion --ckpt_dir /ext_disk/xqw/checkpoints
tensorboard --logdir=lightning_logs/