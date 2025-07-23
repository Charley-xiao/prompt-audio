export CUDA_VISIBLE_DEVICES=0
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/workspace
export TOKENIZERS_PARALLELISM=false
huggingface-cli download --repo-type dataset laion/LAION-Audio-300M --include flash_15_2_random_snippets_0.tar
python train.py --max_rows 10000 \
    --batch_size 32 \
    --segment_ms 2000 \
    --epochs 100 \
    --model diffusion \
    --ckpt_dir /workspace \
    --data_files flash_15_2_random_snippets_0.tar
tensorboard --logdir=lightning_logs/