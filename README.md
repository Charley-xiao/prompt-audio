# Prompt-Audio

![](asset/overview.jpg)

## Installation

```bash
conda create -n prompt-audio python=3.12 -y
conda activate prompt-audio
pip install torch torchvision torchaudio pytorch-lightning datasets transformers diffusers sentencepiece librosa soundfile torcheval lightning xformers matplotlib
```

## Usage

### Training

See `train.sh`.