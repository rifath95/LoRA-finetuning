# LoRA Finetuning from Scratch (Under construction)

A from-scratch implementation of parameter-efficient finetuning for a GPT-style language model in PyTorch.

This project first **pretrains** a decoder-only language model on Tiny Shakespeare, then **finetunes** it with **LoRA (Low-Rank Adaptation)** on a separate instruction-style dataset. The goal is to show how a pretrained model can be adapted efficiently by updating only a small number of parameters instead of retraining the full network.

---

## Features

- GPT-style decoder-only language model in PyTorch
- Pretraining on character-level text (`tinyShakespeare.txt`)
- LoRA-based finetuning with frozen base weights
- Latent attention architecture with:
  - grouped latent key/value pathway
  - RoPE applied to a subset of attention dimensions
  - NoPE/RoPE decomposition
- Mixture-of-Experts (MoE) architecture in the pretrained model
- Gradient accumulation for larger effective batch sizes
- Prompt/response masking for supervised finetuning
- Side-by-side sampling from:
  - pretrained model
  - LoRA-finetuned model

---

## Repository Structure

    .
    ├── config.py               # Hyperparameters and device setup
    ├── data.py                 # Pretraining dataset loading and batching
    ├── data_finetuning.py      # Finetuning dataset loading, padding, and loss masking
    ├── model.py                # Base pretrained GPT-style model
    ├── lora_model.py           # LoRA-wrapped finetuning model
    ├── train.py                # Pretraining script
    ├── train_finetuning.py     # LoRA finetuning script
    ├── sample.py               # Compare pretrained vs finetuned generations
    ├── tinyShakespeare.txt     # Pretraining text corpus
    ├── finetuning_dataset.txt  # Finetuning prompt/response dataset
    ├── README.md

---

## Training Pipeline

### 1. Pretraining

`train.py` trains the base language model on `tinyShakespeare.txt` and saves:

    model_pretrained.pth

The pretrained model uses:
- latent attention
- partial RoPE
- Mixture-of-Experts
- load balancing loss during training

### 2. LoRA Finetuning

`train_finetuning.py`:
- loads `model_pretrained.pth`
- copies pretrained weights into the LoRA model
- freezes all non-LoRA parameters
- finetunes only the LoRA parameters on `finetuning_dataset.txt`
- saves:

    model_finetuned.pth

---

## LoRA Setup

LoRA is applied to selected attention projections instead of updating the full model.

This keeps the number of trainable parameters small while preserving most of the pretrained model weights.

The finetuning workflow is:

1. initialize LoRA model
2. load pretrained weights into base layers
3. freeze base parameters
4. train only LoRA parameters
5. compare generations before and after finetuning

---

## Finetuning Data Format

The finetuning dataset is structured as prompt/response text separated by:

    :ANSWER:

During finetuning, loss is masked so that optimization is applied only to the response portion, not the prompt.

This makes the setup closer to supervised instruction tuning.

---

## Setup

Install dependencies:

    pip install torch matplotlib

---

## Pretrain the Base Model

    python train.py

This will:
- train the base model on Tiny Shakespeare
- save `model_pretrained.pth`
- show training loss and learning-rate plots

---

## Finetune with LoRA

    python train_finetuning.py

This will:
- load `model_pretrained.pth`
- finetune only the LoRA parameters
- save `model_finetuned.pth`
- show finetuning loss and learning-rate plots

---

## Sample from Both Models

    python sample.py

This script:
- loads the pretrained model
- loads the LoRA-finetuned model
- generates text from the same prompt
- prints both outputs for direct comparison

By default the prompt is:

    What is love?:ANSWER:

---

## Implemented Concepts

### Pretraining
The repository includes a full language-model pretraining stage before finetuning.

### LoRA
Finetuning is done by adding low-rank trainable adapters while freezing the pretrained base model.

### Parameter-Efficient Finetuning
Only a small subset of parameters is updated during finetuning, reducing memory and compute requirements.

### Loss Masking
The finetuning loader masks the prompt tokens so that only the answer tokens contribute to the training loss.

### Latent Attention
The architecture uses a latent key/value pathway for memory-efficient attention.

### RoPE / NoPE Split
Rotary position embeddings are applied only to part of the attention representation, while the remaining dimensions are left unrotated.

### Mixture-of-Experts
The pretrained model includes sparse expert routing with top-k expert selection.

---

## Notes

- Run `python train.py` before `python train_finetuning.py`
- Run finetuning before `python sample.py` if you want pretrained vs finetuned comparison
- This repository is focused on understanding the mechanics of modern LLM finetuning from first principles

---

## Future Improvements

- Add training plots and sample outputs to the README
- Add parameter count comparison between full finetuning and LoRA finetuning
- Add support for multiple finetuning prompts and evaluation examples
- Extend to token-level or subword tokenization
