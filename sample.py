import time
import os
import sys
import torch

from config import *
from data import encode, decode
from model import MyGPT
from lora_model import MyGPTwithLoRA

prompt = 'Tell me who Caius Marcius is.'
context = torch.tensor(encode(prompt+":ANSWER:"), dtype=torch.long,
                       device=device).unsqueeze(0)


if not os.path.exists("model_pretrained.pth"):
    print("model_pretrained.pth not found.")
    print("Please run `python train.py` first. That does pretraining and creates model_pretrained.pth.")
    print("Then run `python sample.py` to generate text using the trained weights.")
    sys.exit(1)

model_pretrained = MyGPT()
model_pretrained.load_state_dict(torch.load("model_pretrained.pth", map_location=device))
model_pretrained = model_pretrained.to(device)
model_pretrained.eval()


# generate from the model_pretrained using KV cache
with torch.no_grad():
      print(f'Pretrained model output: {decode(model_pretrained.generate(context, max_new_tokens,
            temperature=0.9, top_k=50)[0].tolist())}') 


# Generating with Finetuned model
if not os.path.exists("model_finetuned.pth"):
    print("model_finetuned.pth not found.")
    print("Please run `python train_finetunig.py` first. That does finetuning and creates model_finetuned.pth.")
    print("Then run `python sample.py` to generate text using the trained weights.")
    sys.exit(1)

model_finetuned = MyGPTwithLoRA()
model_finetuned.load_state_dict(torch.load("model_finetuned.pth", map_location=device))
model_finetuned = model_finetuned.to(device)
model_finetuned.eval()

# generate from the model_finetuned using KV cache
with torch.no_grad():
      print(f'Finetuned model output: {decode(model_finetuned.generate(context, max_new_tokens,
            temperature=0.9, top_k=50)[0].tolist())}') 