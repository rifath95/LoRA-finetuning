import time
import os
import sys
import torch

from config import *
from data import *
from model import *

if not os.path.exists("model_pretrained.pth"):
    print("model_pretrained.pth not found.")
    print("Please run `python train.py` first. That does pretraining and creates model_pretrained.pth.")
    print("Then run `python sample.py` to generate text using the trained weights.")
    sys.exit(1)

model = MyGPT()
model.load_state_dict(torch.load("model_pretrained.pth", map_location=device))
model = model.to(device)
model.eval()


prompt = 'Why are the citizens angry?'
context = torch.tensor(encode(prompt), dtype=torch.long,
                       device=device).unsqueeze(0)


start_time = time.perf_counter()
# generate from the model using KV cache
with torch.no_grad():
      print(decode(model.generate(context, max_new_tokens,
            temperature=0.9, top_k=50)[0].tolist())) 
time_taken = time.perf_counter()-start_time
print(f'Time taken to generate:{time_taken:.4f} seconds.')