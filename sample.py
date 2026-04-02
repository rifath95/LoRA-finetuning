import os
import sys
import torch

from config import *
from data import *
from model import *

if not os.path.exists("model.pth"):
    print("model.pth not found.")
    print("Please run `python train.py` first. That trains the model and creates model.pth.")
    print("Then run `python sample.py` to generate text using the trained weights.")
    sys.exit(1)

model = MyGPT()
model.load_state_dict(torch.load("model.pth", map_location=device))
model = model.to(device)
model.eval()


prompt = '\n'
context = torch.tensor(encode(prompt), dtype=torch.long,
                       device=device).unsqueeze(0)


# generate from the model using KV cache
with torch.no_grad():
      print(decode(model.generate(context, max_new_tokens,
            temperature=0.9, top_k=50)[0].tolist())) 
