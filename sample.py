import torch

from config import *
from data import *
from model import *

model = MyGPT()
model = model.to(device)


prompt = 'Hello'
context = torch.tensor(encode(prompt), dtype=torch.long,
                       device=device).unsqueeze(0)


# generate from the model using KV cache
print(decode(model.generate(context, max_new_tokens,
      temperature=0.9, top_k=50)[0].tolist()))
