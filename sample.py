import torch

from config import *
from data import *
from model import *

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
