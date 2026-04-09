import torch
import torch.nn.functional as F

from config import *
from data import encode, decode

# Opening the dataset and text is a long string
with open('finetuning_dataset.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Convert text(a single long string) to a list with each element being prompt:ANSWER:respose:END:
text = text.split("\n")

# create mask to mask out the prompt
text_mask = []
for example in text:
    parts = example.split(":ANSWER:")
    example_mask = torch.cat((torch.zeros(len(parts[0])+8), torch.ones(len(parts[1]))), dim=0)
    text_mask.append(example_mask)

#encode text
data = [torch.tensor(encode(e), dtype=torch.long) for e in text]

# train vs val data split
n = int(0.9 * len(data))
train_data = data[:n]
train_mask = text_mask[:n]
val_data = data[n:]
val_mask = text_mask[n:]

# Data loading
def get_batch(split):
    data = train_data if split == 'train' else val_data
    data_mask = train_mask if split == 'train' else val_mask
    ix = torch.randint(len(data), (micro_batch_size_fn,))
    selected_data = [data[i] for i in ix]
    selected_mask = [data_mask[i] for i in ix]
    len_selected_data = [len(e) for e in selected_data]
    max_len = max(len_selected_data)+1
    for i in range(len(selected_data)):
        selected_data[i] = F.pad(selected_data[i],(0,max_len-len(selected_data[i])))
        selected_mask[i] = F.pad(selected_mask[i],(0,max_len-len(selected_mask[i])))
    selected_data = torch.stack([selected_data[i] for i in range(len(selected_data))])[:,:block_size+1]
    selected_mask = torch.stack([selected_mask[i] for i in range(len(selected_data))])[:,:block_size+1]

    x    = selected_data[:,:-1]
    y    = selected_data[:,1:]
    mask = selected_mask[:,1:]

    x, y, mask = x.to(device), y.to(device), mask.to(device)
    return x, y, mask

# # Checking if outputs are as intended: answer is yes
# x,y, mask = get_batch('train')
# print(x.shape, y.shape, mask.shape)
# print(decode(x[0].tolist()))
# print(decode(y[0].tolist()))
# masked_y = y * mask
# print(decode(masked_y[0].tolist()))