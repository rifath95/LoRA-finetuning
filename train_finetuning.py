import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
import matplotlib.pyplot as plt

from config import *
from data_finetuning import *
from lora_model import MyGPTwithLoRA

# make sure that pre-trained weights are available
if not os.path.exists("model_pretrained.pth"):
    print("model_pretrained.pth not found.")
    print("Please run `python train.py` first. That does pretraining and creates model_pretrained.pth.")
    print("Then run `python train_finetuning.py` to do finetuning of the pretrained model.")
    sys.exit(1)


# Optional CUDA optimization
if torch.cuda.is_available():
    torch.set_float32_matmul_precision('high')

# Initiate LoRA applied Model
model = MyGPTwithLoRA()

# Get current state dict of LoRA Model
lora_state_dict = model.state_dict()

# Load pre-trained weights and copy them to lora_state_dict
pretrained_state_dict = torch.load("model_pretrained.pth", map_location="cpu")

for old_key, value in pretrained_state_dict.items():
    if old_key in lora_state_dict:
        if lora_state_dict[old_key].shape != value.shape:
            raise ValueError(f'Shape mismatch for {old_key}: {lora_state_dict[old_key].shape} vs {value.shape}')
        lora_state_dict[old_key] = value
    else:
        parts = old_key.split(".")
        new_key = ".".join(parts[:-1] + ["base", parts[-1]])
        if new_key in lora_state_dict:
            if lora_state_dict[new_key].shape != value.shape:
                raise ValueError(f'Shape mismatch for {new_key}: {lora_state_dict[new_key].shape} vs {value.shape}')
            lora_state_dict[new_key] = value
        else:
            new_key = old_key + ".base"
            if new_key in lora_state_dict:
                if lora_state_dict[new_key].shape != value.shape:
                    raise ValueError(f'Shape mismatch for {new_key}: {lora_state_dict[new_key].shape} vs {value.shape}')
                lora_state_dict[new_key] = value
            else:
                raise ValueError(f'{old_key} missing in LoRA Model')
        
# Load lora_state_dict to the LoRA Model
model.load_state_dict(lora_state_dict)

# Freeze all non-LoRA parameters and require grad only for LoRA parameters in LoRA Model
for name, param in model.named_parameters():
    if "lora" in name:
        param.requires_grad = True
    else:
        param.requires_grad = False
        

# move model to device
model = model.to(device)
model = torch.compile(model)  # [speedup] not much speedup in mps
parameter_size = sum(p.numel() for p in model.parameters())
frozen_param_size = sum(p.numel() for p in model.parameters() if not p.requires_grad)
trainable_param_size = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'{model_size} model on {device} with total parameters: {parameter_size*1e-6} M parameters = {frozen_param_size*1e-6} M frozen parameters + {trainable_param_size} finetuning parameters')


# Learning rate scheduler
max_lr = 6e-4
min_lr = max_lr * 0.1
max_steps = n_epoch_fn
warmup_steps = int(max_steps * 0.1)
def get_lr(it):
    if it < warmup_steps:
        return max_lr * (it+1)/warmup_steps
    elif it > max_steps:
        return min_lr
    else:
        decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return min_lr + coeff * (max_lr - min_lr)

weight_decay = 1e-1
#optimizer = torch.optim.AdamW(model.parameters(), lr=max_lr, betas=(0.9,0.95), eps=1e-8)
#trainable_parameters = [p for p in model.parameters() if p.requires_grad]  ## This line is not needed because I already filter out only the trainable parameters into the optimizer in the model's custom configured optimizer itself
optimizer = model.configure_optimizers(weight_decay, learning_rate = max_lr, betas=(0.9,0.95), device_type=device)


total_batch_size_fn = batch_size_fn * block_size # 524288   # = 2**19 so nice number. This number is the total number of tokens processed in a entire batch. We will split this up and do gradient accumulation to simulate doing this.
assert total_batch_size_fn % (micro_batch_size_fn * block_size) == 0, 'Make sure total batch size is divisible by micro batch size times block size'
grad_accum_steps = total_batch_size_fn // (micro_batch_size_fn * block_size)
print(f'Desired total batch size {total_batch_size_fn} and grad accum steps {grad_accum_steps}')


#oldtime = time.perf_counter()
losses = []
lrs = []

# Training

for epoch in range(n_epoch_fn):
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elif torch.backends.mps.is_available():
        torch.mps.synchronize()
    t0 = time.perf_counter()

    # zero the gradients
    optimizer.zero_grad() 
    
    # # evaluate loss once in a while
    # if epoch % (n_epoch_fn/10) == 0:
    #     testing_loss = estimate_loss()
    #     print(f"epoch {epoch}, train loss {testing_loss['train']}, val loss {testing_loss['val']}")

    # gradient accumulation
    loss_accum = 0.0
    cross_entropy_loss_accum = 0.0
    load_balance_loss_accum = 0.0
    for micro_step in range(grad_accum_steps):
        # sample a batch
        x, y, mask = get_batch('train')
        # forward pass
        with torch.autocast(device_type=device, dtype=torch.bfloat16):  # [speedup] useless for mps
            logits, cross_entropy_loss, load_balance_loss = model(x,y, loss_mask=mask)
            loss = cross_entropy_loss # No load balancing loss because LoRA is not applied on the MoE (router and expert MLPs) and only applied inside Attention (in nope parts of QKV)
                
        loss = loss / grad_accum_steps # this division is so that the gradients obtained by this way (grad accumulation) will match that of the entire big batch in one go.
        loss_accum += loss.item()
        cross_entropy_loss_accum += cross_entropy_loss.item()/grad_accum_steps
        load_balance_loss_accum  += load_balance_loss.item()/grad_accum_steps
        # backward pass to get new gradients
        loss.backward()
        
    losses.append(loss_accum)

    # gradient clipping
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    # get learning rate
    for param_group in optimizer.param_groups:
        lr = get_lr(epoch)
        param_group["lr"] = lr
    lrs.append(lr)
    
    # update parameters 
    optimizer.step()

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elif torch.backends.mps.is_available():
        torch.mps.synchronize()
    t1 = time.perf_counter()
    dt = t1-t0
    tokens_per_sec = (micro_batch_size_fn * block_size * grad_accum_steps) / dt

    # track
    print(f'step: {epoch} | loss: {loss_accum:.4f} | lr {lr:.4f} | grad norm: {norm:.4f} | dt: {dt*1000:.4f}ms | tok/sec: {tokens_per_sec:.4f} | cse loss: {cross_entropy_loss_accum:.4f} | load_loss: {load_balance_loss_accum:.4f}')



# # Estimation of final train and val losses
# testing_loss = estimate_loss(model)
# print(
#     f"Training finished at {len(losses)} epochs with Cross Entropy train loss {testing_loss['train']} and Cross Entropy val loss {testing_loss['val']}")

# Saving the trained model
def remove_orig_mod_prefix(state_dict): # this is to remove the prefix introduced by torch.compile
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("_orig_mod."):
            new_state_dict[k[len("_orig_mod."):]] = v
        else:
            new_state_dict[k] = v
    return new_state_dict
trained_state_dict = model.state_dict()
trained_state_dict = remove_orig_mod_prefix(trained_state_dict)
torch.save(trained_state_dict, "model_finetuned.pth")
print("Saved trained weights to model_finetuned.pth")

# Plotting loss and learning rate
plt.figure()
plt.plot(losses)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.grid(True)

plt.figure()
plt.plot(lrs)
plt.xlabel("Epochs")
plt.ylabel("lrs")
plt.title("Learning rate")
plt.grid(True)

plt.show()
