import torch

# Device
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

model_size = 'Big'

# Context Size
block_size = 256

# Width and Depth
d_hidden = 512
n_layers = 6

# Attention
d_latent = d_hidden // 4
n_heads = 16
assert d_hidden % n_heads == 0, 'd_head will not be an integer'
d_head = d_hidden//n_heads
d_rope = d_head // 4
assert d_rope % 2 == 0, 'need even d_rope for RoPE'
d_nope = d_head - d_rope

# MoE
d_intermediate = (d_hidden * 4 * 2 // 3)//2  # halving cuz I am using top-2 MoE
n_experts = 4
n_top_experts = 2
load_balance_strength = 1e-2
capacity_factor = 1.25

# Pre-Training
n_epoch = 500
batch_size = 64
micro_batch_size = 32 # micro-batch (for gradient accumulation)
dropout = 0.2

# Validation
eval_iters = 200

# Finetuning
n_epoch_fn = 1000
batch_size_fn = 64
micro_batch_size_fn =32 # micro-batch (for gradient accumulation)


#Generation
max_new_tokens = 100