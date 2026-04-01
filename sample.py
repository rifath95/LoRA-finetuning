# generate from the model using KV cache
context = torch.zeros((1, block_size), dtype=torch.long, device=device)
print(decode(model.generate(context, max_new_tokens, temperature=0.9, top_k=50)[0].tolist()))

# generate from the model without KV cache
context = torch.zeros((1, block_size), dtype=torch.long, device=device)
print(decode(model.generate(context, max_new_tokens, temperature=0.9, top_k=50, use_kvcaching=False)[0].tolist()))
