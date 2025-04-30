import torch

def print_topk_tokens(logits, tokenizer, k):
  probs = torch.softmax(logits, dim=-1)
  topk_probs, topk_ids = torch.topk(probs, k)
  raw_tokens = tokenizer.convert_ids_to_tokens(topk_ids.tolist())
  decoded = [tokenizer.decode([tid]) for tid in topk_ids.tolist()]

  print(f"top-{k} predictions:")
  for id, tok, rep, p in zip(topk_ids.tolist(), raw_tokens, decoded, topk_probs.tolist()):
      print(f"  {id} {tok} ({repr(rep)}): {p:.4f}")

def print_token_ids(prompt, tokenizer):
  ids = tokenizer(prompt, return_tensors="pt",).input_ids[0]
  decoded = [tokenizer.decode([tid]) for tid in ids.tolist()]

  for idx, tok in enumerate(decoded):
    print(f"{idx:3d}: {repr(tok)}")