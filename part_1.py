import torch
from transformer_lens import HookedTransformer

import config

model = HookedTransformer.from_pretrained(
    model_name=config.model_name, dtype="float16", device="cuda"
)

yes_token_id = model.to_single_token("Yes")
no_token_id = model.to_single_token("No")
print(f"Yes token ID: {yes_token_id}")
print(f"No token ID: {no_token_id}")

statement_tokens = model.to_tokens(config.statement)
statement_length = len(statement_tokens[0])

prompt = config.prompt_template.format(statement=config.statement)
prompt_tokens = model.to_tokens(prompt)

prefix_tokens = model.to_tokens(config.template_prefix)
prefix_length = len(prefix_tokens[0])

prompt_embeddings = model.embed(prompt_tokens).detach()

mask = torch.zeros_like(prompt_embeddings)
mask[0, prefix_length : prefix_length + statement_length, :] = 1.0

logits = model(prompt_tokens)

[pos_prob, neg_prob] = (
    logits[0, -1, [yes_token_id, no_token_id]].softmax(dim=-1).tolist()
)

print(f"Yes prob: {pos_prob}, No prob: {neg_prob}")
if pos_prob > neg_prob:
    print("Prediction: Yes")
else:
    print("Prediction: No")

# Find adversarial example that flips the prediction
target_class = 1 if pos_prob > neg_prob else 0

perturbed_embedding = prompt_embeddings.clone().detach().requires_grad_(True)


# Hook to replace the embedding with the perturbed one
def embed_hook(value, hook):
    return perturbed_embedding


for step in range(config.max_steps):
    print(f"Step {step + 1}/{config.max_steps}")

    with model.hooks(fwd_hooks=[("hook_embed", embed_hook)]):
        logits = model(prompt_tokens)

    # TODO: Add regularization term for token similarity (maybe use semantic similarity via BERTScore)
    loss = -torch.nn.functional.log_softmax(
        logits[0, -1, [yes_token_id, no_token_id]], dim=-1
    )[target_class]
    print(f"Loss: {loss.item()}")
    loss.backward()

    with torch.no_grad():
        perturbed_embedding -= config.learning_rate * (perturbed_embedding.grad * mask)

    # Check if the prediction has flipped
    with torch.no_grad():
        with model.hooks(fwd_hooks=[("hook_embed", embed_hook)]):
            logits = model(prompt_tokens)
        [pos_prob, neg_prob] = (
            logits[0, -1, [yes_token_id, no_token_id]].softmax(dim=-1).tolist()
        )
        print(
            f"Step {step + 1}/{config.max_steps}: Yes prob: {pos_prob}, No prob: {neg_prob}"
        )
        if [pos_prob, neg_prob][target_class] > config.threshold:
            print("Prediction flipped! Stopping")
            break

# Get activations from source pass at layer "l" for statement tokens (to later be used for patching into target pass)
with model.hooks(fwd_hooks=[("hook_embed", embed_hook)]):
    _, cache = model.run_with_cache(prompt_tokens)
layer_name = f"blocks.{config.source_layer}.hook_resid_post"
source_activations = cache[layer_name][
    :, prefix_length : prefix_length + statement_length, :
]

torch.save(source_activations, config.activation_cache_path)
