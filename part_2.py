from functools import partial
from typing import Callable, cast

import torch
import transformer_lens.utils as utils
from transformer_lens import HookedTransformer
from transformer_lens.ActivationCache import ActivationCache
from transformer_lens.hook_points import (
    HookPoint,
)

import config

model = HookedTransformer.from_pretrained(
    model_name=config.model_name, dtype="float16", device="cuda"
)

statement_tokens = model.to_tokens(config.statement)
statement_length = len(statement_tokens[0])

target_prompt_tokens = model.to_tokens(config.target_prompt)
target_prefix_length = len(target_prompt_tokens[0])

fill_token_id = model.to_single_token("x")
additional_tokens = torch.full(
    (1, statement_length), fill_token_id, device=target_prompt_tokens.device
)
target_prompt_tokens = torch.cat([target_prompt_tokens, additional_tokens], dim=1)



source_activations: torch.Tensor = cast(
    torch.Tensor, torch.load("source_activations.pt")
)


def feed_source_representation(
    source_rep: torch.Tensor,
    tokens: torch.Tensor,
    model: HookedTransformer,
    layer_id: int,
) -> ActivationCache:
    def resid_ablation_hook(value, _: HookPoint):
        value[:, target_prefix_length : target_prefix_length + statement_length, :] = (
            source_rep
        )
        return value

    logits: ActivationCache = model.run_with_hooks(
        tokens,
        return_type="logits",
        fwd_hooks=[(utils.get_act_name("resid_post", layer_id), resid_ablation_hook)],
    )

    return logits


def generate_with_patching(
    model: HookedTransformer,
    tokens: torch.Tensor,
    target_f: Callable,
    max_new_tokens: int,
):
    input_tokens = tokens
    for _ in range(max_new_tokens):
        logits = target_f(
            input_tokens,
        )
        next_tok = torch.argmax(logits[:, -1, :])
        input_tokens = torch.cat(
            (input_tokens, next_tok.view(input_tokens.size(0), 1)), dim=1
        )

    return model.to_string(input_tokens)[0]


target_f = partial(
    feed_source_representation,
    source_rep=source_activations,
    model=model,
    layer_id=config.target_layer,
)
print(generate_with_patching(model, target_prompt_tokens, target_f, config.max_new_tokens))
