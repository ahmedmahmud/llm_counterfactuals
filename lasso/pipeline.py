import re
import config
import torch
from nnsight import LanguageModel

def pipeline(model, statement, device, debug=False):
    tokenizer = model.tokenizer

    neg_id = tokenizer.convert_tokens_to_ids("No")
    pos_id = tokenizer.convert_tokens_to_ids("Yes")

    source_prefix, source_suffix = config.source_prompt.split("{statement}")
    source_prefix_tokens = tokenizer(source_prefix, return_tensors="pt", add_special_tokens=False).input_ids[0].to(device)
    source_suffix_tokens = tokenizer(source_suffix, return_tensors="pt", add_special_tokens=False).input_ids[0].to(device)

    source_prefix_len = source_prefix_tokens.size(0)

    stmt_tokens = tokenizer(statement, return_tensors="pt", add_special_tokens=False).input_ids[0].to(device)
    stmt_len = stmt_tokens.size(0)

    prompt_tokens = torch.cat([source_prefix_tokens, stmt_tokens, source_suffix_tokens])

    if debug:
        print(source_prefix_len, stmt_len, prompt_tokens.size(0), source_suffix_tokens.size(0))

    with model.trace(prompt_tokens) as _:
        x = model.model.embed_tokens.output.save()
        emb_orig_proxy = model.model.embed_tokens.output[
            0, source_prefix_len : source_prefix_len + stmt_len, :
        ].save()

    if debug:
        print(x.size(), "size of embedding matrix")

    emb_orig = emb_orig_proxy.detach()

    # 2) Initial forward to get original prediction
    with model.trace(prompt_tokens) as _:
        logits = model.lm_head.output.save()
    probs = torch.softmax(logits[0, -1], dim=-1)
    print(f"Initial Yes prob: {probs[pos_id]:.4f}, No prob: {probs[neg_id]:.4f}")

    target_cls = neg_id if probs[pos_id] > probs[neg_id] else pos_id
    print(f"Target class index: {target_cls} ({'Yes' if target_cls == 1 else 'No'})")

    # 3) Setup adversarial embedding
    emb_pert = emb_orig.clone().detach().requires_grad_(True)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam([emb_pert], lr=config.learning_rate)

    # 4) Adversarial optimization loop
    for step in range(1, config.max_steps + 1):
        with model.trace(prompt_tokens) as _:
            # Inject perturbed embeddings
            model.model.embed_tokens.output[
                0, source_prefix_len : source_prefix_len + stmt_len, :
            ] = emb_pert

            logits = model.lm_head.output.save()

        ce_loss = criterion(logits[0, -1].unsqueeze(0), torch.tensor([target_cls]).to(device))
        l1_reg = torch.sum(torch.abs(emb_pert - emb_orig))
        loss = ce_loss + config.l1_reg_factor * l1_reg

        with torch.no_grad():
            probs = torch.softmax(logits[0, -1], dim=-1)
            if debug:
                print(f"Step {step}/{config.max_steps} — total loss: {loss.item():.4f}, CE loss: {ce_loss.item():.4f}, L1 loss: {l1_reg.item():.4f}, probs: [No: {probs[neg_id].item()}, Yes: {probs[pos_id].item()}]")

        if probs[target_cls] > config.threshold and step > config.min_steps:
            if debug:
                print("Stopping as threshold and minstep reached")
            break

        # Backprop and update
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    # 6) Record activations at source_layer
    with model.trace(prompt_tokens) as _:
        model.model.embed_tokens.output[0][source_prefix_len : source_prefix_len + stmt_len, :] = emb_pert
        source_acts = model.model.layers[config.source_layer].output[0][:, source_prefix_len : source_prefix_len + stmt_len, :].save()

    # Build the prompt: prefix + fill tokens + suffix
    target_prefix, target_suffix = config.target_prompt.split("{statement}")
    target_prefix_tokens = tokenizer(target_prefix, return_tensors="pt", add_special_tokens=False).input_ids[0].to(device)
    target_suffix_tokens = tokenizer(target_suffix, return_tensors="pt", add_special_tokens=False).input_ids[0].to(device)

    target_prefix_len = target_prefix_tokens.size(0)

    fill_id = tokenizer.convert_tokens_to_ids("?")
    fill_tokens = torch.tensor([fill_id] * stmt_len).to(device)

    prompt = torch.cat([target_prefix_tokens, fill_tokens, target_suffix_tokens])

    with model.edit() as model_edited:
        model.model.layers[config.target_layer].output[0][:, target_prefix_len : target_prefix_len + stmt_len, :] = source_acts

    with model_edited.generate(prompt, max_new_tokens=100) as _:
        output = model.generator.output.save()

    counterfactual = extract_model_response(model.tokenizer.decode(output[0].cpu()))

    return {
        "original": statement,
        "counterfactual": counterfactual,
    }

def extract_model_response(generated_text):
    match = re.search(r"<start_of_turn>model\s*(.*?)(?:<end_of_turn>|$)", generated_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = LanguageModel(
        config.model_name,
        device_map="auto",
        load_in_4bit=True,
        torch_dtype=torch.bfloat16,
    )
    print(model)

    statement = "the plot is nothing but boilerplate clichés from start to finish ,"

    result = pipeline(model, statement, debug=True)
    print(result['counterfactual'])
