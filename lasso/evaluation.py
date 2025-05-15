import config
import Levenshtein
import pandas as pd
import torch
from datasets import load_dataset
from nnsight import LanguageModel
from pipeline import pipeline
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load SST-2 validation set
dataset = load_dataset("stanfordnlp/sst2", split="validation[:50]")

# Load model
model = LanguageModel(
    config.model_name,
    device_map="auto",
    load_in_4bit=True,
    torch_dtype=torch.bfloat16,
)

tokenizer = model.tokenizer

neg_id = tokenizer.convert_tokens_to_ids("No")
pos_id = tokenizer.convert_tokens_to_ids("Yes")


def classify(statement):
    prompt = config.source_prompt.replace("{statement}", statement)

    with model.trace(prompt) as _:
        logits = model.lm_head.output[0, -1].save()
    print(logits.size())
    label = 0 if logits[neg_id] > logits[pos_id] else 1
    return label


# def compute_perplexity(statement):
#     return


def compute_similarity(original, counterfactual):
    distance = Levenshtein.distance(original, counterfactual)
    max_len = max(len(original), len(counterfactual))
    if max_len == 0:
        return 1.0  # avoid division by zero
    return 1 - (distance / max_len)


results = []

for sample in tqdm(dataset):
    statement = sample["sentence"]
    orig_label = sample["label"]

    try:
        output = pipeline(model, statement, device, debug=False)
        counterfactual = output["counterfactual"]

        new_label = classify(counterfactual)

        flip = int(orig_label != new_label)
        # ppl = compute_perplexity(counterfactual)
        sim = compute_similarity(statement, counterfactual)

        results.append(
            {
                "original": statement,
                "counterfactual": counterfactual,
                "original_label": orig_label,
                "counterfactual_label": new_label,
                "flip": flip,
                # "perplexity": ppl,
                "similarity": sim,
            }
        )
    except Exception as e:
        print(f"Error on: {statement[:60]}... — {e}")

# Save results
df = pd.DataFrame(results)
df.to_csv("sst2_counterfactual_results.csv", index=False)
print(df.head())
# print(
#     f"\nFlip Rate: {df['flip'].mean():.2f}, Avg Perplexity: {df['perplexity'].mean():.2f}, Avg Similarity: {df['similarity'].mean():.2f}"
# )
