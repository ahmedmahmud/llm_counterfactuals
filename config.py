model_name = "meta-llama/Llama-3.1-8B-Instruct"

# Config for embedding pertubation finder
max_steps = 100
learning_rate = 1
threshold = 0.8

statement = "The weather is great today."

# Source model classifier prompt
prompt_template = """
Is this statement positive? Answer Yes or No:
Statement: {statement}
Answer:"""

# Prefix needed for creating mask to extract `statement`
template_prefix = """Is this statement positive? Answer Yes or No:
Statement: """

# Few-show prompt for target model
# TODO: Test if this prompt actually works
target_prompt = "I love pizza->I love pizza; The weather is nice->The weather is nice; They didn't enjoy the show->They didn't enjoy the show; "
# target_prompt = "cat->cat; 135->135; hello->hello; x"

# Patchscopes layer config
source_layer = 25
target_layer = 0

# Token generation count after patchscopes from target model pass
max_new_tokens = 10

# Path to save the source activations between part_1 and part_2
activation_cache_path = "source_activations.pt"