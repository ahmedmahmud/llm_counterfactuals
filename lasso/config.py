model_name = "google/gemma-2-2b-it"

# Config for embedding pertubation finder
max_steps = 50
min_steps = 20
learning_rate = 0.01
threshold = 0.95

l1_reg_factor = 1

source_prompt = """<start_of_turn>user
{statement}
Is the above statement positive in sentiment? Answer Yes or No only<end_of_turn>
<start_of_turn>model
"""

# Few-show prompt for target model

target_prompt = """<start_of_turn>user
Repeat the following sentence below.
{statement}<end_of_turn>
<start_of_turn>model
"""

# Patchscopes layer config
source_layer = 5
target_layer = 0

# Token generation count after patchscopes from target model pass
max_new_tokens = 100