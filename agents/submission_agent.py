import yaml
import sys
import random
import os

from agents.agent import HangmanAgent
from agents.utils import (
    HangmanTransformerModel,
    HangmanTransformerModelV2,
    HangmanTransformerModelV3,
    load_dictionary,
    load_cache,
    save_cache,
    data_path,
    model_path,
    config_path,
    results_path,
    checkpoint_path,
    get_or_create_affixes,
    prune_nested_affixes,
    consolidate_affixes,
    get_or_create_train_test_dictionary,
    get_or_create_ordered_dictionary,
    get_or_create_cluster_data,
    stratify_words_by_features,
    sample_stratified_words,
    HangmanTransformerDataset,
)

with open(config_path(), "r") as f:
    config = yaml.safe_load(f)

# Load dictionary and preprocess
dictionary = load_dictionary(data_path("words_250000_train.txt"))

train_dictionary, test_dictionary = get_or_create_train_test_dictionary(
    dictionary=dictionary,
    train_path=data_path("train_dictionary.pkl"),
    test_path=data_path("test_dictionary.pkl")
)

ordered_train_dictionary = get_or_create_ordered_dictionary(
    dictionary=train_dictionary,
    cache_path=data_path("ordered_train_dictionary.pkl")
)

label_map, cluster_summary_by_label = get_or_create_cluster_data(
    train_dictionary,
    n_clusters=50
)

max_word_len = max(len(word) for word in train_dictionary)

# Load trained agent
agent = HangmanAgent(
    label_map=label_map,
    cluster_summary_by_label=cluster_summary_by_label,
    n_estimators=200,
    max_depth=30,
    incorrect_guesses_allowed=6,
    use_parallel_data_generation=True,
    parallelize_data_from_subpattern_agent_generation=True,
    sample_data_size=50000,
    stratified_sample_data_size=50000,
    transformer_sample_data_size=200000,
    transformer_subpattern_sample_data_size=25000,
    transformer_subpattern_short_sample_data_size=50000,
    dictionary="train_dictionary.pkl",
    ordered_dictionary="ordered_train_dictionary.pkl",
    max_word_len=max_word_len,
    use_subpattern_data_for_transformer=True,
    use_short_words_for_transformer=True,
    TransformerModel=HangmanTransformerModelV3,
    mode="Subpattern_Greedy",  # Choose from:
        # "ML",
        # "Frequency",
        # "Entropy",
        # "Logical",
        # "Subpattern_Greedy",
        # "Hybrid_nonML",
        # "Hybrid_ML",
        # "Hybrid_SB_Transformer",
        # "Hybrid_SB_Transformer_V2",
        # "Transformer"
)

agent.mode = config.get("agent_mode", "Subpattern_Greedy")

#########################################
# Load or generate affix data for SB agent
#########################################

try:
    agent.affix_data_nonconsolidated = load_cache(data_path("pruned_affixes_train.pkl"))
except FileNotFoundError:
    print("Pruned affix data not found. Generating pruned affix data...")

    try:
        affix_data = load_cache(data_path("affixes_train.pkl"))
    except FileNotFoundError:
        print("Affix data not found. Generating affix data...")
        affix_data = get_or_create_affixes(
            train_dictionary,
            affix_cache_path=data_path("affixes_train.pkl")
        )
        print("Affix data generated and cached.")

    # Combine all affixes into a single dictionary
    combined_affixes = {}
    for category in ["prefixes", "suffixes", "midfixes"]:
        combined_affixes.update(affix_data[category])

    # Prune using new logic (across all categories)
    relative_thresh = 0.2  # can tune this
    pruned_combined = prune_nested_affixes(combined_affixes, relative_freq_thresh=relative_thresh)

    # Split pruned affixes back into categories
    pruned_prefixes = {k: v for k, v in pruned_combined.items() if k in affix_data["prefixes"]}
    pruned_suffixes = {k: v for k, v in pruned_combined.items() if k in affix_data["suffixes"]}
    pruned_midfixes = {k: v for k, v in pruned_combined.items() if k in affix_data["midfixes"]}

    pruned_affix_data = {
        "prefixes": pruned_prefixes,
        "suffixes": pruned_suffixes,
        "midfixes": pruned_midfixes
    }

    save_cache(pruned_affix_data, "pruned_affixes_train.pkl")
    print("Saved pruned affix set to pruned_affixes_train.pkl")
    agent.affix_data_nonconsolidated = pruned_affix_data

try:
    affix_data_cons = load_cache(data_path("affixes_train_cons.pkl"))
except FileNotFoundError:
    print("Consolidated affix data not found. Generating consolidated affix data...")
    affix_data_cons = consolidate_affixes(agent.affix_data_nonconsolidated)
    save_cache(affix_data_cons, data_path("affixes_train_cons.pkl"))
    print("Saved consolidated affix set to affixes_train_cons.pkl")

agent.affix_data = affix_data_cons

#######################################
# Random Forest Model Data and Training
#######################################
if agent.mode in ["ML", "Hybrid_ML"]:
    try:
        agent.sample_data = load_cache(data_path(f"sample_data{agent.sample_data_size}.pkl"))
        print(f"Loaded cached sample data: {len(agent.sample_data)} samples.")
    except FileNotFoundError:
        if agent.use_parallel_data_generation:
            print("Run generate_data.py" \
            " to generate training data in parallel.")
            sys.exit(1)
        else:
            agent.data()

    try:
        agent.training_data = load_cache(
                data_path(f"training_data{agent.sample_data_size}.pkl")
            )
        agent.validation_data = load_cache(
                data_path(f"validation_data{agent.sample_data_size}.pkl")
            )
        agent.testing_data = load_cache(
                data_path(f"testing_data{agent.sample_data_size}.pkl")
            )
    except FileNotFoundError:
            agent.split_and_save_data(agent.sample)

    if not agent.model_loaded:
        agent.model.n_jobs = -1 # Multithread for fitting
        agent.train_model()
        # agent.plot_learning_curve()

    agent.model.n_jobs = 1 # Single thread for inference
    agent.model.verbose = 0 # No verbose output

agent.test_dictionary = test_dictionary
agent.train_dictionary = train_dictionary

#########################################
# Transformer Model Data and Training
#########################################

# Extra split: 80% SB training, 10% Transformer train, 10% Transformer test

try:
    agent.train_words_for_transformer = load_cache(data_path("train_words_for_transformer.pkl"))
    agent.test_words_for_transformer = load_cache(data_path("test_words_for_transformer.pkl"))
    print("Loaded cached train and test words for transformer.")
except FileNotFoundError:
    filtered_words = [w for w in agent.test_dictionary if len(w) > 5]
    random.shuffle(filtered_words)
    N = len(filtered_words)
    agent.train_words_for_transformer = filtered_words[:int(0.5 * N)]
    agent.test_words_for_transformer = filtered_words[int(0.5 * N):]
    save_cache(agent.train_words_for_transformer, data_path("train_words_for_transformer.pkl"))
    save_cache(agent.test_words_for_transformer, data_path("test_words_for_transformer.pkl"))

# Load or create Transformed data from stratified sample of full dictionary
# and completely random pattern and guesses

try:
    agent.stratified_sample = load_cache(data_path(f"stratified_word_sample"
                                         f"{agent.stratified_sample_data_size}.pkl"))
    print(f"Loaded stratified word sample of size {agent.stratified_sample_data_size} "
          f"from stratified_word_sample{agent.stratified_sample_data_size}.pkl")
except FileNotFoundError:
    print(f"No cached stratified sample found. Generating now...")

    # Generate stratified sample using your agent's dictionary and affixes
    stratified_buckets = stratify_words_by_features(agent.dictionary, agent.affix_data)

    # Sample ~50k stratified words
    agent.stratified_sample = sample_stratified_words(stratified_buckets,
                            total_samples=agent.stratified_sample_data_size)

    # Save to disk
    save_cache(agent.stratified_sample, data_path(f"stratified_word_sample{agent.stratified_sample_data_size}.pkl"))
    print(f"Stratified word sample created and saved "
          f"to stratified_word_sample{agent.stratified_sample_data_size}.pkl")

try:
    agent.transformer_data = load_cache(data_path(f"transformer_data"
                                        f"{agent.transformer_sample_data_size}.pkl"))
    print(f"Loaded cached transformer data: {len(agent.transformer_data)} samples.")
except FileNotFoundError:
    print("Transformer data not found. Generating new samples...")

    # Generate late-stage samples
    agent.transformer_data = agent.generate_late_stage_samples_from_words(
        words=agent.stratified_sample,
        num_samples=agent.transformer_sample_data_size
    )

    save_cache(agent.transformer_data,
               data_path(f"transformer_data{agent.transformer_sample_data_size}.pkl"))
    print(f"Saved {agent.transformer_sample_data_size} transformer samples to cache.")

# Load or create Transformed data from Transformer train dictionary
# and pattern and guesses generated by Subpattern Greedy agent

try:
    agent.transformer_data_from_subpattern = load_cache(data_path(f"transformer_data_from_subpattern{agent.transformer_subpattern_sample_data_size}.pkl"))
    print(f"Loaded cached transformer data from subpattern agent.")
except FileNotFoundError:
    print("Transformer data from subpattern agent not found. Generating new samples...")
    if agent.parallelize_data_from_subpattern_agent_generation:
        print("Parallel generation enabled. To generate data in parallel, "
              "Please run generate_transformer_data_subpattern.py first.")
        sys.exit(1)
    else:
        print("Parallel generation disabled. Generating sequentially...")
    # Generate late-stage samples using subpattern agent
    agent.transformer_data_from_subpattern = agent.generate_full_games_from_subpattern_agent(
        word_list=agent.train_words_for_transformer,
        num_samples=agent.transformer_subpattern_sample_data_size
    )

    save_cache(agent.transformer_data_from_subpattern,
               data_path(f"transformer_data_from_subpattern{agent.transformer_subpattern_sample_data_size}.pkl"))
    print(f"Saved transformer samples from subpattern agent to cache.")

try:
    eval_data = load_cache(data_path("transformer_eval_data_from_subpattern.pkl"))
    print(f"Loaded cached evaluation data from transformer_eval_data_from_subpattern.pkl ({len(eval_data)} samples)")
except FileNotFoundError:
    print(f"No eval data found — generating 1000 samples from test_words_for_transformer")
    eval_data = agent.generate_full_games_from_subpattern_agent(
        word_list=agent.test_words_for_transformer,
        num_samples=1000
    )
    save_cache(eval_data, data_path("transformer_eval_data_from_subpattern.pkl"))
    print(f"Saved evaluation data to transformer_eval_data_from_subpattern.pkl")

# Load or create Transformed data for short words from Transformer train dictionary
# and pattern and guesses generated by Subpattern Greedy agent

agent.short_word_list = [w for w in train_dictionary if 5 <= len(w) <= 10]
# agent.short_word_list = [w for w in train_dictionary if 5 <= len(w) <= 10] # Alternative
try:
    agent.transformer_data_from_subpattern_short = load_cache(data_path(f"transformer_data_from_subpattern_short{agent.transformer_subpattern_short_sample_data_size}.pkl"))
    print(f"Loaded cached transformer data from subpattern agent for short words.")
except FileNotFoundError:
    print(f"No cached transformer data found for short words — generating new samples...")
    if agent.parallelize_data_from_subpattern_agent_generation:
        print("Parallel generation enabled. To generate data in parallel, "
              "Please run generate_transformer_data_subpattern.py first.")
        sys.exit(1)
    else:
        print("Parallel generation disabled. Generating sequentially...")
        agent.transformer_data_from_subpattern_short = agent.generate_full_games_from_subpattern_agent(
            word_list=agent.short_word_list,
            num_samples=agent.transformer_subpattern_short_sample_data_size
        )
        save_cache(agent.transformer_data_from_subpattern_short,
                data_path(f"transformer_data_from_subpattern_short{agent.transformer_subpattern_short_sample_data_size}.pkl"))
        print(f"Saved transformer data from subpattern agent for short words to cache.")

agent.short_word_list_eval = [w for w in agent.test_words_for_transformer if 5 <= len(w) <= 10]
try:
    eval_data = load_cache(data_path("transformer_eval_data_from_subpattern_short.pkl"))
    print(f"Loaded cached evaluation data from transformer_eval_data_from_subpattern_short.pkl ({len(eval_data)} samples)")
except FileNotFoundError:
    print(f"No eval data found — generating 1000 samples from test_words_for_transformer")
    eval_data = agent.generate_full_games_from_subpattern_agent(
        word_list=agent.short_word_list_eval,
        num_samples=1000
    )
    save_cache(eval_data, data_path("transformer_eval_data_from_subpattern_short.pkl"))
    print(f"Saved evaluation data to transformer_eval_data_from_subpattern_short.pkl")

# Set up transformer dataset

if agent.use_subpattern_data_for_transformer:
    if agent.use_short_words_for_transformer:
        print("Using short word samples with subpattern based pattern and guesses" \
                " from Transformer train dictionary.")
        torch_dataset = f"torch_dataset_from_subpattern_short{agent.transformer_subpattern_short_sample_data_size}.pkl"
        transformer_dataset = agent.transformer_data_from_subpattern_short
    else:
        print("Using samples with subpattern based pattern and guesses" \
                " from Transformer train dictionary.")
        torch_dataset = f"torch_dataset_from_subpattern{agent.transformer_subpattern_sample_data_size}.pkl"
        transformer_dataset = agent.transformer_data_from_subpattern
else:
    print("Using samples with random pattern and guesses from SB train dictionary.")
    torch_dataset = f"torch_dataset{agent.transformer_sample_data_size}.pkl"
    transformer_dataset = agent.transformer_data

use_blank_mask = (agent.TransformerModel == HangmanTransformerModelV3)
try:
    print(f"Loading {torch_dataset} from cache...")
    print(data_path(torch_dataset))
    agent.torch_dataset = load_cache(data_path(torch_dataset))
    print(f"Loaded {torch_dataset} from cache.")
except FileNotFoundError:
    print(f"Transformer dataset not found. Creating now...")
    agent.torch_dataset = HangmanTransformerDataset(
        transformer_dataset,
        max_len=agent.max_len_for_transformer,
        use_blank_mask=use_blank_mask
    )
    save_cache(agent.torch_dataset, data_path(torch_dataset))
    print(f"Saved torch_dataset to cache.")

print(
    f"Training transformer on "
    f"{'subpattern-based' if agent.use_subpattern_data_for_transformer else 'random late-stage'} "
    "patterns."
)

use_cuda = True  # or False if you prefer

# Load Transformer model

checkpoint_dir = (data_path("checkpoints/v3_run") if agent.TransformerModel == HangmanTransformerModelV3
    else data_path("checkpoints/v2_run") if agent.TransformerModel == HangmanTransformerModelV2
    else data_path("checkpoints/v1_run")
)

if agent.use_subpattern_data_for_transformer:
    if agent.TransformerModel == HangmanTransformerModelV3:
        print("Using HangmanTransformerModelV3 with subpattern short-word data.")
        transformer_model_path = (
            "transformer_model_subpattern_v3_short.pt"
            if agent.use_short_words_for_transformer else
            "transformer_model_subpattern_v3.pt"
        )
    elif agent.TransformerModel == HangmanTransformerModelV2:
        print("Using HangmanTransformerModelV2 with subpattern data.")
        transformer_model_path = "transformer_model_subpattern_v2.pt"
    else:
        print("Using HangmanTransformerModel with subpattern data.")
        transformer_model_path = "transformer_model_subpattern.pt"
else:
    print("Using model with random late-stage pattern/guess generation.")
    transformer_model_path = "transformer_model_random_late_stage.pt"

try:
    agent.transformer_model = agent.load_transformer_model(
        agent.TransformerModel,
        model_path(transformer_model_path),
        use_cuda
    )
    print(f"Loaded transformer model from {transformer_model_path}")
except FileNotFoundError:
    print(f"No saved model at {transformer_model_path} — training from scratch.")
    agent.transformer_model = agent.TransformerModel(
        vocab_size=agent.vocab_size,
        max_seq_len=agent.max_len_for_transformer
    )
    agent.train_transformer_model(agent.transformer_model,
                                  epochs=20,
                                  use_cuda=use_cuda,
                                  lr = 1e-4,
                                  save_path=transformer_model_path,
                                  batch_size=1024,
                                  checkpoint_dir=checkpoint_path(checkpoint_dir),
                                  use_blank_mask=use_blank_mask
    )

print(f"Agent mode: {agent.mode}")

#########################################
# Guess function to pass to Trexquant's server
#########################################

def guess_YS(pattern: str, guessed: str) -> str:
    return agent(pattern, guessed)