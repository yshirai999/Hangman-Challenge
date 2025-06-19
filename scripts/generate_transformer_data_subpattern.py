import random
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from agents.agent import HangmanAgent
from agents.utils import load_cache, load_dictionary, get_or_create_train_test_dictionary, data_path
import argparse

## To run the script for short words
# python generate_transformer_data_subpattern.py --short_words
## To run the script for all words
# python generate_transformer_data_subpattern.py

# === Load dictionary and preprocess === ## Need to make this better
dictionary = load_dictionary(data_path("words_250000_train.txt"))
train_dictionary, test_dictionary = get_or_create_train_test_dictionary(
    dictionary = dictionary,
    train_path=data_path("train_dictionary.pkl"),
    test_path=data_path("test_dictionary.pkl")
)

# === CONFIG ===
NUM_SAMPLES = 50000
WORKERS = 20
MIN_REVEAL_FRACTION = 0.6

def simulate_sample_batch(word_list, num_target_samples, seed, word_length_range):
    random.seed(seed)
    agent = HangmanAgent(
        transformer_subpattern_sample_data_size=num_target_samples,
        mode="Subpattern_Greedy",
        load_fallback_order=False
    )
    agent.affix_data = load_cache(data_path("affixes_train_cons.pkl"))

    filtered_words = [
        w for w in word_list
        if word_length_range[0] <= len(w) <= word_length_range[1]
    ]

    return agent.generate_full_games_from_subpattern_agent(
        word_list=filtered_words,
        num_samples=num_target_samples,
        min_reveal_fraction=MIN_REVEAL_FRACTION
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--short_words", action="store_true", help="Use only words of length 5-10")
    args = parser.parse_args()

    if args.short_words:
        word_length_range = (5, 10)
        save_full_path = data_path(f"transformer_data_from_subpattern_short{NUM_SAMPLES}.pkl")
    else:
        word_length_range = (6, 30)  # or whatever upper limit you like
        save_full_path = data_path(f"transformer_data_from_subpattern{NUM_SAMPLES}.pkl")


    #all_words = load_cache("train_words_for_transformer.pkl")
    all_words = [w for w in train_dictionary if word_length_range[0] <= len(w) <= word_length_range[1]]


    print(f"Loaded {len(all_words)} eligible train words")

    samples_per_worker = NUM_SAMPLES // WORKERS
    seeds = [42 + i for i in range(WORKERS)]

    print(f"Spawning {WORKERS} workers to generate {NUM_SAMPLES} samples...")

    all_samples = []
    with ProcessPoolExecutor(max_workers=WORKERS) as executor:
        futures = [
            executor.submit(simulate_sample_batch, all_words, samples_per_worker, seed, word_length_range)
            for seed in seeds
        ]
        for future in tqdm(as_completed(futures), total=WORKERS):
            result = future.result()
            all_samples.extend(result)

    print(f"Total collected full-history samples: {len(all_samples)}")

    # Save full history version
    with open(save_full_path, "wb") as f:
        pickle.dump(all_samples, f)
    print(f"Full history saved to {save_full_path}")

if __name__ == "__main__":
    main()
    
