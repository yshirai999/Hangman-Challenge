# generate_data.py

import argparse
from agents.agent import HangmanAgent
from agents.utils import data_path, load_dictionary, load_cache

"""
Generate training data (i.e. candidates features) for the random forest model.
This file will create self.sample_data, 
which is a list of tuples (features, label), in parallel.
"""

def main(args):
    dictionary = load_dictionary(args.dict_path)
    label_map = load_cache(args.label_map_path)
    cluster_summary_by_label = load_cache(args.cluster_summary_path)
    max_word_len = max(len(word) for word in dictionary)

    print("Initializing HangmanAgent...")
    agent = HangmanAgent(
        label_map=label_map,
        cluster_summary_by_label=cluster_summary_by_label,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        incorrect_guesses_allowed=args.incorrect_guesses_allowed,
        sample_data_size=args.sample_size,
        use_parallel_data_generation=True,
        dictionary=args.dict_path,
        ordered_dictionary=args.ordered_dict_path,
        max_word_len=max_word_len,
        mode="Subpattern_Greedy"
    )

    print("Generating training data in parallel...")
    agent.data()
    print("Data generation complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate training data for Hangman Agent.")
    parser.add_argument("--dict_path", type=str, default=data_path("words_250000_train.txt"))
    parser.add_argument("--ordered_dict_path", type=str, default=data_path("ordered_train_dictionary.pkl"))
    parser.add_argument("--label_map_path", type=str, default=data_path("label_map_50.pkl"))
    parser.add_argument("--cluster_summary_path", type=str, default=data_path("cluster_summary_by_label_50.pkl"))
    parser.add_argument("--sample_size", type=int, default=50000)
    parser.add_argument("--n_estimators", type=int, default=200)
    parser.add_argument("--max_depth", type=int, default=30)
    parser.add_argument("--incorrect_guesses_allowed", type=int, default=6)

    args = parser.parse_args()
    main(args)
