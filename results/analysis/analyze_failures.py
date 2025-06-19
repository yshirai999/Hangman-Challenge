import json
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import os

from agents.utils import data_path

LOG_FILE = data_path("logs/val_failures.jsonl")

# Load all logs into a DataFrame
def load_failures(path=LOG_FILE):
    if not os.path.exists(path):
        print(f"No log file found at {path}")
        return pd.DataFrame()
    with open(path, "r") as f:
        data = [json.loads(line) for line in f if line.strip()]
    return pd.DataFrame(data)

# Analyze key failure types
def summarize_failures(df):
    if df.empty:
        print("No data to summarize.")
        return

    df["word_length"] = df["word"].apply(len)
    df["num_missed"] = df["missed_letters"].apply(len)
    df["num_correct"] = df["word_length"] - df["num_missed"]
    df["missed_ratio"] = df["num_missed"] / df["word_length"]
    df["correct_ratio"] = df["num_correct"] / df["word_length"]

    print("\n--- Summary ---")
    print(f"Total Failures: {len(df)}")
    print(f"Average Word Length: {df['word_length'].mean():.2f}")
    print(f"Average Missed Letters: {df['num_missed'].mean():.2f}")
    print(f"Average Correct Guesses: {df['num_correct'].mean():.2f}")
    print(f"Avg Missed/Word Ratio: {df['missed_ratio'].mean():.2f}")
    print(f"Avg Correct/Word Ratio: {df['correct_ratio'].mean():.2f}\n")

    print("Most common missed letters:")
    all_missed = Counter(l for lst in df['missed_letters'] for l in lst)
    print(all_missed.most_common(10))

    print("\nMost frequent affixes involved (last turn):")
    last_affixes = [x[-1] if x else [] for x in df["affix_matches"]]
    flat = [a for aff in last_affixes for a in aff]
    print(Counter(flat).most_common(10))

    if "strategy_used" in df.columns:
        print("\nStrategy Usage in Failures:")
        print(df["strategy_used"].value_counts())

        print("\nAvg Missed/Correct Ratios by Strategy:")
        grouped = df.groupby("strategy_used")[["missed_ratio", "correct_ratio"]].mean()
        print(grouped.round(2))

    if "guess_history" in df.columns:
        guess_records = []
        for _, row in df.iterrows():
            word = row["word"]
            for guess_entry in row["guess_history"]:
                guess = guess_entry.get("guess")
                strategy = guess_entry.get("strategy", "unknown")
                is_correct = guess in word
                guess_records.append({"strategy": strategy, "correct": is_correct})

        guess_df = pd.DataFrame(guess_records)
        if not guess_df.empty:
            print("\nPer-Strategy Guess Accuracy:")
            strategy_accuracy = guess_df.groupby("strategy")["correct"].mean().sort_values(ascending=False)
            print(strategy_accuracy.round(2))

            print("\nPer-Strategy Guess Count:")
            strategy_usage = guess_df["strategy"].value_counts()
            print(strategy_usage)

# Plot word length vs correct guesses
def plot_lengths_vs_correct(df):
    if df.empty:
        return
    plt.scatter(df["word_length"], df["num_correct"], alpha=0.6)
    plt.xlabel("Word Length")
    plt.ylabel("Correct Guesses")
    plt.title("Correct Guesses vs Word Length")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    df = load_failures()
    summarize_failures(df)
    plot_lengths_vs_correct(df)
