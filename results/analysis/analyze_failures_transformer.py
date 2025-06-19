import json
from collections import defaultdict
import argparse
import os

def load_jsonl(path):
    data = []
    with open(path, "r") as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return data

def summarize(data):
    total = len(data)
    exact = sum(d["correct"] for d in data)
    avg_acc = sum(d["letter_accuracy"] for d in data) / total
    avg_blanks = sum(d["total_blanks"] for d in data) / total
    avg_recovered = sum(d["blanks_recovered"] for d in data) / total
    high_acc = sum(d["letter_accuracy"] >= 0.8 for d in data)

    print(f"\n[Summary for {total} samples]")
    print(f"Exact matches:         {exact} ({exact/total:.2%})")
    print(f"Avg % blanks recovered: {avg_acc:.2%}")
    print(f"Avg # blanks per word:  {avg_blanks:.2f}")
    print(f"Avg # blanks recovered: {avg_recovered:.2f}")
    print(f"Samples ≥ 80% recovery: {high_acc} ({high_acc/total:.2%})")

def filter_failures(data, acc_threshold=0.5, min_blanks=3):
    return [
        d for d in data
        if not d["correct"]
        and d["letter_accuracy"] < acc_threshold
        and d["total_blanks"] >= min_blanks
    ]

def group_by_length(data):
    stats = defaultdict(list)
    for d in data:
        word_len = len(d["true_word"])
        stats[word_len].append(d["letter_accuracy"])
    return stats

def main(log_path, show_failures=False, show_by_length=False):
    if not os.path.exists(log_path):
        print(f"[!] Log file not found: {log_path}")
        return

    data = load_jsonl(log_path)
    summarize(data)

    if show_by_length:
        print("\n[Avg letter recovery by word length:]")
        grouped = group_by_length(data)
        for length in sorted(grouped):
            avg = sum(grouped[length]) / len(grouped[length])
            print(f" - Length {length:2d}: {avg:.2%} over {len(grouped[length])} samples")

    if show_failures:
        print("\n[Low-accuracy failures (acc < 50%, ≥3 blanks):]")
        failures = filter_failures(data)
        for d in failures[:10]:
            print(f"\nPattern:   {d['pattern']}")
            print(f"Guessed:   {d['guessed_letters']}")
            print(f"True:      {d['true_word']}")
            print(f"Pred:      {d['pred_word']}")
            print(f"Blank Acc: {d['letter_accuracy']:.2%} ({d['blanks_recovered']}/{d['total_blanks']})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=str, default="logs/v3_eval/transformer_best.jsonl")
    parser.add_argument("--failures", action="store_true", help="Show low-performing examples")
    parser.add_argument("--lengths", action="store_true", help="Show avg acc by word length")
    args = parser.parse_args()

    main(args.log, show_failures=args.failures, show_by_length=args.lengths)
