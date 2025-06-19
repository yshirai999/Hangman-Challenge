import random
import json
import os
import sys
import argparse

from agents.submission_agent import guess_YS, agent, dictionary, test_dictionary, train_dictionary
import agents.utils
from agents.utils import results_path, results_path

from collections import Counter, defaultdict

os.makedirs(results_path('logs/evaluate_bot'), exist_ok=True)

###############################################
# Creating testing dictionary preserving
# Trexquant proportions for Hybrid-Transformer agent
# where subpattern greedy agent was trained
# on train dictionary and transformer trained
# on half of the test dictionary
###############################################

def test_dictionary_transformers(num_games: int = 1000) -> list:
    """
    Create a test dictionary for Hybrid-Transformer agents that preserves
    the original Trexquant proportions of word lengths.
    """
    random.seed(42)

    # Ensure the agent has a training dictionary
    if not train_dictionary or not agent.train_words_for_transformer:
        raise ValueError("Agent must have a training dictionary and words for Transformer mode.")

    # Use the original Trexquant dictionary
    if not dictionary:
        raise ValueError("The original Trexquant dictionary is empty.")

    # Count lengths in original Trexquant dictionary
    length_counts = Counter(len(w) for w in dictionary)
    total_words = sum(length_counts.values())

    # Compute proportions
    length_proportions = {k: v / total_words for k, v in length_counts.items()}

    # Remove all seen words
    seen_words = set(train_dictionary) | set(agent.train_words_for_transformer)
    unseen_words = [w for w in dictionary if w not in seen_words]

    # Group unseen words by length
    unseen_by_length = defaultdict(list)
    for w in unseen_words:
        unseen_by_length[len(w)].append(w)
    
      # total evaluation games
    test_dictionary_transformers = []

    for length, proportion in length_proportions.items():
        group = unseen_by_length[length]
        n = min(int(proportion * num_games), len(group))
        if n > 0:
            test_dictionary_transformers.extend(random.sample(group, n))

    actual_games = len(test_dictionary_transformers)
    print(f"Requested: {num_games}, Created: {actual_games} (after filtering and stratification)")

    dist = Counter(len(w) for w in test_dictionary_transformers)
    print("Stratified test set distribution:", dict(dist))

    random.shuffle(test_dictionary_transformers)

    #sys(exit(1))
    return test_dictionary_transformers

###############################################
# Logging failures
###############################################

def log_failure(word, pattern, guessed_letters, incorrect_guesses,
                guess_history, affix_matches=None, fallback_used=None,
                source="Hybrid-Transformer", strategy_used=None,
                log_dir=results_path("logs/evaluate_bot/")
):
    """
    Log failure diagnostics to a JSON lines file for later analysis.
    """
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(results_path(log_dir), f"{source}_{agent.mode}_failures.jsonl")


    missed_letters = [c for c in word if c not in guessed_letters]
    failure_record = {
        "word": word,
        "final_pattern": ''.join(pattern),
        "guessed_letters": guessed_letters,
        "wrong_guesses": incorrect_guesses,
        "remaining_lives": 6 - incorrect_guesses,
        "guess_history": guess_history,
        "missed_letters": missed_letters,
        "affix_matches": affix_matches,
        "fallback_used": fallback_used,
        "source": source,
        "failure_type_tag": ""
    }

    if strategy_used:
        failure_record["strategy_used"] = strategy_used

    with open(log_path, "a") as f:
        json.dump(failure_record, f)
        f.write("\n")

###############################################
# Main game loop for the agent
###############################################

def play_game(secret_word: str,
            incorrect_guesses_allowed: int = 6,
            log_failures: bool = True,
            source: str = "Hybrid-Transformer"
    ) -> tuple:
    
    guessed_letters = []
    pattern = ['_' for _ in secret_word]
    incorrect_guesses = 0
    guess_history = []

    affix_matches_all = []
    fallback_used_flag = False  # optional flag for future use

    while incorrect_guesses < incorrect_guesses_allowed and '_' in pattern:
        curr_pattern = ''.join(pattern)
        matched_affixes = agents.utils.fuzzy_affix_match(
            pattern=curr_pattern,
            affix_dicts=agent.affix_data,
            guessed_letters=set(guessed_letters),
            return_all_matches=True,
            verbose=False
        )
        pruned_affixes = agents.utils.filter_redundant_affixes(
            matched_affixes,
            agent.affix_data,
            verbose=False
        )
        affix_matches_all.append(pruned_affixes)

        g = guess_YS(curr_pattern, ''.join(sorted(guessed_letters)))
        try:
            strategy_used = agent.last_strategy_used
        except AttributeError:
            strategy_used = "unknown"
        if g in guessed_letters:
            incorrect_guesses += 1
            continue
        guessed_letters.append(g)

        if g in secret_word:
            for i, c in enumerate(secret_word):
                if c == g:
                    pattern[i] = g
        else:
            incorrect_guesses += 1

        guess_history.append({
            "guess": g,
            "pattern": ''.join(pattern),
            "strategy": strategy_used
        })

    success = ('_' not in pattern)

    if log_failures and not success:
        try:
            log_failure(
                word=secret_word,
                pattern=pattern,
                guessed_letters=guessed_letters,
                incorrect_guesses=incorrect_guesses,
                guess_history=guess_history,
                affix_matches=affix_matches_all,
                fallback_used=fallback_used_flag,
                strategy_used=strategy_used,
                source=source
            )
        except Exception as e:
            print(f"[Warning] Logging failed: {e}")

    return [success, guessed_letters]

###############################################
# Evaluate the agent on a set number of games
###############################################

def evaluate_agent(num_games: int = 1000, seed: int = 42) -> float:
    random.seed(seed)
    if agent.mode == "Transformer":
        source = "Transformer"
        test_words = random.sample(agent.short_word_list_eval, num_games)
    elif agent.mode in ["Hybrid_SB_Transformer", "Hybrid_SB_Transformer_V2"]:
        source = "Hybrid-Transformer"
        test_words = random.sample(test_dictionary, num_games)
        #test_dictionary_transformers(num_games=num_games) 
    else:
        source = "Non-Transformers"
        test_words = random.sample(test_dictionary, num_games)
    
    if not test_words:
        raise ValueError("Test words list is empty. Check the dictionary or the agent's mode.")        
    wins = 0

    print(f"Loaded {len(test_words)} test words from hybrid dictionary")
    print(f"\n=== Evaluating '{agent.mode}' strategy on {num_games} words ===\n")

    for i, word in enumerate(test_words, 1):
        #print(word)
        result, guessed_letters = play_game(word, source=source)
        last_guess = guessed_letters[-1] if guessed_letters else None
        was_correct = last_guess in word
        print(f"[Result] {'✔️' if was_correct else '❌'} Letter '{last_guess}' was {'correct' if was_correct else 'wrong'}")
        if result:
            wins += 1
            print(f"[Victory] The agent {agent.mode} successfully guessed the word '{word}'")
        else:
            print(f"[Defeat] The agent {agent.mode} failed to guess the word '{word}'")
        print(f"[{i}/{num_games}] Word: {word} "
              f"-> {'Win' if result else 'Lose'}"
              f" | Guessed Letters: {', '.join(guessed_letters)}\n")

    win_rate = wins / num_games
    print(f"\nFinal Win Rate: {win_rate:.2%}")
    return win_rate

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_games", type=int, default=1000, help="Number of games to run")
    args = parser.parse_args()

    evaluate_agent(num_games=args.n_games)
