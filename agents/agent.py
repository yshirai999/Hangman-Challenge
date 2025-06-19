import string
import random
import math
import multiprocessing as mp
from collections import Counter, defaultdict
import os
import json

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import ge, nn, optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.model_selection import StratifiedKFold

from agents.utils import (
    VOCAB_DICT,
    HangmanTransformerModel,
    HangmanTransformerModelV2,
    HangmanTransformerModelV3,
    encode_sequence,
    filter_redundant_affixes,
    fuzzy_affix_match,
    load_dictionary,
    load_cache,
    mask_logits,
    save_cache,
    data_path,
    model_path,
    results_path,
    VOCAB,
    letter_frequency_order,
    filter_candidates,
    filtered_candidates_features,
    load_model,
    save_model,
)


os.makedirs(results_path("logs"), exist_ok=True)
os.makedirs(results_path("logs/v3_eval"), exist_ok=True)


def process_sample(
        k,
        dictionary,
        label_map,
        cluster_summary_by_label,
        ordered_dictionary,
        max_word_len,
        incorrect_guesses_allowed,
        random_pattern_and_guesses,
        filtered_candidates_features,
        expected_length
):
    '''
    Process a single sample generation task for the hangman agent.
    '''
    word = dictionary[k]
    curr_pattern, guessed_letters = random_pattern_and_guesses(word, incorrect_guesses_allowed)
    features = filtered_candidates_features(curr_pattern, guessed_letters, label_map, cluster_summary_by_label, ordered_dictionary, max_word_len)
    if len(features) != expected_length:
        return None
    letter_counts = {}
    for c in word:
        if c not in guessed_letters:
            letter_counts[c] = letter_counts.get(c, 0) + 1
    if not letter_counts:
        return None
    label = max(letter_counts, key=letter_counts.get)
    return (features, label)

def random_pattern_and_guesses(word: str, incorrect_guesses_allowed: int) -> tuple[str, str]:
        """
        Generates a valid current pattern and guessed letters such that:
        - The word is not fully revealed.
        - The number of incorrect guesses is strictly less than allowed.
        - The number of correct guesses is strictly less than the word length.
        """
        word_letters = list(set(word))
        alphabet = list(string.ascii_lowercase)
        
        # Choose number of correct guesses (could be zero, but less than word length)
        max_correct = len(word_letters) - 1 if len(word_letters) > 1 else 1
        num_correct = random.randint(0, max_correct)

        # Choose number of incorrect guesses (could be zero, but less than allowed)
        num_incorrect = random.randint(0, incorrect_guesses_allowed - 1)

        correct_guesses = random.sample(word_letters, num_correct)
        incorrect_pool = list(set(alphabet) - set(word_letters))
        incorrect_guesses = random.sample(incorrect_pool, num_incorrect)

        guessed_letters = correct_guesses + incorrect_guesses

        # Build pattern string
        pattern = ''.join([c if c in correct_guesses else '_' for c in word])

        return pattern, ''.join(guessed_letters)


class HangmanAgent:
    MODES = ['Hybrid_nonML',
             'Hybrid_ML',
             'Hybrid_SB_Transformer',
             'Hybrid_SB_Transformer_V2',
             'Frequency',
             'Entropy',
             'Logical',
             'Subpattern_Greedy',
             'ML',
             'Transformer']

    def __init__(self,
             label_map=None,
             cluster_summary_by_label=None,
             n_estimators=200,
             max_depth=30,
             incorrect_guesses_allowed=6,
             sample_data_size=15000,
             stratified_sample_data_size=50000,
             transformer_sample_data_size=200000,
             transformer_subpattern_sample_data_size=25000,
             transformer_subpattern_short_sample_data_size=50000,
             use_parallel_data_generation=True,
             parallelize_data_from_subpattern_agent_generation=True,
             dictionary="train_dictionary.pkl",
             ordered_dictionary="ordered_train_dictionary.pkl",
             max_word_len=20,
             mode: str = 'Subpattern_Greedy',
             use_subpattern_data_for_transformer=True,
             use_short_words_for_transformer=False,
             TransformerModel=HangmanTransformerModelV3,
             max_len_for_transformer=None,
             load_fallback_order=True,
             transformer_to_sg_ratio_threshold=1.5
        ):
        """
        Initialize the HangmanAgent with the given parameters.
        """
        # Load dictionary and ordered dictionary
        if isinstance(dictionary, str):
            self.dictionary = load_dictionary(data_path(dictionary))
        else:
            self.dictionary = dictionary

        if isinstance(ordered_dictionary, str):
            self.ordered_dictionary = load_cache(data_path(ordered_dictionary))
        else:
            self.ordered_dictionary = ordered_dictionary
        
        # Set agent parameters
        self.max_word_len = max_word_len
        self.label_map = label_map
        self.cluster_summary_by_label = cluster_summary_by_label
        self.alphabet = list(string.ascii_lowercase)
        self.POS_CONSONANT_RANKS = [
            ['r', 't', 'n', 'l', 's'],
            ['r', 't', 'n', 'l', 's'],
            ['t', 'n', 'r', 'l', 's'],
            ['l', 't', 'h', 'r', 'n'],
            ['l', 'p', 'h', 't', 'c'],
            [], [], []
        ]
        self.consonants = [c for c in self.alphabet if c not in 'aeiou']
        self.incorrect_guesses_allowed = min(incorrect_guesses_allowed, 26)
        self.sample_data_size = sample_data_size
        self.stratified_sample_data_size = stratified_sample_data_size
        self.transformer_sample_data_size = transformer_sample_data_size
        self.transformer_subpattern_sample_data_size = transformer_subpattern_sample_data_size
        self.transformer_subpattern_short_sample_data_size = transformer_subpattern_short_sample_data_size
        self.use_subpattern_data_for_transformer = use_subpattern_data_for_transformer
        self.use_short_words_for_transformer = use_short_words_for_transformer
        self.load_fallback_order = load_fallback_order
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.use_parallel_data_generation = use_parallel_data_generation
        self.parallelize_data_from_subpattern_agent_generation = parallelize_data_from_subpattern_agent_generation
        self.vocab_size = len(VOCAB)
        self.TransformerModel = TransformerModel # Placeholder for the Transformer model class
        self.transformer_model = None # Placeholder for the trained Transformer model
        self.transformer_to_sg_ratio_threshold = transformer_to_sg_ratio_threshold
        self.affix_data = {}
        if not max_len_for_transformer:
            self.max_len_for_transformer = 11 if TransformerModel == HangmanTransformerModelV3 else 20
        TransformerModel.max_seq_len = self.max_len_for_transformer
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {self.MODES}")
        self.mode = mode
        # Try to load the model
        if self.label_map and self.cluster_summary_by_label:
            print("Loading Random Forest model...")
            self.feature_length = self.compute_feature_length()
            try:
                self.model = load_model(model_path(
                    f"hangman_model_{self.n_estimators}_"
                    f"{self.max_depth}_"
                    f"{self.sample_data_size}.pkl"
                ))
                self.model_loaded = True
                print("Model loaded successfully.")
            except FileNotFoundError:
                print("Model not found. " \
                "Please train a new model using the train_model() method.")
                self.model = RandomForestClassifier(
                    n_estimators=self.n_estimators,
                    random_state=42,
                    verbose=1,
                    n_jobs=-1,
                    max_depth=self.max_depth
                )
                self.model_loaded = False
        if self.load_fallback_order:
            # Try to load fallback order
            try:
                self.fallback_order = load_cache(data_path(f"order_letter_{dictionary}"))
            except FileNotFoundError:
                self.fallback_order = letter_frequency_order(self.dictionary)
                save_cache(self.fallback_order, data_path(f"order_letter_{dictionary}"))
            print(f"Fallback order loaded successfully")

    ################################################
    ################################################
    ## Callable and hybrid strategies
    ################################################
    ################################################

    def __call__(self, curr_pattern, guessed_letters):
        if self.mode == 'Hybrid_nonML':
            return self.hybrid_nonML(curr_pattern, guessed_letters)
        if self.mode == 'Hybrid_ML':
            return self.hybrid_ML(curr_pattern, guessed_letters)
        elif self.mode == 'Hybrid_SB_Transformer':
            return self.hybrid_SB_Transformer(curr_pattern, guessed_letters)
        elif self.mode == 'Hybrid_SB_Transformer_V2':
            return self.hybrid_SB_Transformer_V2(curr_pattern, guessed_letters)
        elif self.mode == 'Frequency':
            return self.frequency_prediction(curr_pattern, guessed_letters)
        elif self.mode == 'Entropy':
            return self.entropy_prediction(curr_pattern, guessed_letters)
        elif self.mode == 'Logical':
            return self.logical_prediction(curr_pattern, guessed_letters)
        elif self.mode == 'Subpattern_Greedy':
            return self.greedy_structure_match_prediction(curr_pattern, guessed_letters)
        elif self.mode == 'ML':
            return self.ML_prediction(curr_pattern, guessed_letters)
        elif self.mode == 'Transformer':
            return self.transformer_prediction(curr_pattern,
                    guessed_letters,
                    use_blank_mask=(self.TransformerModel == HangmanTransformerModelV3))

    def hybrid_nonML(self, curr_pattern, guessed_letters):
        guessed_letters = set(guessed_letters)
        n_known = sum(c != '_' for c in curr_pattern)
        proportion_known = n_known / len(curr_pattern)

        remaining_guesses = self.incorrect_guesses_allowed - len([
            g for g in guessed_letters if g not in curr_pattern
        ])

        # Phase 1: Subpattern
        if proportion_known < 0.7:
            return self.greedy_structure_match_prediction(curr_pattern, guessed_letters)

        # Phase 2: Late-game switch
        if proportion_known >= 0.7 and remaining_guesses <= 2:
            letter = self.positional_consonant_prediction(curr_pattern, guessed_letters)
            print(f"[Strategy] POS-CONSONANT | Guess: '{letter}'")
            return letter

        # Phase 3: Fallback (logical)
        letter = self.logical_prediction(curr_pattern, guessed_letters)
        print(f"[Strategy] LOGIC-FALLBACK | Guess: '{letter}'")
        return letter
    
    def hybrid_ML(self, curr_pattern, guessed_letters):
        '''
        Hybrid strategy that chooses between entropy, frequency, and logical elimination
        based on the number of known letters in the current pattern.
        '''
        n_known = sum(c != '_' for c in curr_pattern)
        if n_known < 1 and len(guessed_letters) < 2:
            return self.ML_prediction(curr_pattern, guessed_letters)
        else:
            return self.logical_prediction(curr_pattern, guessed_letters)
        
    def hybrid_SB_Transformer(self, curr_pattern, guessed_letters):
        """
        Hybrid strategy that chooses between subpattern (logical) and Transformer agent
        based on % of known letters and length of the secret word.
        
        - If <60% of the word is known, use logical (subpattern) strategy
        - If ≥60% is known AND word length ∈ [5, 11], switch to Transformer
        - If word is outside this range, always use logical strategy
        """

        total_len = len(curr_pattern)
        n_known = sum(c != '_' for c in curr_pattern)
        frac_known = n_known / total_len

        # Use Transformer only if enough is known AND word is in desired length range
        if 5 <= total_len <= 11 and frac_known >= 0.6:
            return self.transformer_prediction(curr_pattern, guessed_letters, use_blank_mask=True)
        else:
            return self.greedy_structure_match_prediction(curr_pattern, guessed_letters)
        
    def hybrid_SB_Transformer_V2(self, curr_pattern, guessed_letters, verbose=False):
        """
        Hybrid strategy that chooses between subpattern (logical) and Transformer agent
        based on % of known letters and length of the secret word.
        
        - If subpattern is confident, use subpattern strategy
        - If transformer is confident, use Transformer strategy
        - If word is outside this range, always use logical strategy
        """

        word_len = len(curr_pattern)

        if word_len < 5 or word_len > 10:
            # Transformer not trained on this length — fall back to subpattern
            return self.greedy_structure_match_prediction(curr_pattern, guessed_letters)

        sg_letter, sg_probs = self.greedy_structure_match_prediction(curr_pattern, guessed_letters, return_probs=True)
        tf_letter, tf_probs = self.transformer_prediction(curr_pattern, guessed_letters, return_probs=True)
        # Extract maximum confidence scores
        sg_max_prob = max((prob for pos_dict in sg_probs.values() for prob in pos_dict.values()), default=1e-6)
        tf_max_prob = max((prob for pos_dict in tf_probs.values() for prob in pos_dict.values()), default=1e-6)

        ratio = tf_max_prob / sg_max_prob if sg_max_prob > 0 else float('inf')

        if verbose:
            print(f"[HybridV2] TF max={tf_max_prob:.4f}, SG max={sg_max_prob:.4f}, R={ratio:.2f}")
        # Threshold hyperparameter (tune this!)
        THRESHOLD = self.transformer_to_sg_ratio_threshold
        return tf_letter if ratio > THRESHOLD else sg_letter

    ################################################
    ################################################
    ## Non ML Predictions
    ################################################
    ################################################

    def fallback_letter(self, guessed_letters):
        '''
        Returns the next letter in the fallback order that has not been guessed yet.
        If all letters in the fallback order have been guessed, returns a random
        unguessed letter from the alphabet.
        '''

        self.last_strategy_used = "fallback"

        for letter in self.fallback_order:
            if letter not in guessed_letters:
                return letter
        return random.choice([l for l in self.alphabet
                               if l not in guessed_letters])

    def frequencies(self, curr_pattern, guessed_letters):
        '''
        Returns a dictionary of letter frequencies among the candidate words
        that match the current pattern and guessed letters.
        '''
        candidates = filter_candidates(curr_pattern,
                            guessed_letters,
                            self.ordered_dictionary)
        if not candidates:
            return None
        letter_frequencies = {}
        guessed_letters = set(guessed_letters)
        for word in candidates:
            for letter in set(word) - guessed_letters:
                letter_frequencies[letter] = letter_frequencies.get(letter, 0) + 1
        return letter_frequencies

    def frequency_prediction(self, curr_pattern, guessed_letters):
        '''
        Returns the letter with the highest frequency among the candidate words
        '''

        self.last_strategy_used = "frequency"

        # Get the letter frequencies based on the current pattern and guessed letters
        letter_frequencies = self.frequencies(curr_pattern, guessed_letters)
        if not letter_frequencies:
            return self.fallback_letter(guessed_letters)
        # Choose the letter with the highest frequency
        best_letter = max(letter_frequencies, key=letter_frequencies.get)
        return best_letter

    def entropy_prediction(self, curr_pattern, guessed_letters):
        '''
        Returns the letter with the highest entropy among the candidate words
        that match the current pattern and guessed letters.
        '''

        self.last_strategy_used = "entropy"

        candidates = filter_candidates(curr_pattern,
                            guessed_letters,
                            self.ordered_dictionary)
        if not candidates:
            return self.fallback_letter(guessed_letters)
        total_candidates = len(candidates)
        letter_entropies = {}
        guessed_letters = set(guessed_letters)
        letter_counts = Counter()
        for word in candidates:
            for letter in set(word):
                if letter not in guessed_letters:
                    letter_counts[letter] += 1
        for letter, count in letter_counts.items():
            p = count / total_candidates
            if p <= 0 or p >= 1:
                entropy = 0
            else:
                entropy = -p * math.log2(p) - (1 - p) * math.log2(1 - p)
            letter_entropies[letter] = entropy
        if not letter_entropies:
            return self.fallback_letter(guessed_letters)
        best_letter = max(letter_entropies, key=letter_entropies.get)
        return best_letter

    def logical_prediction(self, curr_pattern, guessed_letters):
        '''
        Returns the letter that appears most frequently in a fixed position
        among the candidate words that match the current pattern and guessed letters.
        '''
        candidates = filter_candidates(curr_pattern,
                            guessed_letters,
                            self.ordered_dictionary)
        position_frequencies = [{} for _ in curr_pattern]
        for word in candidates:
            for i, char in enumerate(word):
                if curr_pattern[i] == '_':
                    position_frequencies[i][char] = position_frequencies[i].get(char, 0) + 1
        # Find the most frequent letter across all positions, unguessed
        best_letter = self.fallback_letter(guessed_letters)
        self.last_strategy_used = "logic" #Must be set after fallback_letter call
        max_count = -1
        for i, freq_dict in enumerate(position_frequencies):
            for letter, count in freq_dict.items():
                if letter not in guessed_letters and count > max_count:
                    best_letter = letter
                    max_count = count  
        return best_letter

    def greedy_structure_match_prediction(self,
                                          curr_pattern,
                                          guessed_letters,
                                          verbose=False,
                                          return_probs=False):
        guessed_letters = set(guessed_letters)

        self.last_strategy_used = "vowel-affix"
        n_vowels_known = sum(c in 'aeiou' for c in curr_pattern)
        if n_vowels_known < 2:
            vowel_freq = self.vowel_affix_prediction(guessed_letters, verbose=verbose)
            if vowel_freq:
                if not return_probs:
                    return vowel_freq
                else:
                    prob_dict = defaultdict(dict)
                    for i, ch in enumerate(curr_pattern):
                        if ch == '_':
                            prob_dict[i][vowel_freq] = 1.0
                    return vowel_freq, prob_dict
                
        self.last_strategy_used = "subpattern_greedy"
        matched_affixes = fuzzy_affix_match(
            pattern=curr_pattern,
            affix_dicts=self.affix_data,
            guessed_letters=guessed_letters,
            return_all_matches=True
        )

        if matched_affixes:
            pruned_affixes = filter_redundant_affixes(
                matched_affixes,
                self.affix_data,
                verbose=verbose
            )

            letter_scores = Counter()
            for affix in pruned_affixes:
                for letter in set(affix) - guessed_letters:
                    letter_scores[letter] += self.affix_data.get(affix, 1)

            if letter_scores:
                best_letter = letter_scores.most_common(1)[0][0]

                if verbose:
                    print(f"[Strategy] AFFIX-MATCH | Guess: '{best_letter}'")
                  
                if return_probs:
                    total = sum(letter_scores.values()) + 1e-6
                    prob_dict = defaultdict(dict)
                    for ltr, score in letter_scores.items():
                        for pos, ch in enumerate(curr_pattern):
                            if ch == '_' and ltr in self.affix_dict_positions(pruned_affixes, pos):
                                prob_dict[pos][ltr] = score / total
                    return best_letter, prob_dict

                return best_letter

        fallback_letter = self.affix_logical_prediction(curr_pattern, guessed_letters, verbose=verbose)
        if fallback_letter is None:
            fallback_letter = self.fallback_letter(guessed_letters)

        if return_probs:
            prob_dict = defaultdict(dict)
            for i, ch in enumerate(curr_pattern):
                if ch == '_':
                    prob_dict[i][fallback_letter] = getattr(self, 'fallback_prob', 0.01)  # default fallback prob
            return fallback_letter, prob_dict

        return fallback_letter


    # Helper to extract which letters appear at each position in affix set
    def affix_dict_positions(self, affixes, position):
        result = set()
        for word in affixes:
            if len(word) > position:
                result.add(word[position])
        return result

    def affix_logical_prediction(self, curr_pattern, guessed_letters, verbose=False):
        """
        Uses all affixes and aligns them with the pattern.
        At each open position ('_'), collects letter frequencies across all matching affixes.
        """

        self.last_strategy_used = "affix-logic-fallback"

        pattern_str = ''.join(curr_pattern)
        guessed_letters = set(guessed_letters)
        pos_letter_counts = defaultdict(Counter)

        affix_dict = self.affix_data  # Already consolidated

        for affix, freq in affix_dict.items():
            if len(affix) > len(pattern_str):
                continue  # skip if affix longer than pattern

            for i in range(len(pattern_str) - len(affix) + 1):
                match = True
                for j in range(len(affix)):
                    p_char = pattern_str[i + j]
                    if p_char != '_' and p_char != affix[j]:
                        match = False
                        break

                if match:
                    for j in range(len(affix)):
                        p_idx = i + j
                        if pattern_str[p_idx] == '_' and affix[j] not in guessed_letters:
                            pos_letter_counts[p_idx][affix[j]] += freq
                    break  # avoid double-counting same affix

        if pos_letter_counts:
            all_scores = Counter()
            for counter in pos_letter_counts.values():
                all_scores.update(counter)

            best_letter = all_scores.most_common(1)[0][0]
            if verbose:
                print(f"[Strategy] AFFIX-LOGIC | Guess: '{best_letter}'")
            return best_letter

        if verbose:
            print("[Strategy] AFFIX-LOGIC | No match found — fallback to LOGIC")
        return self.logical_prediction(curr_pattern, guessed_letters)

    def vowel_affix_prediction(self, guessed_letters, verbose=False):

        self.last_strategy_used = "vowel-affix"

        guessed_letters = set(guessed_letters)
        vowels = {'a', 'e', 'i', 'o', 'u'}
        vowel_counter = Counter()

        for affix, freq in self.affix_data.items():
            for v in vowels:
                if v not in guessed_letters and v in affix:
                    vowel_counter[v] += freq

        if verbose:
            print("[DEBUG] Vowel frequencies in affix dict (unguessed):", dict(vowel_counter))

        if vowel_counter:
            best = vowel_counter.most_common(1)[0][0]
            if verbose:
                print(f"[Strategy] AFFIX-VOWEL | Guess: '{best}'")
            return best
        else:
            if verbose:
                print("[Strategy] AFFIX-VOWEL | No valid vowels found, fallback")
            return self.positional_consonant_prediction("_" * 5, guessed_letters)

    def positional_consonant_prediction(self, curr_pattern, guessed_letters):

        self.last_strategy_used = "pos_consonant"

        guessed = set(guessed_letters)
        candidates = []

        for i, c in enumerate(curr_pattern):
            if c == '_' and i < len(self.POS_CONSONANT_RANKS):
                for con in self.POS_CONSONANT_RANKS[i]:
                    if con not in guessed:
                        candidates.append((con, i))
                        break

        if not candidates:
            for con in ['t', 'n', 's', 'r', 'l']:
                if con not in guessed:
                    return con
            self.last_strategy_used = "logic-fallback"
            return self.logical_prediction(curr_pattern, guessed_letters)

        candidates.sort(key=lambda x: x[1])
        return candidates[0][0]

    ################################################
    ################################################
    ## Features Engineering for ML Predictions
    ################################################
    ################################################

    def compute_feature_length(self):
        # Use a sample word and guessed letters
        sample_word = self.dictionary[0]
        curr_pattern, guessed_letters = random_pattern_and_guesses(sample_word, self.incorrect_guesses_allowed)
        features = filtered_candidates_features(
            curr_pattern,
            guessed_letters,
            self.label_map,
            self.cluster_summary_by_label,
            self.ordered_dictionary,
            self.max_word_len
        )
        return len(features)

    def data(self):
        """
        Generate training data (i.e. candidates features) for the random forest model.
        This function will create self.sample_data, 
        which is a list of tuples (features, label).
        """
        try:
            self.sample = load_cache(data_path(
                f"sample_data{self.sample_data_size}.pkl"
            ))
            self.training_data = load_cache(
                data_path(f"training_data{self.sample_data_size}.pkl")
            )
            self.validation_data = load_cache(
                data_path(f"validation_data{self.sample_data_size}.pkl")
            )
            self.testing_data = load_cache(
                data_path(f"testing_data{self.sample_data_size}.pkl")
            )
        except FileNotFoundError:
            print("One or more data files not found. Generating new data...")
            if self.use_parallel_data_generation:
                print("Using parallel data generation.")
                self.parallel_data_generation()
            else:
                print("Using sequential data generation.")
                sample_data = []
                n = len(self.dictionary)
                if self.sample_data_size <= n:
                    idx = np.random.choice(n,
                            size=self.sample_data_size,
                            replace=False
                            )
                else:
                    # fallback to with replacement
                    idx = np.random.randint(0, n, size=self.sample_data_size)  
                random.shuffle(idx)  # Shuffle indices to ensure randomness
                for k in idx:
                    result = process_sample(
                        k,
                        self.dictionary,
                        self.label_map,
                        self.cluster_summary_by_label,
                        self.ordered_dictionary,
                        self.max_word_len,
                        self.incorrect_guesses_allowed,
                        random_pattern_and_guesses,
                        filtered_candidates_features,
                        self.feature_length)
                    if result is not None:
                        sample_data.append(result)
                self.split_and_save_data(sample_data)

    def parallel_data_generation(self):
        n = len(self.dictionary)
        if self.sample_data_size <= n:
            idx = np.random.choice(n,
                                   size=self.sample_data_size,
                                   replace=False
            )
        else:
            idx = np.random.randint(0, n, size=self.sample_data_size)
        np.random.shuffle(idx)
        args = [(k,
                 self.dictionary,
                 self.label_map,
                 self.cluster_summary_by_label,
                 self.ordered_dictionary,
                 self.max_word_len,
                 self.incorrect_guesses_allowed,
                 random_pattern_and_guesses,
                 filtered_candidates_features,
                 self.feature_length) for k in idx]
        with mp.Pool(processes=mp.cpu_count()) as pool:
            sample_data = pool.starmap(process_sample, args)
        sample_data = [data for data in sample_data if data is not None]
        self.split_and_save_data(sample_data)

    def split_and_save_data(self, sample_data):
        '''
        Split sample_data in training, validation, and testing sets,
        and save them to disk.
        '''
        # Remove None entries
        sample_data = [data for data in sample_data if data is not None]
        # Split the sample data into training, validation, and testing sets
        self.training_data, self.testing_data = train_test_split(
            sample_data, test_size=0.2, random_state=42)
        self.validation_data, self.testing_data = train_test_split(
            self.testing_data, test_size=0.5, random_state=42)
        self.sample_data = sample_data
        # Save the sample data
        save_cache(self.sample_data,
                   data_path(f"sample_data{self.sample_data_size}.pkl"))
        save_cache(self.training_data,
                   data_path(f"training_data{self.sample_data_size}.pkl"))
        save_cache(self.validation_data,
                   data_path(f"validation_data{self.sample_data_size}.pkl"))
        save_cache(self.testing_data,
                   data_path(f"testing_data{self.sample_data_size}.pkl"))
        print(f"Sample data of size {len(self.sample_data)} generated.")
        print(f"Training data of size {len(self.training_data)} generated.")
        print(f"Validation data of size {len(self.validation_data)} generated.")
        print(f"Testing data of size {len(self.testing_data)} generated.")

    ################################################
    ################################################
    ## Sample Generation for Late-Stage Hangman
    ################################################
    ################################################

    def generate_late_stage_sample(self, 
                    word,
                    min_reveal_ratio=0.6,
                    max_reveal_ratio=0.9,
                    max_incorrect=5,
                    seed=None):
        """
        Generate supervised learning sample for training a late-stage word prediction model.

        The sample is a tuple (pattern, guessed_letters, full_word), where the pattern hides
        10-40% of the letters, guessed_letters includes both correct and up to `max_incorrect`
        incorrect letters, and full_word is the ground-truth label.

        Parameters:
            words (list of str): Source words to sample from.
            min_reveal_ratio (float): Minimum proportion of letters to reveal.
            max_reveal_ratio (float): Maximum proportion of letters to reveal.
            max_incorrect (int): Max number of incorrect letters to add.
            seed (int or None): For reproducibility.

        Returns:
            List of (pattern, guessed_letters, full_word) triples.
        """

        if seed is not None:
            random.seed(seed)

        word = word.lower()
        unique_letters = list(set(word))
        total_letters = len(word)

        min_reveal = max(1, int(min_reveal_ratio * total_letters))
        max_reveal = min(len(unique_letters), int(max_reveal_ratio * total_letters))

        if max_reveal < min_reveal:
            return None # Handle edge case of, say, 5-letter words with multiple repeats

        num_reveal = random.randint(min_reveal, max_reveal)
        correct_guesses = random.sample(unique_letters, num_reveal)

        pattern = ''.join([c if c in correct_guesses else '_' for c in word])

        incorrect_pool = [c for c in string.ascii_lowercase if c not in word]
        num_incorrect = random.randint(0, max_incorrect)
        incorrect_guesses = random.sample(incorrect_pool, num_incorrect)

        guessed_letters = ''.join(sorted(set(correct_guesses + incorrect_guesses)))

        return (pattern, guessed_letters, word)

    def generate_late_stage_samples_from_words(self,
                    words: list[str],
                    num_samples: int = 200000
        ) -> list[tuple[str, str, str]]:
        """
        Generate late-stage samples from the input word list.

        Each sample consists of:
        - pattern: the current partially guessed word (≥60% revealed)
        - guessed_letters: all letters guessed so far (correct + up to 3 incorrect)
        - label: the next correct letter to guess

        Returns a list of (pattern, guessed_letters, label) triples.
        """
        samples = []
        while len(samples) < num_samples:
            word = random.choice(words)
            result = self.generate_late_stage_sample(word)
            if result:
                samples.append(result)
        return samples
    
    def generate_full_games_from_subpattern_agent(
        self,
        word_list,
        num_samples=50000,
        min_reveal_fraction=0.6,
        max_attempts_per_sample=10,
    ):
        import random

        samples = []
        total_attempts = 0
        max_attempts = num_samples * max_attempts_per_sample

        while len(samples) < num_samples and total_attempts < max_attempts:
            word = random.choice(word_list)
            curr_pattern = ['_' for _ in word]
            guessed_letters = set()
            incorrect_guesses = 0
            snapshots = []

            while incorrect_guesses < self.incorrect_guesses_allowed and '_' in curr_pattern:
                letter = self.greedy_structure_match_prediction(curr_pattern, guessed_letters)
                if letter is None or letter in guessed_letters:
                    break

                guessed_letters.add(letter)

                if letter in word:
                    for i, c in enumerate(word):
                        if c == letter:
                            curr_pattern[i] = letter
                else:
                    incorrect_guesses += 1

                # Early exit if no letters revealed in first 4 guesses
                if sum(c != '_' for c in curr_pattern) == 0 and len(guessed_letters) >= 4:
                    break

                # Save snapshot if reveal is between 60% and <100%
                revealed_count = sum(c != '_' for c in curr_pattern)
                if int(min_reveal_fraction * len(word)) <= revealed_count < len(word):
                    snapshots.append((''.join(curr_pattern), ''.join(sorted(guessed_letters)), word))

            if snapshots:
                samples.append(random.choice(snapshots))

            total_attempts += 1
            if total_attempts % 5000 == 0:
                print(f"[...] Attempts: {total_attempts} | Samples: {len(samples)}")

        print(f"\nFinished generating transformer samples")
        print(f"Final count: {len(samples)} samples from {total_attempts} games")
        print(f"Hit rate: {len(samples)/total_attempts:.2%}\n")

        return samples

    ################################################
    ################################################
    ## ML and Transformer Prediction
    ################################################
    ################################################

    def ML_prediction(self, curr_pattern, guessed_letters):
        # Get the feature vector for the current pattern and guessed letters
        feature_vector = filtered_candidates_features(curr_pattern,
                                                guessed_letters,
                                                self.label_map,
                                                self.cluster_summary_by_label,
                                                self.ordered_dictionary,
                                                self.max_word_len)
        # Predict the letter using the trained model
        proba = self.model.predict_proba(np.array(feature_vector).reshape(1, -1))[0]
        best_letter = max(
            (letter for idx, letter in enumerate(self.alphabet) if letter not in guessed_letters),
            key=lambda l: proba[self.alphabet.index(l)],
            default=None
        )
        return best_letter

    def transformer_prediction(self, curr_pattern, guessed_letters, return_probs=False, use_blank_mask=True):
        """
        Predict the next letter using the transformer model.
        If `return_probs` is True, returns (best_letter, prob_dict_by_position).
        """
        if not self.transformer_model:
            return None if not return_probs else (None, {})

        device = next(self.transformer_model.parameters()).device

        pattern_encoded = encode_sequence(seq=curr_pattern, max_len=self.max_len_for_transformer)
        guessed_encoded = encode_sequence(seq=guessed_letters, max_len=self.max_len_for_transformer)

        pattern_tensor = torch.tensor(pattern_encoded, dtype=torch.long).unsqueeze(0).to(device)
        guess_tensor = torch.tensor(guessed_encoded, dtype=torch.long).unsqueeze(0).to(device)

        if use_blank_mask:
            blank_mask = [1 if idx == VOCAB_DICT['_'] else 0 for idx in pattern_encoded]
            blank_mask_tensor = torch.tensor(blank_mask + [0] * (self.max_len_for_transformer - len(blank_mask)),
                                             dtype=torch.long, device=device).unsqueeze(0)
            logits = self.transformer_model(pattern_tensor, guess_tensor, blank_mask_tensor)
        else:
            logits = self.transformer_model(pattern_tensor, guess_tensor)

        masked_logits = mask_logits(logits, set(guessed_letters))
        probs = F.softmax(masked_logits, dim=-1).squeeze(0)  # shape: (L, vocab)

        best_letter = None
        best_score = -1
        prob_dict = defaultdict(dict)

        for pos, char in enumerate(curr_pattern):
            if char == '_':
                for idx in range(len(VOCAB)):
                    c = VOCAB[idx]
                    if c in guessed_letters or c in curr_pattern or c not in self.consonants:
                        continue
                    score = probs[pos, idx].item()
                    prob_dict[pos][c] = score
                    if score > best_score:
                        best_score = score
                        best_letter = c
        
        return (best_letter, prob_dict) if return_probs else best_letter

    ################################################
    ################################################
    ## Training and Validation Methods for ML Model
    ################################################
    ################################################

    def train_model(self):
        """
        Train the Random Forest model using the current training data.
        Saves the trained model to disk.
        """
        if not hasattr(self, 'training_data') or not self.training_data:
            print("No training data available." \
            " Please ensure the model is trained with data.")
            print("Generate sample data first.")
            return
        X_train = np.array([data[0] for data in self.training_data])
        y_train = np.array([data[1] for data in self.training_data])
        self.model.fit(X_train, y_train)
        # save the model after training
        save_model(self.model,
                        f"hangman_model_{self.n_estimators}_"
                        f"{self.max_depth}"
                        f"_{self.sample_data_size}.pkl"
        )

    def validate_model(self):
        """
        Validate the model using the provided validation data.
        The validation data should be a list of tuples (features, label).
        """
        if not hasattr(self, 'validation_data') or not self.validation_data:
            print("No validation data available." \
            " Please ensure the model is trained with validation data.")
            print("Generate sample data first.")
            return
        validation_data = self.validation_data
        X_val = np.array([data[0] for data in validation_data])
        y_val = np.array([data[1] for data in validation_data])
        accuracy = self.model.score(X_val, y_val)
        print(f"Validation Accuracy: {accuracy * 100:.2f}%")

    def plot_learning_curve(self):
        if not hasattr(self, 'sample_data') or not self.sample_data:
            print("No sample data available for plotting learning curve.")
            print("Generate sample data first.")
            return
        X = np.array([data[0] for data in self.sample_data])
        y = np.array([data[1] for data in self.sample_data])
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        train_sizes, train_scores, val_scores = learning_curve(
            self.model, X, y, cv=cv, scoring='accuracy',
            train_sizes=np.linspace(0.1, 1.0, 10)
        )
        train_mean = 1 - np.mean(train_scores, axis=1)
        val_mean = 1 - np.mean(val_scores, axis=1)
        plt.plot(train_sizes, train_mean, label='Training Error')
        plt.plot(train_sizes, val_mean, label='Validation Error')
        plt.xlabel('Training Set Size')
        plt.ylabel('Error')
        plt.legend()
        plt.title('Learning Curve')
        plt.show()

    ################################################
    ################################################
    ## Transformer Model Training
    ################################################
    ################################################

    def train_transformer_model(
        self,
        model,
        epochs=5,
        batch_size=256,
        lr=1e-3,
        save_path="transformer_model.pt",
        eval_data_path="transformer_eval_data_from_subpattern_short.pkl",
        checkpoint_dir=None,
        use_cuda=True,
        use_blank_mask=False
    ):
        """
        Train a transformer model using cross-entropy over unrevealed (blank) letters only.

        Args:
            model: Transformer model (nn.Module)
            epochs: Number of epochs
            batch_size: Batch size
            lr: Learning rate
            save_path: File to save the model
            use_cuda: Use GPU if available
        """
        if self.torch_dataset is None:
            raise ValueError("torch_dataset is not set. Run build_torch_dataset() first.")

        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        if checkpoint_dir:
            os.makedirs(checkpoint_dir, exist_ok=True)

        device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        model = model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss(ignore_index=-100)

        loader = DataLoader(
            self.torch_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True
        )

        model.train()
        best_loss = float('inf')
        for epoch in range(epochs):
            total_loss = 0
            total_blanks = 0

            pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}")
            for batch in pbar:
                inputs, word_tensor = batch

                if use_blank_mask:
                    pattern_input, guess_input, blank_mask = inputs
                else:
                    pattern_input, guess_input = inputs
                    blank_mask = None

                pattern_input = pattern_input.to(device)
                guess_input = guess_input.to(device)
                word_tensor = word_tensor.to(device)
                if use_blank_mask:
                    blank_mask = blank_mask.to(device)

                optimizer.zero_grad()
                if use_blank_mask:
                    logits = model(pattern_input, guess_input, blank_mask)
                else:
                    logits = model(pattern_input, guess_input)

                # Build target tensor: only predict letters at blank positions
                mask = (pattern_input == VOCAB_DICT['_'])  # blanks in pattern
                target = word_tensor.clone()
                target[~mask] = -100  # ignore revealed positions

                loss = criterion(logits.view(-1, logits.size(-1)), target.view(-1))
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                total_blanks += mask.sum().item()
                pbar.set_postfix(loss=loss.item())

            avg = total_loss / len(loader)
            print(f"Epoch {epoch+1} avg loss: {avg:.4f} | Avg blanks per batch: {total_blanks / len(loader):.2f}")

            # Save checkpoint every N epochs
            if (epoch + 1) % 10 == 0:
                ckpt_path = os.path.join(checkpoint_dir, f"transformer_epoch{epoch+1}.pt")
                torch.save(model.state_dict(), ckpt_path)
                print(f"Saved checkpoint: {ckpt_path}")

                # Run evaluation and save detailed log
                eval_log_path = results_path(f"logs/transformer_eval_epoch{epoch+1}.jsonl")
                self.evaluate_transformer_prediction(
                    eval_data_path=eval_data_path,
                    log_path=eval_log_path,
                    n_samples=1000,
                    use_blank_mask_for_transformer=True
                )
            
            # Optional: Save best model
            if avg < best_loss:
                best_loss = avg
                torch.save(model.state_dict(), os.path.join(checkpoint_dir, "transformer_best.pt"))

        torch.save(model.state_dict(), save_path)
        print(f"Model saved to {save_path}")

    def load_transformer_model(self,
            model_class,
            model_path="transformer_model.pt",
            use_cuda=True
    ):
        """
        Load a transformer model from disk.

        Args:
            model_class: Class of the transformer model (subclass of nn.Module)
            model_path: Path to the saved model file
            use_cuda: Use GPU if available

        Returns:
            Loaded model
        """
        model_path = model_path
        device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
        model = model_class(
            self.vocab_size,
            max_seq_len=self.max_len_for_transformer).to(device)  # assumes self.vocab_size is defined

        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        print("Model max sequence length set to:", model.max_seq_len)
        print(f"Transformer model loaded from {model_path}")
        return model

    def evaluate_transformer_prediction(self,
                                        eval_data_path=results_path("transformer_eval_data_from_subpattern.pkl"),
                                        log_path=results_path("logs/transformer_eval_with_letter_guesses.jsonl"),
                                        n_samples=1000,
                                        use_blank_mask_for_transformer=False,
                                        n_eval=None):
        """
        Evaluates transformer predictions and logs:
        - Whether the predicted letter helps recover blanks
        - Accuracy on blank positions
        - JSONL log of each sample
        """

        def compute_blank_accuracy(letter, true_word, pattern):
            correct = 0
            total = 0
            for i in range(len(true_word)):
                if pattern[i] == '_':
                    total += 1
                    if true_word[i] == letter:
                        correct += 1
            return correct, total

        try:
            eval_data = load_cache(eval_data_path)
            print(f"[✓] Loaded {len(eval_data)} samples from {eval_data_path}")
        except FileNotFoundError:
            print(f"[!] No eval data found — generating {n_samples} new samples.")
            eval_data = self.generate_late_stage_samples_from_words(
                self.test_dictionary,
                num_samples=n_samples
            )
            save_cache(eval_data, eval_data_path)

        model = self.transformer_model
        model.eval()
        device = next(model.parameters()).device

        total_words = len(eval_data)
        total_recovered = 0
        total_blanks = 0
        samples_high_recovery = 0

        with open(log_path, "w") as f:
            for i, (pattern, guessed, true_word) in enumerate(eval_data):

                pattern_encoded = encode_sequence(seq=pattern, max_len=self.max_len_for_transformer)
                guessed_encoded = encode_sequence(seq=guessed, max_len=self.max_len_for_transformer)

                pattern_tensor = torch.tensor(pattern_encoded, dtype=torch.long).unsqueeze(0).to(device)
                guess_tensor = torch.tensor(guessed_encoded, dtype=torch.long).unsqueeze(0).to(device)

                if use_blank_mask_for_transformer:
                    blank_mask = [1 if idx == VOCAB_DICT['_'] else 0 for idx in pattern_encoded]
                    blank_mask_tensor = torch.tensor(blank_mask
                                             + [0] * (self.max_len_for_transformer
                                                       - len(blank_mask)),
                                             dtype=torch.long,
                                             device=device).unsqueeze(0)
                    logits = model(pattern_tensor,
                                   guess_tensor,
                                   blank_mask_tensor)
                else:
                    logits = model(pattern_tensor, guess_tensor)

                masked_logits = mask_logits(logits, set(guessed))

                probs = masked_logits.softmax(dim=-1)  # (1, L, vocab)
                probs = probs.squeeze(0)  # (L, vocab)

                best_letter = None
                best_score = -1

                for pos, char in enumerate(pattern):
                    if char == '_':
                        for idx in range(len(VOCAB)):
                            c = VOCAB[idx]
                            if c not in guessed and c not in pattern:
                                score = probs[pos, idx].item()
                                if score > best_score:
                                    best_score = score
                                    best_letter = c
                if n_eval is not None and i >= n_eval:
                    return
                recovered, blanks = compute_blank_accuracy(best_letter, true_word, pattern)
                percent_recovered = recovered / max(blanks, 1)
                total_recovered += recovered
                total_blanks += blanks
                if percent_recovered >= 0.8:
                    samples_high_recovery += 1
                
                # Extract position-wise scores for all consonants
                consonant_scores = {}
                for pos, char in enumerate(pattern):
                    if char == '_':
                        scores_at_pos = {}
                        for idx in range(len(VOCAB)):
                            c = VOCAB[idx]
                            if c in 'bcdfghjklmnpqrstvwxyz':
                                scores_at_pos[c] = probs[pos, idx].item()
                        consonant_scores[pos] = scores_at_pos

                record = {
                    "pattern": pattern,
                    "guessed_letters": guessed,
                    "true_word": true_word,
                    "pred_word": "Never predicted for transformer",
                    "predicted_letter": best_letter,
                    "blanks_recovered": recovered,
                    "total_blanks": blanks,
                    "letter_accuracy": percent_recovered,
                    "correct": (recovered == blanks),
                    "consonant_scores": consonant_scores
                }

                f.write(json.dumps(record) + "\n")

                if i < (n_eval if n_eval else 5):
                    print(f"\n[Sample {i+1}]")
                    print(f"Pattern: {pattern}")
                    print(f"Guessed: {guessed}")
                    print(f"True:    {true_word}")
                    print(f"Predicted letter: {best_letter}")
                    print(f"Blank Acc: {percent_recovered:.2%} ({recovered}/{blanks})")
                    print(f"Prediction based on transformer_prediction(): {self.transformer_prediction(pattern, guessed, use_blank_mask=True)}")

        print(f"\n[Transformer Evaluation]")
        print(f"Avg. Blank Accuracy: {total_recovered / max(total_words, 1):.2%}")
        print(f"Avg. Blanks Recovered: {total_recovered / max(total_words, 1):.2f}")
        print(f"High-Recovery Samples (≥80%): {samples_high_recovery}/{total_words}")
        print(f"Detailed log saved to {log_path}")





