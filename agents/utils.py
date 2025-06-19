import string
import pickle
import os
import random
from collections import Counter

import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn as nn
from sklearn.cluster import KMeans
from collections import defaultdict, Counter
VOCAB = ['<PAD>', '<SEP>', '<EOS>'] + list(string.ascii_lowercase) + ['_']
VOCAB_DICT = {ch: idx for idx, ch in enumerate(VOCAB)}
PAD_IDX = VOCAB_DICT['<PAD>']
SEP_IDX = VOCAB_DICT['<SEP>']
EOS_IDX = VOCAB_DICT['<EOS>']

def split_dictionary(dictionary, seed=42) -> None:
    random.seed(seed)
    random.shuffle(dictionary)
    train_dict = dictionary[:200_000]
    test_dict = dictionary[200_000:]
    with open("train_dictionary.txt", "w") as f:
        for word in train_dict:
            f.write(word + "\n")
    with open("test_dictionary.txt", "w") as f:
        for word in test_dict:
            f.write(word + "\n")

################################################
################################################
## Data loading and saving methods
################################################
################################################

def load_dictionary(path="words_250000_train.txt"):
    with open(path, "r") as f:
        return [line.strip() for line in f.readlines()]

def save_dictionary(obj, filename):
    with open(filename, 'w') as f:
        for item in obj:
            f.write("%s\n" % item)

def load_cache(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)

def save_cache(obj, filename):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)

def save_model(model, filename):
    with open(filename, 'wb') as f:
        pickle.dump(model, f)

def load_model(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# Common directories
DATA_DIR = os.path.join(ROOT, "data")
MODELS_DIR = os.path.join(ROOT, "models")
CHECKPOINTS_DIR = os.path.join(ROOT, "checkpoints")
CONFIG_DIR = os.path.join(ROOT, "misc")
RESULTS_DIR = os.path.join(ROOT, "results")
# Resolvers
def data_path(filename: str) -> str:
    return os.path.join(DATA_DIR, filename)
def model_path(filename: str) -> str:
    return os.path.join(MODELS_DIR, filename)
def checkpoint_path(filename: str) -> str:
    return os.path.join(CHECKPOINTS_DIR, filename)
def config_path(filename: str = "config.yaml") -> str:
    return os.path.join(CONFIG_DIR, filename)
def results_path(filename: str) -> str:
    return os.path.join(RESULTS_DIR, filename)


################################################
################################################
## Dictionary processing methods
################################################
################################################

def order_dictionary(dictionary):
    wordlen = max(len(s) for s in dictionary)
    x = [[] for _ in range(wordlen + 1)]
    for i in range(wordlen + 1):
        x[i] = [s for s in dictionary if len(s) == i]
    return x

def vectorize_word(word: str) -> np.ndarray:
    alphabet = list(string.ascii_lowercase)
    vec = np.zeros(len(alphabet))
    for letter in word:
        i = ord(letter) - ord('a')
        vec[i] += 1
    return vec

def vectorize_dictionary( dictionary: list[str]) -> np.ndarray:
    dictionary_vectorized = np.zeros((len(dictionary), 26))
    for i, word in enumerate(dictionary):
        dictionary_vectorized[i] = vectorize_word(word)
    return dictionary_vectorized

def cluster_dictionary(dictionary: list[str], n_clusters: int = 30):
    dictionary_vectorized = vectorize_dictionary(dictionary)
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    kmeans.fit(dictionary_vectorized)

    labels = kmeans.labels_
    label_map = {word: labels[i] for i, word in enumerate(dictionary)}

    # Group indices by cluster
    cluster_summary_by_label = {}
    for label in range(n_clusters):
        indices = np.where(labels == label)[0]
        cluster_words = [dictionary[i] for i in indices]
        cluster_vectors = dictionary_vectorized[indices]

        summary = {
            "avg_word_length": np.mean([len(w) for w in cluster_words]),
            "letter_frequency": np.mean(cluster_vectors, axis=0).tolist(),
            "letter_variance": np.var(cluster_vectors, axis=0).tolist(),
        }
        cluster_summary_by_label[label] = summary

    return label_map, cluster_summary_by_label

def letter_frequency_order(dictionary: list[str]) -> np.ndarray:
    total_vector = np.zeros(26)
    for word in dictionary:
        total_vector += vectorize_word(word)
    letter_freq = {char: total_vector[i] / len(dictionary)
                    for i, char in enumerate(string.ascii_lowercase)}
    sorted_letters = sorted(letter_freq.items(),
                             key=lambda x: x[1], reverse=True)
    order = [char for char, _ in sorted_letters]
    return order

def extract_affixes(dictionary: list[str], min_count=5, min_len=2, max_len=4):
    """
    Extract common prefixes and suffixes from a list of words.

    Args:
        dictionary (List[str]): List of words (training dictionary).
        min_count (int): Minimum frequency to keep an affix.
        min_len (int): Minimum affix length.
        max_len (int): Maximum affix length.

    Returns:
        Tuple[Dict[str, int], Dict[str, int]]: prefix_dict, suffix_dict
    """
    prefix_counter = Counter()
    suffix_counter = Counter()

    for word in dictionary:
        for k in range(min_len, max_len + 1):
            if len(word) >= k:
                prefix = word[:k]
                suffix = word[-k:]
                prefix_counter[prefix] += 1
                suffix_counter[suffix] += 1

    # Filter out infrequent affixes
    common_prefixes = {p: c for p, c in prefix_counter.items() if c >= min_count}
    common_suffixes = {s: c for s, c in suffix_counter.items() if c >= min_count}

    return common_prefixes, common_suffixes

def extract_midfixes(dictionary, min_len=3, max_len=5, min_count=5):
    """
    Extract mid-word substrings that appear frequently.
    """
    midfix_counter = Counter()

    for word in dictionary:
        word = word.lower()
        for k in range(min_len, max_len + 1):
            for i in range(1, len(word) - k):  # exclude prefix/suffix
                midfix = word[i:i+k]
                midfix_counter[midfix] += 1

    common_midfixes = {m: c for m, c in midfix_counter.items() if c >= min_count}
    return common_midfixes

def find_matching_affixes(pattern: str, prefixes: list[str], suffixes: list[str], midfixes: list[str], guessed_letters: set[str]) -> dict:
    """
    Returns matching affixes (prefixes, suffixes, midfixes) that align with the pattern.
    '_' in the pattern is treated as a wildcard ('.').
    """
    pattern_regex = pattern.replace('_', '.')
    matches = {
        "prefixes": [],
        "suffixes": [],
        "midfixes": []
    }

    # Match prefixes
    for prefix in prefixes:
        if len(prefix) > len(pattern):
            continue
        segment = pattern[:len(prefix)]
        if all(p == '.' or p == a for p, a in zip(segment, prefix)):
            matches["prefixes"].append(prefix)

    # Match suffixes
    for suffix in suffixes:
        if len(suffix) > len(pattern):
            continue
        segment = pattern[-len(suffix):]
        if all(p == '.' or p == a for p, a in zip(segment, suffix)):
            matches["suffixes"].append(suffix)

    # Match midfixes
    for mid in midfixes:
        L = len(mid)
        for i in range(len(pattern) - L + 1):
            segment = pattern[i:i+L]
            if all(p == '.' or p == m for p, m in zip(segment, mid)):
                matches["midfixes"].append(mid)
                break  # No need to match this mid again

    return matches

def fuzzy_affix_match(pattern,
                affix_dicts,
                guessed_letters,
                min_shared=3,
                verbose=True,
                return_all_matches=False):
    """
    Tries to find affixes that match parts of the pattern via sliding-window.
    Scores unguessed letters in matching affixes by their frequency.

    If return_all_matches is True, returns the list of affix strings instead.
    """
    pattern_str = ''.join(pattern)
    letter_scores = Counter()
    matched_affixes = []

    if not isinstance(affix_dicts, dict):
        raise TypeError("affix_dicts must be a flat dictionary of affix → frequency")

    for affix, freq in affix_dicts.items():
        affix_len = len(affix)
        if len(pattern_str) < affix_len:
            continue

        for i in range(len(pattern_str) - affix_len + 1):
            window = pattern_str[i : i + affix_len]
            if '_' not in window:
                continue
            match_count = sum(
                1 for j in range(affix_len)
                if pattern_str[i + j] != '_' and pattern_str[i + j] == affix[j]
            )
            if all([match_count >= min(min_shared, affix_len-1),
                    affix_len>=3,
                    not all(c in guessed_letters for c in affix)]):
                matched_affixes.append((affix, freq))
                if not return_all_matches:
                    for letter in set(affix) - guessed_letters:
                        letter_scores[letter] += freq
                break  # prevent double-counting same affix

    if return_all_matches:
        return [affix for affix, _ in matched_affixes]

    if letter_scores:
        if verbose:
            print("[DEBUG] Fuzzy match letter scores:", dict(letter_scores))
        return letter_scores.most_common(1)[0][0]

    return None

def filter_redundant_affixes(affixes, affix_freqs, verbose=False):
    """
    Filters out affixes that are nested inside longer affixes with higher or comparable frequency.
    
    Args:
        affixes: list of affix strings (matched candidates)
        affix_freqs: dict mapping affix → frequency (flat dict)
        verbose: whether to print debug info

    Returns:
        List of non-redundant affixes
    """
    affixes = sorted(affixes, key=len)
    to_remove = set()

    for i, a in enumerate(affixes):
        for j in range(i + 1, len(affixes)):
            a_prime = affixes[j]
            if a in a_prime:
                if affix_freqs.get(a_prime, 0) >= 0.8*affix_freqs.get(a, 0):
                    to_remove.add(a)
                    if verbose:
                        print(f"[PRUNE] Discarding '{a}' as redundant inside '{a_prime}'")
                    break  # stop once a dominating container is found

    return [a for a in affixes if a not in to_remove]

def get_letters_from_affixes(matched_affixes, guessed_letters):
    """
    From a dict of matched affixes, return a frequency counter of unguessed letters.
    """
    letter_freq = defaultdict(int)
    for affix_list in matched_affixes.values():
        for affix in affix_list:
            for char in affix:
                if char not in guessed_letters:
                    letter_freq[char] += 1
    return dict(letter_freq)

def get_or_create_train_test_dictionary(dictionary, train_path, test_path):
    if os.path.exists(train_path) and os.path.exists(test_path):
        return load_dictionary(train_path), load_dictionary(test_path)
    split_dictionary(dictionary)
    train_dictionary = load_dictionary("train_dictionary.txt")
    test_dictionary = load_dictionary("test_dictionary.txt")
    save_dictionary(train_dictionary, train_path)
    save_dictionary(test_dictionary, test_path)
    return train_dictionary, test_dictionary

def get_or_create_ordered_dictionary(dictionary, cache_path):
    if os.path.exists(cache_path):
        return load_cache(cache_path)
    ordered_dictionary = order_dictionary(dictionary)
    save_cache(ordered_dictionary, cache_path)
    return ordered_dictionary

def get_or_create_cluster_data(dictionary, n_clusters = 50):
    label_map_path = data_path(f"label_map_{n_clusters}.pkl")
    summary_path = data_path(f"cluster_summary_by_label_{n_clusters}.pkl")
    if os.path.exists(label_map_path) and os.path.exists(summary_path):
        label_map = load_cache(label_map_path)
        cluster_summary_by_label = load_cache(summary_path)
        return label_map, cluster_summary_by_label
    label_map, cluster_summary_by_label = cluster_dictionary(dictionary, n_clusters=n_clusters)
    save_cache(label_map, label_map_path)
    save_cache(cluster_summary_by_label, summary_path)
    return label_map, cluster_summary_by_label

def get_or_create_affixes(dictionary, affix_cache_path="affixes_train.pkl", min_count=5):
    """
    Loads affix data from a single pickle file or creates it from dictionary.
    """
    if os.path.exists(affix_cache_path):
        with open(affix_cache_path, 'rb') as f:
            return pickle.load(f)
    else:
        prefixes, suffixes = extract_affixes(dictionary, min_count=min_count)
        midfixes = extract_midfixes(dictionary, min_count=min_count)

        affix_data = {
            "prefixes": prefixes,
            "suffixes": suffixes,
            "midfixes": midfixes
        }

        with open(affix_cache_path, 'wb') as f:
            pickle.dump(affix_data, f)

        return affix_data

def flatten_affix_dict(affix_dict):
    """
    Combine prefix, suffix, and midfix dictionaries into one Counter.
    """
    flat = Counter()
    for affix_type in ['prefixes', 'suffixes', 'midfixes']:
        flat.update(affix_dict.get(affix_type, {}))
    return flat


def prune_nested_affixes(affix_counts, relative_freq_thresh=0.2):
    """
    For each affix a, removes any longer affix a' where:
    - a is a substring of a'
    - and frequency of a' < relative_freq_thresh × frequency of a

    Keeps a if a' doesn’t add much additional information.
    """
    affixes = sorted(affix_counts.keys(), key=len)
    to_prune = set()

    for i, a in enumerate(affixes):
        f_a = affix_counts[a]
        for j in range(i + 1, len(affixes)):
            a_prime = affixes[j]
            f_ap = affix_counts[a_prime]

            if a in a_prime and len(a_prime) > len(a):
                if f_ap < relative_freq_thresh * f_a:
                    to_prune.add(a_prime)

    return {a: f for a, f in affix_counts.items() if a not in to_prune}

def regroup_by_affix_type(pruned_flat_affixes, original_affix_dict):
    """
    Given a pruned flat affix dict, split it back into types based on original dict.
    """
    regrouped = {'prefixes': {}, 'suffixes': {}, 'midfixes': {}}

    for affix_type in ['prefixes', 'suffixes', 'midfixes']:
        for affix in original_affix_dict[affix_type]:
            if affix in pruned_flat_affixes:
                regrouped[affix_type][affix] = pruned_flat_affixes[affix]

    return regrouped

def consolidate_affixes(affix_data: dict) -> dict:
    """
    Merges prefix, suffix, and midfix dictionaries into a single affix frequency dictionary.
    """
    from collections import Counter
    all_affixes = Counter()

    for category in ['prefixes', 'suffixes', 'midfixes']:
        if category in affix_data:
            all_affixes.update(affix_data[category])

    return dict(all_affixes)

################################################
################################################
## Encoding methods
################################################
################################################

def encode_pattern(pattern: str, max_len: int = 15) -> np.ndarray:
    '''
    Encode a pattern string into a vector of fixed length.
    Each character in the pattern is represented as follows:
    - '_' (underscore) is encoded as 0
    - any other character is encoded as its position in the alphabet (1-26)
    The vector is padded with -1s to ensure it has a fixed length.
    '''
    vec = [ord(c) - ord('a') + 1 if c != '_' else 0 for c in pattern]
    vec += [-1] * (max_len - len(vec))  # pad with -1s
    return np.array(vec[:max_len])

def encode_guessed_letters(guessed_letters: str) -> np.ndarray:
    """
    Encode guessed letters into a vector.
    Each letter is represented as its position in the alphabet (1-26).
    """
    mask = np.zeros(26, dtype=int)
    for c in guessed_letters:
        mask[ord(c) - ord('a')] = 1
    return np.array(mask)

def encode_input(pattern: str, guessed: str, max_len: int) -> torch.Tensor:
    seq = list(pattern) + ['<SEP>'] + list(guessed)
    token_ids = [VOCAB_DICT[c] for c in seq]
    if len(token_ids) < max_len:
        token_ids += [PAD_IDX] * (max_len - len(token_ids))
    else:
        token_ids = token_ids[:max_len]
    return torch.tensor(token_ids, dtype=torch.long)

def encode_output(word, max_len):
    token_ids = [VOCAB_DICT[c] for c in word]
    if len(token_ids) < max_len:
        token_ids.append(VOCAB_DICT['<EOS>'])  # insert EOS as explicit stop signal
        while len(token_ids) < max_len:
            token_ids.append(-100)  # mask unused positions
    else:
        token_ids = token_ids[:max_len]

    return torch.tensor(token_ids, dtype=torch.long)

def encode_sequence(seq, max_len):
        """
        Converts a string sequence into a list of vocab indices.
        Pads the result to max_len.
        """
        # seq_idx = [VOCAB_DICT[c] for c in seq]
        # filter out unknown characters: need to check why this happens
        seq_idx = [VOCAB_DICT[c] for c in seq if c in VOCAB_DICT] 

        if len(seq_idx) > max_len:
            return seq_idx[:max_len]
        return seq_idx + [PAD_IDX] * (max_len - len(seq_idx))

################################################
################################################
## Filtering candidate words and feature extraction
################################################
################################################

def filter_candidates(curr_pattern, guessed_letters, ordered_dictionary):
    incorrect_letters = set(guessed_letters)
    for c in curr_pattern:
        if c != '_':
            incorrect_letters.discard(c)
    n = len(curr_pattern)
    pos = []
    for i in range(n):
        if curr_pattern[i] != '_':
            pos.append(i)
    filtered_dictionary = []
    for s in ordered_dictionary[n]:
        if all(s[i] == curr_pattern[i] for i in pos) and all(c not in s for c in incorrect_letters):
            filtered_dictionary.append(s)
    return filtered_dictionary 

def filtered_candidates_features(curr_pattern: str,
                                guessed_letters: str,
                                label_map: dict,
                                cluster_summary_by_label: dict,
                                ordered_dictionary: list[list[str]],
                                max_word_len=20):
    """
    Generate a comprehensive feature vector for the current Hangman game state.

    Features:
    1. [1] Log(1 + number of remaining candidates)
    2. [26] Letter frequency distribution over remaining candidates
    3. [max_word_len * 26] Positional letter frequency matrix (flattened)
    4. [1+26+26] Weighted cluster summary stats
    5. [max_word_len] Encoded current pattern (a=0, b=1, ..., z=25, _=-1)
    6. [26] Guessed letter status (-1 = not guessed, 0 = guessed & not present, 1 = guessed & present)
    """
    guessed_letters = set(guessed_letters)
    candidates = filter_candidates(curr_pattern,
                                   guessed_letters,
                                   ordered_dictionary)
    total_candidates = len(candidates)

    if not candidates:
        return [0.0] * (1 + 26 + max_word_len * 26 + 53 + max_word_len + 26)

    # 1. Log-count of candidates
    feature_1 = [np.log1p(total_candidates)]

    # 2. Letter frequency (a–z)
    letter_freq = [0] * 26
    for word in candidates:
        for c in set(word):
            if c.isalpha():
                idx = ord(c) - ord('a')
                letter_freq[idx] += 1
    feature_2 = [f / total_candidates for f in letter_freq]

    # 3. Positional letter frequency matrix (flattened)
    pos_matrix = np.zeros((max_word_len, 26))
    for word in candidates:
        for i, c in enumerate(word[:max_word_len]):
            if c.isalpha():
                pos_matrix[i, ord(c) - ord('a')] += 1
    pos_matrix /= total_candidates
    feature_3 = pos_matrix.flatten().tolist()

    # 4. Cluster summary stats
    cluster_features = encode_clusters(candidates, label_map, cluster_summary_by_label)

    # 5. Encoded pattern
    pattern_encoded = [-1] * max_word_len
    for i, c in enumerate(curr_pattern[:max_word_len]):
        if c.isalpha():
            pattern_encoded[i] = ord(c) - ord('a')

    # 6. Guessed letter status
    guessed_status = [-1] * 26
    for c in guessed_letters:
        idx = ord(c) - ord('a')
        guessed_status[idx] = 1 if c in curr_pattern else 0

    return feature_1 + feature_2 + feature_3 + cluster_features + pattern_encoded + guessed_status

def get_feature_names(max_word_len=20):
    names = ["log_num_candidates"]
    names += [f"freq_{chr(ord('a') + i)}" for i in range(26)]
    names += [f"pos_{p}_{chr(ord('a') + i)}" for p in range(max_word_len) for i in range(26)]
    names += [f"pattern_{i}" for i in range(max_word_len)]
    names += [f"guessed_{chr(ord('a') + i)}" for i in range(26)]
    return names

def encode_clusters(
    candidates: list[str],
    label_map: dict,
    cluster_summary_by_label: dict,
) -> list[float]:
    """
    Given current pattern and guessed letters,
    computes:
    - Weighted average of cluster summary stats across candidate words
    """

    # Step 1: Map candidates to cluster labels and count frequencies
    candidate_labels = [label_map[word] for word in candidates]
    label_counts = Counter(candidate_labels)
    total = sum(label_counts.values())

    # Step 2: Compute weighted cluster-level features
    weighted_avg_length = 0
    weighted_freq = np.zeros(26)
    weighted_var = np.zeros(26)

    for label, count in label_counts.items():
        w = count / total
        summary = cluster_summary_by_label[label]
        weighted_avg_length += summary["avg_word_length"] * w
        weighted_freq += np.array(summary["letter_frequency"]) * w
        weighted_var += np.array(summary["letter_variance"]) * w

    features = [weighted_avg_length]
    features.extend(weighted_freq.tolist())
    features.extend(weighted_var.tolist())

    return features

################################################
################################################
## Word stratification and sampling methods
## for late stage attention model training
################################################
################################################

def stratify_words_by_features(word_list, affix_set):
    """
    Stratifies words into buckets based on:
    - Word length: short (5-7), medium (8-10), long (11+)
    - Presence of affix (any matching prefix/suffix from affix set)
    - Character diversity (low if <5 unique characters, else high)

    Skips words with fewer than 5 characters (not useful for late-stage game modeling).
    """
    buckets = defaultdict(list)

    for word in word_list:
        length = len(word)
        if length < 5:
            continue  # Skip too-short words

        # Length bucket
        if length <= 7:
            size_bucket = "short"
        elif length <= 10:
            size_bucket = "medium"
        else:
            size_bucket = "long"

        # Affix match bucket
        has_affix = any(word.startswith(a) or word.endswith(a) for a in affix_set)
        affix_bucket = "affix" if has_affix else "no_affix"

        # Diversity bucket
        unique_chars = len(set(word))
        diversity_bucket = "low" if unique_chars < 5 else "high"

        # Final stratification key
        key = (size_bucket, affix_bucket, diversity_bucket)
        buckets[key].append(word)

    return buckets

def sample_stratified_words(buckets, total_samples=50000):
    sampled_words = []
    per_bucket = total_samples // len(buckets)
    for key, words in buckets.items():
        if len(words) <= per_bucket:
            sampled_words.extend(words)
        else:
            sampled_words.extend(random.sample(words, per_bucket))
    return sampled_words

# --- Dataset for Transformer Model ---

class HangmanTransformerDataset(Dataset):
    """
    PyTorch Dataset for training a transformer to guess the full word
    given the current pattern and guessed letters.
    """
    def __init__(self, data, max_len=20, use_blank_mask=False):
        """
        Args:
            data: list of tuples (pattern, guessed_letters, full_word)
            max_len: maximum word length for padding
            use_blank_mask: whether to use a mask for blank positions
        """
        self.data = data
        self.max_len = max_len
        self.use_blank_mask = use_blank_mask

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        pattern, guessed_letters, word = self.data[idx]

        pattern_encoded = encode_sequence(pattern, self.max_len)
        guessed_encoded = encode_sequence(guessed_letters, self.max_len)
        word_encoded = encode_sequence(word, self.max_len)

        pattern_tensor = torch.tensor(pattern_encoded, dtype=torch.long)
        guess_tensor = torch.tensor(guessed_encoded, dtype=torch.long)
        target_tensor = encode_output(word, self.max_len)

        if self.use_blank_mask:
            blank_mask = [1 if idx == VOCAB_DICT['_'] else 0 for idx in pattern_encoded]
            blank_mask_tensor = torch.tensor(blank_mask 
                                             + [0] * (self.max_len - len(blank_mask)),
                                             dtype=torch.long)
            return (pattern_tensor, guess_tensor, blank_mask_tensor), target_tensor
        else:
            return (pattern_tensor, guess_tensor), target_tensor

##################################################
##################################################
## Transformer model for Hangman word prediction
##################################################
##################################################

class HangmanTransformerModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        nhead: int = 4,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.2,
        max_seq_len: int = 20,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = nn.Parameter(torch.randn(max_seq_len * 2 + 1, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        self.decoder = nn.Linear(d_model, vocab_size)

    def forward(self, pattern_input, guess_input):
        # pattern_input and guess_input: (batch_size, seq_len)
        batch_size = pattern_input.size(0)

        # Concatenate pattern and guessed letters, separated by <SEP>
        sep = torch.full((batch_size, 1), fill_value=SEP_IDX, dtype=torch.long, device=pattern_input.device)
        x = torch.cat([pattern_input, sep, guess_input], dim=1)  # (batch_size, total_seq_len)

        # Apply embedding + positional encoding
        x = self.embedding(x) + self.pos_encoder[:x.size(1)]

        encoded = self.transformer_encoder(x)  # (batch_size, seq_len, d_model)
        output = self.decoder(encoded[:, :self.max_seq_len])  # Predict only the pattern positions

        return output  # shape: (batch_size, max_seq_len, vocab_size)

class HangmanTransformerModelV2(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 384,
        nhead: int = 6,
        num_layers: int = 4,
        dim_feedforward: int = 768,
        dropout: float = 0.2,
        max_seq_len: int = 20,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = nn.Parameter(torch.randn(max_seq_len * 2 + 1, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.decoder = nn.Linear(d_model, vocab_size)

    def forward(self, pattern_input, guess_input):
        batch_size = pattern_input.size(0)
        sep_idx = 2  # Assuming VOCAB_DICT['<SEP>'] = 2
        sep = torch.full((batch_size, 1), fill_value=sep_idx, dtype=torch.long, device=pattern_input.device)

        x = torch.cat([pattern_input, sep, guess_input], dim=1)
        x = self.embedding(x) + self.pos_encoder[:x.size(1)]

        encoded = self.transformer_encoder(x)
        output = self.decoder(encoded[:, :self.max_seq_len])
        return output

class HangmanTransformerModelV3(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 384,
        nhead: int = 6,
        num_layers: int = 4,
        dim_feedforward: int = 768,
        dropout: float = 0.2,
        max_seq_len: int = 11,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.blank_embedding = nn.Embedding(2, d_model)  # 0 = known letter, 1 = blank
        self.pos_encoder = nn.Parameter(torch.randn(max_seq_len * 2 + 1, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.decoder = nn.Linear(d_model, vocab_size)

    def forward(self, pattern_input, guess_input, blank_mask):
        batch_size = pattern_input.size(0)
        sep_idx = 2  # Assuming VOCAB_DICT['<SEP>'] = 2
        sep = torch.full((batch_size, 1), fill_value=sep_idx, dtype=torch.long, device=pattern_input.device)

        x_token = torch.cat([pattern_input, sep, guess_input], dim=1)

        # Construct padded blank mask (pattern part gets embedding, rest is 0)
        blank_mask_padded = torch.cat([
            blank_mask,                             # [batch, max_seq_len]
            torch.zeros((batch_size,
                        1 + guess_input.size(1)),
                        dtype=torch.long,
                        device=blank_mask.device)
        ], dim=1)                                   # total shape: [batch, total_seq_len]

        x = self.embedding(x_token) + \
            self.blank_embedding(blank_mask_padded) + \
            self.pos_encoder[:x_token.size(1)]


        encoded = self.transformer_encoder(x)
        output = self.decoder(encoded[:, :self.max_seq_len])
        return output

def mask_logits(logits: torch.Tensor, guessed_letters: str, alphabet: str = "abcdefghijklmnopqrstuvwxyz") -> torch.Tensor:
    guessed_idxs = [alphabet.index(c) for c in guessed_letters if c in alphabet]
    if not guessed_idxs:
        return logits

    mask = torch.zeros_like(logits)
    for idx in guessed_idxs:
        mask[:, :, idx] = float('-inf')

    return logits + mask

