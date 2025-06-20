# Hangman

This repository contains multiple agents, training and evaluation pipelines, and a hybrid architecture combining affix-based heuristics with machine learning to play the hangman gain on a dictionary of 50000 words (test set). The agents are trained on a similar by disjoint dictionary of 200000 words (training set).

Note: this challenge was developed in one week as part of a recruiting process and has been anonymized for general sharing.

## Final Result

The agent "Subpattern_Greedy" achieved a 27.5% win rate on the test set. The final strategy integrated affix-aware subpattern matching, fallback logical reasoning, with affixes estimated on the training dictionary. Despite operating under constrained black-box conditions, the approach demonstrated significant gains over baseline frequency methods and exploratory agents.

## Setup

This project uses Python 3.8+ and requires a few packages such as `torch`, `scikit-learn`, `seaborn`, etc.

To install dependencies (note: some dependencies like torch may need to be installed separately):

```bash
pip install -r requirements.txt
```

## Folder structure

Below is the general structure of this folder, showing the most relevant files contained in it for my submission.

```bash
/Hangman
│
├── README.md
├── requirements.txt                 # Optional
├── .gitignore
│
├── agents/                          # Final agent implementations
│   ├── submission_agent.py
│   ├── agent.py                     # Core strategy controller
│   ├── utils.py                     # Shared logic
│
├── scripts/                         # Run and evaluation scripts
│   ├── generate_data.py
│   ├── generate_transformer_data_subpattern.py
│   ├── train_agent.py
│   ├── evaluate_bot.py
│   ├── evaluate_transformer.py
│
├── models/                          # Final trained models
│   ├── transformer_model_subpattern.pt
│   ├── transformer_model_subpattern_v2.pt
│   ├── transformer_model_subpattern_v3_short.pt
│   └── checkpoints/
│       ├── v2_run/
│       └── v3_run/
│
├── data/                            # Final datasets + wordlists
│   ├── train_dictionary.txt
│   ├── test_dictionary.txt
│   ├── stratified_word_sample50000.pkl
│   ├── transformer_data200000.pkl
│   ├── transformer_eval_data.pkl
│   └── cluster_summary_by_label_50.pkl
│
├── results/                         # Logs + winrate summaries
│   ├── trexquant_log.csv
│   ├── trexquant_log.jsonl
│   ├── heatmaps/
│   │   ├── affix_heatmap.png
│   │   ├── transformer_prediction_heatmap.png
│   │   └── transformer_softmax_score_heatmap.png
│   ├── logs/
│   │   ├── affix_agent_failures.jsonl
│   │   ├── transformer_eval.jsonl
│   │   ├── val_failures.jsonl
│   │   └── evaluate_bot/
│   │       └── (grouped by agent names)
│
├── notebooks/                       # Optional, for EDA and visualization
│   ├── experiments.ipynb
│   ├── hangman_api_user.ipynb
│   ├── heatmaps.ipynb
│
├── Deprecated/                      # Archive for legacy files
│   └── Old_but_Gold/                # Your nostalgic gold
│
└── misc/
    ├── environment.yml
    ├── config.yaml
    ├── all_other_pickle_dumps.pkl
```

## Dictionary Split and Sample Generation

We randomly split the full 250,000-word dictionary into:

train_dictionary.pkl (80%): used for affix extraction, ML training

test_dictionary.pkl (20%): used for evaluation

## Agent Architectures

### Subpattern_Greedy

Our core non-ML agent using affix similarity, vowel heuristics, and logical fallback

Fuzzy match against known affixes (prefixes, suffixes, midfixes)

Votes across unguessed letters in matched affixes, pruned for redundancy

Win rate: ~27%

Able to recover at least 60% of letters on 90% of the words in testing dictionary

### Hybrid_SB_Transformer and Hybrid_SB_Transformer_V2

Uses subpattern agent until ≥60% of letters are known and word length ∈ [5, 11]

Switches to a Transformer trained to predict letters at blank positions

V2 instead determines the switch based on the confidence of each prediction, measured by probability

Transformer uses learned blank mask embeddings to focus on unrevealed letters

It assigns a vector of probabilities at each position for each unguessed  letter

Inputs are first encoded, then passed through a multiheaded attention layer, and finally probabilities are computed

Training aims at minimizing average cross entropy across all blank positions

Specifically, for each position with a blank (_), we compute the cross-entropy loss between the predicted softmax distribution and the true letter at that position, and average this across all blanks in the training batch

Intended to capture long-range dependencies that affix logic can’t

For instance, over_a__ing → likely overlapping, not overmanning, while over_a_ can be both overlap or overman

Also, it does not need an arbitrary definition of the next "best guess", which is sensitive to position

Training time for transformer: ~20 minutes on GPU with 50k samples

Win rate: ~22% (with training on 50k sample generated as described above)

### Hybrid_ML (Random Forest)

Chooses between RF predictions and logical heuristics

Random Forest trained on ~15k structured feature vectors

Offers interpretable backup for pattern-based guessing

Win rate: ~16%

### Frequency, Entropy, Logical

Simpler fallback strategies:

Frequency: most common unguessed letter in candidates

Entropy: most uncertain split across candidates

Logical: most common letter in unknown positions

## Training and Features Extraction

### Affix Extraction

The "Subpattern_Greedy" agent is based on the idea that the 250000 words in the dictionary are combination of common English words (~100-150k) and of prefixes, suffixes and roots. For example, "overageness" is a combination of the prefix "over", the word (or root) "age" and the suffix "ness"

We extract frequent prefixes, suffixes, and midfixes of lengths 2–5 from the training dictionary:

A sliding window is used to collect all substrings

We filter by minimum frequency (≥20) and prune redundant affixes

The idea is to capture frequently occurring patterns (e.g., pre, -ing, ous) that can be reused in unseen words

This forms the backbone of our Subpattern Greedy agent, which uses fuzzy matching to score unguessed letters based on affix frequency overlap

By this I mean that all affixes can be matched with any subpattern in the word, independently of each position

For instance, "over" will be matched with a current pattern such as "___o__e__r__", independently of the position of the subpattern that matches  it

This reflects my idea of how the Trexquant dictionary was formed in the first place

### Transformer Training Samples

Although the Subpattern agent is good with local dependencies across a word, it is possible that a word present long term dependencies. The goal of the transformer is to look for such long term dependencies

It's training dataset is generated from simulated games using the subpattern agent. Specifically:

Snapshots taken when 60–99% of letters are revealed

Each sample is a tuple: (current pattern, guessed letters, ground truth word)

This training data reflects mid-to-late game decision states

Time to generate sample: approximately 20 minutes to generate sample of size 50000 using 20 workers

### Random Forest Training Samples

We extract features from candidate word lists filtered by current pattern and guessed letters. Features include:

Log number of remaining candidates

Letter frequencies over candidates

Positional frequency matrix (letters by position)

Cluster summaries: Each candidate word is assigned a cluster (via KMeans on letter distributions), and we extract:

Average word length in cluster

Cluster-level letter frequency and variance vectors

The motivation is to give the RF model a context-aware summary of how the remaining words “look,” both structurally and statistically.

## Heatmaps

The following heatmaps helped us understand guessed letter distributions for Subpattern Greedy and Transformer:

heatmaps/affix_heatmap.png: Consonant frequency by position in affixes

heatmaps/transformer_prediction_heatmap.png: Position-by-letter frequency of top transformer predictions

heatmaps/transformer_softmax_score_heatmap.png: Softmax-weighted confidence across positions

![Affix Heatmap](heatmaps/affix_heatmap.png)
![Transformer Prediction Heatmap](heatmaps/transformer_prediction_heatmap.png)
![Transformer Confidence Heatmap](heatmaps/transformer_softmax_score_heatmap.png)

Specifically, the Subpattern Greedy agent often guesses letters such as t and r, because included in most common affixes

The Transformer instead does not guess this letters, because it was trained on data were they were already guessed by the Subpattern Greedy agent

## How to run a Game with an agent

On Windows Powershell, set the agent mode (e.g. Hybrid_SB_Transformer) in `config.yaml` by typing

```powershell
(Get-Content misc/config.yaml) -replace '^agent_mode: .*', 'agent_mode: Hybrid_SB_Transformer' | Set-Content misc/config.yaml
```

Then, execute:

```bash
python scripts/evaluate_bot.py --n_games 100
```

## Results

### Agent Performance Summary

| Agent Name                      | Win Rate   | Description                                                                 |
|---------------------------------|------------|-----------------------------------------------------------------------------|
| *Subpattern_Greedy*             | ~25.6%     | Uses affix matching (prefix/suffix/stem) to guess high-frequency letters.   |
| *Hybrid_SB_Transformer*         | ~21.8%     | Switches from Subpattern to Transformer when ≥60% letters are revealed.     |
| *Hybrid_SB_Transformer_V2*      | ~21.2%     | Switches from Subpattern to Transformer based on confidence                 |
| *Hybrid_nonML*                  | ~15.0%     | ML model using filtered candidate features; exploratory and modular.        |
| *Hybrid_ML*                     | ~13.0%     | ML model using filtered candidate features; exploratory and modular.        |
| *Random Forest*                 | ~11.2%     | ML model using filtered candidate features; exploratory and modular.        |
| *Frequency / Entropy / Logical* | Lower      | Simpler rule-based agents used as baselines or fallback logic.              |

## Next Steps

### Improving Transformers prediction

The Transformers predicts different letters than the Subpattern Greedy agent, on samples where the SG agent fails to reveal the word

However, the percentage of correct guesses is around 20-25%, which is still too low for the Hybrid agent to produce better results

This is likely because the hyperparameters chosen for deciding which agent to use have not been optimize

Of course, these hyperparameters can be optimized on the testing dataset, and that's one direction to look into

In addition, the transformer architecture, if provided with more data, may be flexible enough to capture the errors of the SG agent

The sample can be created through bootstrapping, while choosing different snapshot of current pattern and guesses for each sample

This could yield, from just the training dictionary, hundreds or even more than a million samples

This would require, however, a day or so of calculations, so I did not pursue this road for this challenge

### Meta agent

Instead of setting a decision rule for the hybrid agent, one could train a meta agent on the test dataset

In other words, one could learn from the data when to use the subpattern Greedy agent, and when the Transformer

### Generative Adversarial Networks

Another idea is to reverse engineer the way the dataset was constructed

To do so, for each set of affixes, we could identify all words that are composed by one or more of them

Then, we train a GAN to determine, for each such set, what are the possible words available

When a sufficiently large set of affixes is determined by the subpattern gradient agent, we then use the GAN to identify possible words

This should reduce the set of filtered candidates, because it is possible that a certain affix is most frequently used as, say, a prefix

A letter could then be selected based on frequency across these candidates

## Final Thoughts

This project was a great opportunity to combine linguistic intuition with machine learning and symbolic heuristics. It reflects both analytical reasoning and creative exploration under time constraints. I'm grateful to the Trexquant team for the challenge.
