from datasets import load_dataset
from datasets import load_dataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tiktoken
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer


# --- Configuration ---
seed: int = 1337
num_titles: int = 100_000
val_frac: float = 0.10

# Plotting Config
sns.set_theme(style="whitegrid", context="talk")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.family'] = 'sans-serif'

def get_titles(num_titles: int, seed: int, val_frac: float) -> tuple[list[str], list[str]]:
    """
    Fetches data from Hugging Face, shuffles, and splits into train/val.
    """
    print(f"Downloading/Loading dataset (taking {num_titles} samples)...")
    try:
        # Load streaming or full. Here we load 'train' split.
        ds = load_dataset("julien040/hacker-news-posts", split="train", cache_dir="./data").shuffle(seed=seed)
        
        # Extract titles
        titles = [str(row["title"]).strip() for row in ds.take(num_titles)]
        
        # Split logic
        n = int(num_titles * (1 - val_frac))
        return titles[:n], titles[n:]
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return [], []

def analyze_lengths(df):
    """
    Analyzes character, word, and BPE token lengths.
    Crucial for determining 'block_size' (context window).
    """
    if df.empty:
        print("Skipping length analysis (DataFrame is empty).")
        return df

    # 1. Word Count (Space split)
    df['word_count'] = df['text'].apply(lambda x: len(str(x).split()))
    
    # 2. GPT-2 Token Count (Actual LLM view)
    try:
        enc = tiktoken.get_encoding("gpt2")
        df['token_count'] = df['text'].apply(lambda x: len(enc.encode(str(x))))
    except Exception as e:
        print(f"Tiktoken error: {e}")
        return df
    
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    
    # Histogram of Token Counts
    sns.histplot(df['token_count'], bins=30, kde=True, ax=ax[0], color='#2c3e50')
    ax[0].set_title('Distribution of Token Lengths (GPT-2 Tokenizer)')
    ax[0].set_xlabel('Tokens per Headline')
    ax[0].axvline(df['token_count'].mean(), color='r', linestyle='--', label=f'Mean: {df["token_count"].mean():.1f}')
    ax[0].axvline(df['token_count'].quantile(0.99), color='g', linestyle='--', label=f'99th %: {df["token_count"].quantile(0.99):.0f}')
    ax[0].legend()

    # Boxplot for Outliers
    sns.boxplot(x=df['token_count'], ax=ax[1], color='#e74c3c')
    ax[1].set_title('Token Length Outliers')
    ax[1].set_xlabel('Tokens')
    
    plt.tight_layout()
    plt.savefig('1_length_distribution.png')
    print("-> Generated 1_length_distribution.png")
    
    return df

def analyze_vocabulary(df):
    """
    Analyzes top words and n-grams.
    Helps identify domain-specific jargon and stopword impact.
    """
    if df.empty: return

    # Helper to get top ngrams
    def get_top_ngrams(corpus, n=None, k=20):
        vec = CountVectorizer(ngram_range=(n, n), stop_words='english').fit(corpus)
        bag_of_words = vec.transform(corpus)
        sum_words = bag_of_words.sum(axis=0) 
        words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
        words_freq = sorted(words_freq, key = lambda x: x[1], reverse=True)
        return words_freq[:k]

    try:
        # Get Unigrams (Words) and Bigrams
        top_words = get_top_ngrams(df['text'], n=1, k=20)
        top_bigrams = get_top_ngrams(df['text'], n=2, k=20)

        fig, ax = plt.subplots(1, 2, figsize=(18, 8))

        # Plot Unigrams
        x, y = zip(*top_words)
        sns.barplot(x=list(y), y=list(x), ax=ax[0], palette="viridis")
        ax[0].set_title('Top 20 Unigrams (No Stopwords)')
        
        # Plot Bigrams
        x, y = zip(*top_bigrams)
        sns.barplot(x=list(y), y=list(x), ax=ax[1], palette="magma")
        ax[1].set_title('Top 20 Bigrams')

        plt.tight_layout()
        plt.savefig('2_vocabulary_frequency.png')
        print("-> Generated 2_vocabulary_frequency.png")
    except ValueError:
        print("Not enough data to generate vocabulary plots.")

def analyze_zipf_law(df):
    """
    Checks if the dataset follows Zipf's Law (Power Law).
    """
    if df.empty: return

    all_words = " ".join(df['text'].astype(str)).split()
    counts = Counter(all_words)
    
    # Sort by frequency
    frequencies = sorted(counts.values(), reverse=True)
    ranks = range(1, len(frequencies) + 1)

    plt.figure(figsize=(10, 6))
    plt.loglog(ranks, frequencies, marker=".", linestyle="none", alpha=0.5, color='#2980b9')
    plt.title("Zipf's Law Plot (Log-Log Scale)")
    plt.xlabel("Rank (Log)")
    plt.ylabel("Frequency (Log)")
    plt.grid(True, which="both", ls="--")
    
    plt.savefig('3_zipfs_law.png')
    print("-> Generated 3_zipfs_law.png")

# --- Main Execution ---
if __name__ == "__main__":
    print("Starting EDA...")
    
    # 1. Load Data using the new HF function
    train_titles, val_titles = get_titles(num_titles, seed, val_frac)
    
    print(f"Retrieved {len(train_titles)} training samples and {len(val_titles)} validation samples.")
    
    if not train_titles:
        print("No data retrieved. Exiting.")
        exit()

    # Convert to DataFrame for analysis tools
    # We primarily analyze the Training set to understand what the model will see
    df = pd.DataFrame(train_titles, columns=['text'])
    
    # 2. Lengths (Context Window decision)
    df = analyze_lengths(df)
    
    # 3. N-Grams (Content checks)
    analyze_vocabulary(df)
    
    # 4. Distribution check
    analyze_zipf_law(df)

    print("EDA Complete. Check the generated PNG files.")