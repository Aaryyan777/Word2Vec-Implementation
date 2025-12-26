import torch
from torch.utils.data import Dataset
import numpy as np
from collections import Counter
import random

class Text8Dataset(Dataset):
    def __init__(self, data_path, vocab_size=20000, window_size=5, subsample_t=1e-5):
        self.window_size = window_size
        
        # Read data
        print("Reading data...")
        with open(data_path, 'r') as f:
            text = f.read()
        self.tokens = text.split()
        
        # Build Vocab
        print("Building vocabulary...")
        word_counts = Counter(self.tokens)
        self.vocab = sorted(word_counts, key=word_counts.get, reverse=True)[:vocab_size]
        self.word2idx = {word: idx for idx, word in enumerate(self.vocab)}
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}
        
        # Convert tokens to indices and Subsample
        print("Subsampling and indexing...")
        total_count = len(self.tokens)
        self.data = []
        for word in self.tokens:
            if word in self.word2idx:
                idx = self.word2idx[word]
                # Subsampling formula from the paper (or common implementation)
                count = word_counts[word]
                freq = count / total_count
                p_keep = (np.sqrt(freq / subsample_t) + 1) * (subsample_t / freq)
                
                if random.random() < p_keep:
                    self.data.append(idx)
                    
        print(f"Original tokens: {len(self.tokens)}")
        print(f"After subsampling: {len(self.data)}")

        # Pre-calculate counts for negative sampling distribution
        self.vocab_counts = np.array([word_counts[self.idx2word[i]] for i in range(len(self.vocab))])
        self.freqs = self.vocab_counts / self.vocab_counts.sum()
        self.unigram_dist = self.freqs ** 0.75
        self.unigram_dist /= self.unigram_dist.sum()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # This approach generates one pair at a time per index, which is simple but
        # technically iterates over the dataset multiple times implicitly if we consider
        # the window. A more standard PyTorch way for huge datasets is usually an IterableDataset
        # or generating all pairs upfront (memory intensive).
        # For simplicity/demo with text8: we'll randomly sample a context word for the center word at 'idx'
        # each time this is called. This adds some randomness (stochastic) but works for training.
        
        center_word = self.data[idx]
        
        # Dynamic window size (as per paper: random R in [1, C])
        R = random.randint(1, self.window_size)
        
        # Get context window bounds
        start = max(0, idx - R)
        end = min(len(self.data), idx + R + 1)
        
        context_indices = [i for i in range(start, end) if i != idx]
        
        if not context_indices:
            # Fallback if isolated (rare)
            return self.data[idx], self.data[idx]
            
        context_idx = random.choice(context_indices)
        context_word = self.data[context_idx]
        
        return torch.tensor(center_word), torch.tensor(context_word)

    def get_negatives(self, batch_size, n_neg=5):
        # Vectorized negative sampling
        return torch.from_numpy(
            np.random.choice(len(self.vocab), size=(batch_size, n_neg), p=self.unigram_dist)
        )
