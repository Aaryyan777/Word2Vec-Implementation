import os
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class Word2VecEvaluator:
    def __init__(self, base_dir):
        self.base_dir = base_dir
        self.embeddings = np.load(os.path.join(base_dir, 'embeddings.npy'))
        
        with open(os.path.join(base_dir, 'vocab.txt'), 'r') as f:
            self.vocab = [line.strip() for line in f]
            
        self.word2idx = {word: idx for idx, word in enumerate(self.vocab)}
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}
        
    def get_vector(self, word):
        if word not in self.word2idx:
            raise ValueError(f"Word '{word}' not in vocabulary")
        return self.embeddings[self.word2idx[word]].reshape(1, -1)
    
    def most_similar(self, word, top_k=5):
        try:
            vec = self.get_vector(word)
        except ValueError:
            return []
            
        sims = cosine_similarity(vec, self.embeddings)[0]
        # Sort desc
        indices = sims.argsort()[::-1][1:top_k+1] # Skip self
        
        return [(self.idx2word[idx], sims[idx]) for idx in indices]

    def analogy(self, a, b, c, top_k=5):
        """ a - b + c = ? (e.g., king - man + woman = queen) """
        try:
            vec_a = self.get_vector(a)
            vec_b = self.get_vector(b)
            vec_c = self.get_vector(c)
        except ValueError as e:
            print(e)
            return []
            
        # Target vector
        vec_target = vec_a - vec_b + vec_c
        
        sims = cosine_similarity(vec_target, self.embeddings)[0]
        indices = sims.argsort()[::-1][:top_k]
        
        return [(self.idx2word[idx], sims[idx]) for idx in indices]

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    evaluator = Word2VecEvaluator(base_dir)
    
    print("\n--- Similarity Tests ---")
    words = ["one", "four", "good", "american", "france"]
    for w in words:
        print(f"\nMost similar to '{w}':")
        for res, score in evaluator.most_similar(w):
            print(f"  {res}: {score:.4f}")
            
    print("\n--- Analogy Tests ---")
    # Note: Accuracy depends heavily on training time and vocab size.
    # With small vocab and short training, results might be noisy.
    analogies = [
        ("king", "man", "woman"),
        ("paris", "france", "germany"),
        ("better", "good", "bad")
    ]
    
    for a, b, c in analogies:
        print(f"\n{a} - {b} + {c} = ?")
        results = evaluator.analogy(a, b, c)
        for res, score in results:
            print(f"  {res}: {score:.4f}")
