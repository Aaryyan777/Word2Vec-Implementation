import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def visualize_training():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    stats_path = os.path.join(base_dir, 'training_stats.json')
    embed_path = os.path.join(base_dir, 'embeddings.npy')
    vocab_path = os.path.join(base_dir, 'vocab.txt')
    plots_dir = os.path.join(base_dir, 'plots')
    
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    # 1. Plot Loss Curve
    if os.path.exists(stats_path):
        print("Generating Loss Curve...")
        with open(stats_path, 'r') as f:
            stats = json.load(f)
        
        losses = stats['loss_history']
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(losses) + 1), losses, marker='o', linestyle='-', color='b', linewidth=2)
        plt.title('Training Loss per Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.savefig(os.path.join(plots_dir, 'loss_curve.png'))
        plt.close()
    else:
        print("No training stats found. Skipping loss curve.")

    # 2. t-SNE Visualization
    if os.path.exists(embed_path) and os.path.exists(vocab_path):
        print("Generating t-SNE Visualization...")
        embeddings = np.load(embed_path)
        with open(vocab_path, 'r') as f:
            vocab = [line.strip() for line in f]
        
        # Select top N words to visualize (to avoid clutter)
        top_n = 300
        vecs = embeddings[:top_n]
        labels = vocab[:top_n]
        
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        reduced_vecs = tsne.fit_transform(vecs)
        
        plt.figure(figsize=(16, 12))
        plt.scatter(reduced_vecs[:, 0], reduced_vecs[:, 1], c='steelblue', edgecolors='k', alpha=0.7)
        
        for i, word in enumerate(labels):
            plt.annotate(word, xy=(reduced_vecs[i, 0], reduced_vecs[i, 1]), 
                         xytext=(5, 2), textcoords='offset points', ha='right', va='bottom', fontsize=9)
            
        plt.title(f't-SNE Visualization of Top {top_n} Word Embeddings')
        plt.axis('off')
        plt.savefig(os.path.join(plots_dir, 'tsne_visualization.png'), dpi=300)
        plt.close()
    else:
        print("Embeddings or vocab not found. Skipping t-SNE.")

if __name__ == "__main__":
    visualize_training()
