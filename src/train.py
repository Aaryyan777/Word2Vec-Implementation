import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from dataset import Text8Dataset
from model import SkipGramModel

# Hyperparameters
BATCH_SIZE = 1024 # Larger batch size for efficiency
EMBED_DIM = 100
EPOCHS = 5
LEARNING_RATE = 0.001
VOCAB_SIZE = 10000 # Restricted for speed on single machine demonstration
WINDOW_SIZE = 5
N_NEG = 5

def train():
    # Paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, 'data', 'text8')
    model_save_path = os.path.join(base_dir, 'word2vec_model.pth')
    embed_save_path = os.path.join(base_dir, 'embeddings.npy')
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Dataset
    dataset = Text8Dataset(data_path, vocab_size=VOCAB_SIZE, window_size=WINDOW_SIZE)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0) # num_workers=0 for Windows compatibility

    # Model
    model = SkipGramModel(len(dataset.vocab), EMBED_DIM).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Training Loop
    print("Starting training...")
    epoch_losses = []
    for epoch in range(EPOCHS):
        total_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for center, context in progress_bar:
            center = center.to(device)
            context = context.to(device)
            negatives = dataset.get_negatives(len(center), N_NEG).to(device)
            
            optimizer.zero_grad()
            loss = model(center, context, negatives)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())
            
        avg_loss = total_loss / len(dataloader)
        epoch_losses.append(avg_loss)
        print(f"Epoch {epoch+1} Average Loss: {avg_loss:.4f}")

    # Save
    print("Saving model and embeddings...")
    torch.save(model.state_dict(), model_save_path)
    
    embeddings = model.get_embeddings()
    np.save(embed_save_path, embeddings)
    
    # Save vocab for later use
    with open(os.path.join(base_dir, 'vocab.txt'), 'w') as f:
        for word in dataset.vocab:
            f.write(f"{word}\n")

    # Save training stats
    import json
    with open(os.path.join(base_dir, 'training_stats.json'), 'w') as f:
        json.dump({'loss_history': epoch_losses}, f)

    print("Done!")

if __name__ == "__main__":
    train()
