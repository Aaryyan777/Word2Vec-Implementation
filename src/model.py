import torch
import torch.nn as nn
import torch.nn.functional as F

class SkipGramModel(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super(SkipGramModel, self).__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        
        # Center word embeddings (Target)
        self.center_embeddings = nn.Embedding(vocab_size, embed_dim)
        
        # Context word embeddings (Output/Context)
        self.context_embeddings = nn.Embedding(vocab_size, embed_dim)
        
        # Initialize weights (small random values)
        self.center_embeddings.weight.data.uniform_(-0.5 / embed_dim, 0.5 / embed_dim)
        self.context_embeddings.weight.data.uniform_(-0.5 / embed_dim, 0.5 / embed_dim)

    def forward(self, center_words, context_words, negative_words):
        """
        center_words: [batch_size]
        context_words: [batch_size]
        negative_words: [batch_size, n_neg]
        """
        # [batch_size, embed_dim]
        center_embeds = self.center_embeddings(center_words)
        
        # [batch_size, embed_dim]
        context_embeds = self.context_embeddings(context_words)
        
        # [batch_size, n_neg, embed_dim]
        neg_embeds = self.context_embeddings(negative_words)
        
        # Positive score: dot product of center and context
        # [batch_size, 1, embed_dim] * [batch_size, embed_dim, 1] -> [batch_size, 1]
        pos_score = torch.bmm(context_embeds.unsqueeze(1), center_embeds.unsqueeze(2)).squeeze()
        pos_score = F.logsigmoid(pos_score)
        
        # Negative score: dot product of center and negatives
        # [batch_size, 1, embed_dim] * [batch_size, embed_dim, n_neg] -> [batch_size, 1, n_neg]
        # We want to minimize similarity, so maximize log(sigmoid(-score))
        neg_score = torch.bmm(neg_embeds, center_embeds.unsqueeze(2)).squeeze()
        neg_score = F.logsigmoid(-neg_score)
        
        # Total loss: negative of the sum (since we want to maximize probability)
        # Sum over negatives, then mean over batch
        return -(pos_score + neg_score.sum(1)).mean()

    def get_embeddings(self):
        # The paper typically uses the center embeddings as the final word vectors
        return self.center_embeddings.weight.data.cpu().numpy()
