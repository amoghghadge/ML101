import torch
import torch.nn as nn
from torchtyping import TensorType

class Solution(nn.Module):
    def __init__(self, vocabulary_size: int):
        super().__init__()
        torch.manual_seed(0)
        self.embedding = nn.Embedding(vocabulary_size, 16)
        self.linear_layer = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: TensorType[int]) -> TensorType[float]:
        # Hint: The embedding layer outputs a B, T, embed_dim tensor
        # but you should average it into a B, embed_dim tensor before using the Linear layer
        
        # Return a B, 1 tensor and round to 4 decimal places

        # x is B x T
        out = self.embedding(x)
        # out is now B x T x embed_dim
        out = torch.mean(out, dim=1)    # average across the columns to get the average embedding vector for each sentence / row in B
        # decision to average the embedding vectors makes this a "Bag of Words" model
        # out is now B x embed_dim
        out = self.linear_layer(out)
        # out is now B x 1
        out = self.sigmoid(out)
        return torch.round(out, decimals=4)
