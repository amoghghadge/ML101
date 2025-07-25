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
        embedded = self.embedding(x)
        # embedded is now B x T x embed_dim
        averaged = torch.mean(embedded, dim=1)    # average across the columns to get the average embedding vector for each sentence / row in B
        # decision to average the embedding vectors across each word in a sentence makes this a "Bag of Words" model
        # averaged is now B x embed_dim
        projected = self.linear_layer(averaged)
        # projected is now B x 1
        predictions = self.sigmoid(projected)
        return torch.round(predictions, decimals=4)

vocabulary_size = 170000

# 2 x 12 tensor
x = torch.tensor([
  [2, 7, 14, 8, 0, 0, 0, 0, 0, 0, 0, 0],    # "The movie was okay"
  [1, 4, 12, 3, 10, 5, 15, 11, 6, 9, 13, 7] # "I don't think anyone should ever waste their money on this movie"
])

model = Solution(vocabulary_size)
print(model(x))
# 2 x 1 output
# tensor([[0.5963],[0.5199]])

# 2 x 2 tensor
x = torch.tensor([[4,1],[2,3]])
print(model(x))
# 2 x 1 output
# tensor([[0.4293],[0.4035]])