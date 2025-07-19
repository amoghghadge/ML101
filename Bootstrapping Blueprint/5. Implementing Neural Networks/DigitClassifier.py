import torch
import torch.nn as nn
from torchtyping import TensorType

class Solution(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        # Define the architecture here
        self.first_layer = nn.Linear(784, 512)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.final_layer = nn.Linear(512, 10)
        self.sigmoid = nn.Sigmoid()     # makes all outputs between 0 and 1 to represent a probability
    
    def forward(self, images: TensorType[float]) -> TensorType[float]:
        torch.manual_seed(0)
        # Return the model's prediction to 4 decimal places
        out = self.sigmoid(self.final_layer(self.dropout(self.relu(self.first_layer(images)))))
        return torch.round(out, decimals=4)