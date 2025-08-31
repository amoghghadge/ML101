import torch
import torch.nn as nn
from torchtyping import TensorType

class Solution(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        # Define the architecture here
        self.first_layer = nn.Linear(784, 512)
        self.relu = nn.ReLU()       # introduces nonlinearity to allow model to learn more complex relationships
        self.dropout = nn.Dropout(0.2)
        self.final_layer = nn.Linear(512, 10)
        self.sigmoid = nn.Sigmoid()     # makes each output between 0 and 1 to represent a probability
    
    def forward(self, images: TensorType[float]) -> TensorType[float]:
        torch.manual_seed(0)
        # Return the model's prediction to 4 decimal places
        # images input is B x 784
        out = self.sigmoid(self.final_layer(self.dropout(self.relu(self.first_layer(images)))))
        # out is B x 10
        return torch.round(out, decimals=4)
    
model = Solution()

# 2 x 784 tensor
images = torch.randn(2, 28 * 28)
print(model(images))
# 2 x 10 tensor
# tensor([
#   [0.5130, 0.4453, 0.4893, 0.3846, 0.4986, 0.5436, 0.5704, 0.5216, 0.5180, 0.4198],
#   [0.5867, 0.4286, 0.4800, 0.4313, 0.5041, 0.4104, 0.4052, 0.5152, 0.4727, 0.5591]
# ])