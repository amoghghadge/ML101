import torch
import torch.nn as nn
from torchtyping import TensorType

# 0. Instantiate the linear layers in the following order: Key, Query, Value.
# 1. Biases are not used in Attention, so for all 3 nn.Linear() instances, pass in bias=False.
# 2. torch.transpose(tensor, 1, 2) returns a B x T x A tensor as a B x A x T tensor.
# 3. This function is useful:
#    https://pytorch.org/docs/stable/generated/torch.nn.functional.softmax.html
# 4. Apply the masking to the TxT scores BEFORE calling softmax() so that the future
#    tokens don't get factored in at all.
#    To do this, set the "future" indices to float('-inf') since e^(-infinity) is 0.
# 5. To implement masking, note that in PyTorch, tensor == 0 returns a same-shape tensor 
#    of booleans. Also look into utilizing torch.ones(), torch.tril(), and tensor.masked_fill(),
#    in that order.
class SingleHeadAttention(nn.Module):
    def __init__(self, embedding_dim: int, attention_dim: int):
        super().__init__()
        torch.manual_seed(0)
        self.key = nn.Linear(embedding_dim, attention_dim, bias=False)
        self.query = nn.Linear(embedding_dim, attention_dim, bias=False)
        self.value = nn.Linear(embedding_dim, attention_dim, bias=False)
    
    def forward(self, embedded: TensorType[float]) -> TensorType[float]:
        # Return your answer to 4 decimal places
        
        # input is B x T x C
        # queries is B x T x A, keys is B x A x T
        queries = self.query(embedded)
        B, T, C, A = embedded.shape[0], embedded.shape[1], embedded.shape[2], queries.shape[2]
        keys = torch.transpose(self.key(embedded), 1, 2)

        # generate scores of size B x T x T
        scores = queries @ keys
        scores /= (A ** 0.5)

        # do masking
        mask = torch.tril(torch.ones(T, T)) == 0
        scores.masked_fill_(mask, float('-inf'))

        # apply softmax
        scores = torch.nn.functional.softmax(scores, dim=2)
        
        # get values of B x T x A
        values = self.value(embedded)

        # get layer's output of B x T x A
        out = scores @ values
        return torch.round(out, decimals=4)
    
embedding_dim = 3
attention_dim = 4
embedded = torch.randn(2, 2, 3)

model = SingleHeadAttention(embedding_dim, attention_dim)
print(model(embedded))
# 2 x 2 x 4 tensor
# tensor([
#          [[-6.2060e-01, -3.0590e-01,  9.0000e-04,  3.6430e-01],
#           [-1.8000e-03, -2.5020e-01,  3.3730e-01, -2.3510e-01]],
# 
#          [[ 1.1870e-01, -2.9230e-01, -8.6350e-01, -1.8560e-01],
#           [ 2.8840e-01,  5.5500e-02, -1.1352e+00, -4.8900e-02]]
#       ])