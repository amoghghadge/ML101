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
        keys = torch.transpose(self.key(embedded), 1, 2)

        # generate scores of size B x T x T
        scores = queries @ keys
        scores /= (queries.shape[2] ** 0.5)

        # do masking
        mask = torch.tril(torch.ones(embedded.shape[1], embedded.shape[1])) == 0
        scores.masked_fill_(mask, float('-inf'))

        # apply softmax
        scores = torch.nn.functional.softmax(scores, dim=2)
        
        # get values of B x T x A
        values = self.value(embedded)

        # get layer's output of B x T x A
        out = scores @ values
        return torch.round(out, decimals=4)