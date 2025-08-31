import torch
import torch.nn as nn
from torchtyping import TensorType

# Even though the original diagram created by Google 
# has "Norm" after Attention in the bottom component, and 
# "Norm" after FeedForward in the top component, Norm should
# be applied first in both cases (before Attention & before FeedForward),
# and in each case, the output (specifically the output of attention
# in the first case & output of FeedForward in the second case) should
# be added to the tensor passed in to Norm. Researchers have found this
# architecture to be superior for LLM performance.
class TransformerBlock(nn.Module):
    def __init__(self, model_dim: int, num_heads: int):
        # attention_dim and embedding_dim are both model_dim 
        super().__init__()
        torch.manual_seed(0)
        # need 2 seperate layer norms to learn different parameters across training
        self.first_norm = nn.LayerNorm(model_dim)
        self.second_norm = nn.LayerNorm(model_dim)
        self.attention = self.MultiHeadedSelfAttention(model_dim, num_heads)
        self.feedforward = self.VanillaNeuralNetwork(model_dim)

    def forward(self, embedded: TensorType[float]) -> TensorType[float]:
        # Round answer to 4 decimal places
        # embedded is B x T x C
        torch.manual_seed(0)
        attention_output = self.attention(self.first_norm(embedded)) + embedded           # would've be self.norm(self.attention(embedded) + embedded) for add and norm, but we're basically doing norm and add
        # attention_output is now B x T x A
        feedforward_output = self.feedforward(self.second_norm(attention_output)) + attention_output
        # feedforward_output stays B x T x A
        return torch.round(feedforward_output, decimals=4)

    class MultiHeadedSelfAttention(nn.Module):
        def __init__(self, model_dim: int, num_heads: int):
            super().__init__()
            torch.manual_seed(0)
            self.att_heads = nn.ModuleList()
            for i in range(num_heads):
                self.att_heads.append(self.SingleHeadAttention(model_dim, model_dim // num_heads))

        def forward(self, embedded: TensorType[float]) -> TensorType[float]:
            head_outputs = []
            for head in self.att_heads:
                head_outputs.append(head(embedded))
            concatenated = torch.cat(head_outputs, dim = 2)
            return concatenated
        
        class SingleHeadAttention(nn.Module):
            def __init__(self, model_dim: int, head_size: int):
                super().__init__()
                torch.manual_seed(0)
                self.key_gen = nn.Linear(model_dim, head_size, bias=False)
                self.query_gen = nn.Linear(model_dim, head_size, bias=False)
                self.value_gen = nn.Linear(model_dim, head_size, bias=False)
            
            def forward(self, embedded: TensorType[float]) -> TensorType[float]:
                k = self.key_gen(embedded)
                q = self.query_gen(embedded)
                v = self.value_gen(embedded)

                scores = q @ torch.transpose(k, 1, 2) # @ is the same as torch.matmul()
                context_length, attention_dim = k.shape[1], k.shape[2]
                scores = scores / (attention_dim ** 0.5)

                lower_triangular = torch.tril(torch.ones(context_length, context_length))
                mask = lower_triangular == 0
                scores = scores.masked_fill(mask, float('-inf'))
                scores = nn.functional.softmax(scores, dim = 2)

                return scores @ v
    
    class VanillaNeuralNetwork(nn.Module):
        def __init__(self, model_dim: int):
            super().__init__()
            torch.manual_seed(0)
            self.up_projection = nn.Linear(model_dim, model_dim * 4)
            self.relu = nn.ReLU()
            self.down_projection = nn.Linear(model_dim * 4, model_dim)
            self.dropout = nn.Dropout(0.2) # using p = 0.2
        
        def forward(self, x: TensorType[float]) -> TensorType[float]:
            torch.manual_seed(0)
            return self.dropout(self.down_projection(self.relu(self.up_projection(x))))

model_dim=4
num_heads=2
embedded=torch.randn(2,2,4)

model = TransformerBlock(model_dim, num_heads)
print(model(embedded))
# 2 x 2 x 4 tensor
# tensor([
#           [[ 0.0548, -2.4181, -0.3976,  1.0065],
#            [-1.5576, -0.7325, -0.4213,  0.6445]],

#           [[ 1.2309,  2.2142,  0.3611, -1.4108],
#            [-0.0208,  1.0229, -0.1482, -0.0578]]
#       ])