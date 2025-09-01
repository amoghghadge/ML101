import torch
import torch.nn as nn
from torchtyping import TensorType

# 1. Remember to include an additional LayerNorm after the block sequence and before the final linear layer
# 2. Instantiate in the following order: Word embeddings, position embeddings, transformer blocks, final layer norm, and vocabulary projection.
class GPT(nn.Module):
    def __init__(self, vocab_size: int, context_length: int, model_dim: int, num_blocks: int, num_heads: int):
        super().__init__()
        torch.manual_seed(0)
        # Hint: nn.Sequential() will be useful for the block sequence
        self.word_embedding = nn.Embedding(vocab_size, model_dim)
        self.position_embedding = nn.Embedding(context_length, model_dim)

        self.transformer_blocks = nn.Sequential()
        for i in range(num_blocks):
            self.transformer_blocks.append(self.TransformerBlock(model_dim, num_heads))

        self.norm = nn.LayerNorm(model_dim)
        self.linear = nn.Linear(model_dim, vocab_size)

    def forward(self, context: TensorType[int]) -> TensorType[float]:
        torch.manual_seed(0)
        # Round answer to 4 decimal places
        # context is B x T
        B, T = context.shape[0], context.shape[1]

        first_embedding = self.word_embedding(context)
        # first_embedding is B x T x C
        positions = torch.tensor([[i for i in range(T)] for _ in range(B)])
        # positions is B x T but contains what position the token is at instead of what the token actually is / what word it represents (as is the case for context)
        second_embedding = self.position_embedding(positions)
        # second_embedding is also B x T x C
        final_embedding = first_embedding + second_embedding
        # final_embedding stays B x T x C
        blocks_output = self.transformer_blocks(final_embedding)
        # blocks_output is now B x T x A
        norm_output = self.norm(blocks_output)
        # norm_output stays B x T x A
        linear_output = self.linear(norm_output)
        # linear_output is now B x T x V
        token_probs = nn.functional.softmax(linear_output, dim=2)
        # token_probs stays B x T x V and now represents probability of each token occuring next
        return torch.round(token_probs, decimals=4)

    # Do NOT modify the code below this line
    class TransformerBlock(nn.Module):
        def __init__(self, model_dim: int, num_heads: int):
            super().__init__()
            torch.manual_seed(0)
            self.attention = self.MultiHeadedSelfAttention(model_dim, num_heads)
            self.linear_network = self.VanillaNeuralNetwork(model_dim)
            self.first_norm = nn.LayerNorm(model_dim)
            self.second_norm = nn.LayerNorm(model_dim)

        def forward(self, embedded: TensorType[float]) -> TensorType[float]:
            torch.manual_seed(0)
            embedded = embedded + self.attention(self.first_norm(embedded)) # skip connection
            embedded = embedded + self.linear_network(self.second_norm(embedded)) # another skip connection
            return embedded
        
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
            
vocab_size = 3
context_length = 2
model_dim = 4
num_blocks = 2
num_heads = 2
context = [[2, 0], [2, 0]]

model = GPT(vocab_size, context_length, model_dim, num_blocks, num_heads)
print(model(context))
# [ 
#   [
#       [0.4160,0.4528,0.1312],
#       [0.4180,0.3717,0.2103]
#   ],
#   [
#       [0.4020,0.4722,0.1258],
#       [0.4181,0.3717,0.2102]
#   ]
# ]
# 2 x 2 x 3