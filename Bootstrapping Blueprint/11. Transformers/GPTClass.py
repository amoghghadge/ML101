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
        # size is (batch_size, context_length, vocab_size)
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
context = torch.tensor([[2, 0], [2, 0]], dtype=torch.long)

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

# --- Data Preparation ---

# 1. Create a simple dataset and vocabulary (using dataset of all of Shakespeare's writings)
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()
chars = sorted(list(set(text)))
vocab_size = len(chars)

string_to_int = { ch:i for i,ch in enumerate(chars) }
int_to_string = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [string_to_int[c] for c in s]
decode = lambda l: ''.join([int_to_string[i] for i in l])

# 2. Create the input and target tensors
# The model will learn to predict the next character in the sequence.
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # 90% for training, 10% for validation (not used in this simple loop)
train_data = data[:n]
val_data = data[n:]

# 3. Define a function to get input/target batches
context_length = 8 # Let's use a context length of 8 for this example
batch_size = 4

def get_batch(split):
    data = train_data if split == 'train' else val_data
    # Generate random starting points for our batches
    ix = torch.randint(len(data) - context_length, (batch_size,))
    # Create the input tensors (x)
    x = torch.stack([data[i:i+context_length] for i in ix])
    # Create the target tensors (y), which are shifted by one position
    y = torch.stack([data[i+1:i+context_length+1] for i in ix])
    return x, y

# Test the batch function
xb, yb = get_batch('train')
print("--- Sample Batch ---")
print("inputs (xb):")
print(xb)
print("\ntargets (yb):")
print(yb)
print("--------------------")

# --- Model, Optimizer, and Loss Setup ---

# Model Hyperparameters
# NOTE: We are overwriting the small values from your initial test case
# with more realistic ones for training.
vocab_size = len(chars)
context_length = 8      # Max length of a sequence
model_dim = 32          # Embedding dimension
num_blocks = 3          # Number of transformer blocks
num_heads = 4           # Number of attention heads
learning_rate = 1e-3    # Learning rate for the optimizer

# 1. Instantiate the model
model = GPT(
    vocab_size=vocab_size,
    context_length=context_length,
    model_dim=model_dim,
    num_blocks=num_blocks,
    num_heads=num_heads
)

# 2. Define the loss function
# CrossEntropyLoss is perfect for multi-class classification like ours.
criterion = nn.CrossEntropyLoss()

# 3. Instantiate the Adam optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

print("\n--- Model & Optimizer Ready ---")
print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters.")
print("Optimizer: AdamW")
print("Loss Function: CrossEntropyLoss")
print("-----------------------------")

# --- The Training Loop ---

# Training Hyperparameters
epochs = 15000
eval_interval = 500 # How often to print the loss

print("\n--- Starting Training ---")

for epoch in range(epochs):

    # 1. Get a batch of data
    xb, yb = get_batch('train')

    # 2. Get model predictions
    logits = model(xb) # The forward pass

    # 3. Calculate the loss
    # PyTorch's CrossEntropyLoss expects logits of shape (Batch, Classes, Sequence_Length)
    # and targets of shape (Batch, Sequence_Length). We need to reshape our tensors.
    B, T, C = logits.shape
    logits_reshaped = logits.view(B*T, C)
    targets_reshaped = yb.view(B*T)
    loss = criterion(logits_reshaped, targets_reshaped)

    # 4. Backpropagation
    optimizer.zero_grad(set_to_none=True) # Zero out old gradients
    loss.backward() # Calculate new gradients

    # 5. Update weights
    optimizer.step()

    # Print loss every so often
    if epoch % eval_interval == 0:
        # Note: In a real project, you'd calculate validation loss here
        # using a separate evaluation loop with torch.no_grad()
        print(f"Epoch {epoch}, Training Loss: {loss.item():.4f}")

print(f"Final Loss: {loss.item():.4f}")
print("--- Training Complete ---")

# --- Inference / Generation ---

def generate(model, start_string, max_new_tokens):
    model.eval() # Set the model to evaluation mode
    
    # Convert the starting string to a tensor of tokens
    tokens = torch.tensor(encode(start_string), dtype=torch.long)
    tokens = tokens.unsqueeze(0) # Add a batch dimension

    for _ in range(max_new_tokens):
        # Ensure context is not longer than context_length
        # We only need the last `context_length` tokens for the prediction
        context = tokens[:, -context_length:]
        
        # Get the model's predictions (logits)
        with torch.no_grad():
             logits = model(context)
        
        # We only care about the prediction for the very last token in the sequence
        last_logits = logits[:, -1, :] # Becomes (B, C)
        
        # Apply softmax to get probabilities
        probs = nn.functional.softmax(last_logits, dim=-1)
        
        # Sample from the probability distribution to get the next token
        next_token = torch.multinomial(probs, num_samples=1)
        
        # Append the new token to our sequence
        tokens = torch.cat((tokens, next_token), dim=1)

    model.train() # Set the model back to training mode
    return decode(tokens[0].tolist())

# Let's generate some text!
start_character = "h"
generated_text = generate(model, start_string=start_character, max_new_tokens=20)

print("\n--- Generating Text ---")
print(f"Starting with: '{start_character}'")
print(f"Generated sequence: {generated_text}")
print("-----------------------")