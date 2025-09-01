### Decoder

- During training we feed in B x T tensors where B is batch size and T is length of each sentence / context length

#### Token Embedding Layer

- Learn / train a feature vector for each token in vocabulary

- Lookup table of size vocab_size x embedding_dim

- vocab_size is number of unique tokens the model can recognize

- For this we pass in the input tokens themselves and this layer looks up the feature vector for each token


#### Positional Embedding Layer

- Lookup table of size context_length x embedding_dim

- We learn a vector of size embedding_dim for every single possible position in the sequence

- context_length is how many tokens back in the sequence can the model read / account for

- For positional embedding layer we pass in torch.arange(T) - generates a tensor of size T sorted in order where its vales go from 0 to T - 1. This is how we index our embedding lookup table

- Result of token embedding layer gets added to the result of positional embedding layer and then this sum gets feeded in to the transformer block


#### Transformer Block

- Allows model to predict next token in sequence well

- We actually have many of these blocks in sequence, the output of one transformer block is passed back in to another transformer block

- We have Nx number of transformer blocks (predetermined beforehand)

- More blocks makes model more complex and lets it learn more complex relationships with more parameters

- Use nn.Sequential() to implement this in code - treat like a python list and use .append() but list must only contain other neural network layers

- Calling forward on nn.Sequential() calls forward method of every block in the list in order from left to right

- Lets us automate process of passing output of one transformer block as input to next transformer block


#### Final Component

- GPTs actually have an additional layer norm after the last transformer block, this makes the range of values not too extreme and makes training process smoother

- Final linear layer is a vocabulary projection layer, dimensions are attention_dim x vocab_size (the layer has vocab_size neurons) to assign a number to every single token in our vocabulary

- Softmax layer squashes all the previous values to be between 0 and 1 as well as all sum to 1, so each value represents a probability that the respective token comes next (creates true probability distribution)