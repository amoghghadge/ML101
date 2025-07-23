### Transformer architecture

- First step is embedding layer, exactly same as sentiment analysis model where B x T input becomes B x T x C (where C is embedding dim)

- How does the model combine/aggregate all embeddings to understand the input (how does it understand the many pairs of relationships between tokens)?

- Simplest way is to average all the rows of embeddings together (referred to as bag of words)

- This simple aggregation can be represented as a matrix multiplication where first matrix is lower triangular and T x T with each element as 1/T, second matrix is a T x C where each row is the respective token's embedding vector: multiplying generates running average vectors accounting for all previous tokens for each row in T x C matrix

- The fact that the first matrix from the simple aggregation's matrix multiplication is T x T actually means something: for row i and col j, the weight at that cell is the strength/affinity/score between token i and j

- We need a more complex aggregation that uses a weighted average to pay attention to some tokens more than others

- The way these weights in the T x T matrix are learned is through self-attention layer

### Self attention layer

- Come up with a T x T tensor of attention weights/scores to figure out how important each token is to every other token

- Is always lower triangular because one token can never see future tokens

- High numbers in a cell means the respective token pairing is important to pay attention to

- We want these weights to be trainable and learnable through gradient descent based on training data

- We then use this T x T, multiply it against T x C embeddings, and send/forward this output to later parts of the NN

### Generating the T x T tensor

- Getting T x T scores = softmax(Q * K^T / sqrt(dk))

- Layer Output: Scores x V

#### What is Q (Query)?

- Every single token in a sentence actually talks to each other by emitting and sending out a vector called a query

- This vector contains information on what that token is searching for and has size attention_dim

- Every token goes from being embedded with size embedding_dim (T x C) to every token emitting a query with size attention dim (T x attention_dim) to represent what the token is looking for

#### What is K (Key)?

- Every token should also emit a key vector (of size attention_dim) representing the information it has

- The queries are then matched up with keys, pairing up tokens that are relevant and then learning the T x T scores