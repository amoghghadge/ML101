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

- Can be represented as a linear layer (nn.Linear, embedding_dim, att_dim) to generate the queries for every single token

#### What is K (Key)?

- Every token also emitts a key vector (of size attention_dim) representing the information it has

- The queries are then matched up with keys, pairing up tokens that are relevant and then learning the T x T scores

- Will also be represented as a linear layer (nn.Linear, embedding_dim, att_dim) which has its trainiable weights and biases

#### Computation

- We take the query matrix (which is T x A) and matrix multiply that with K transpose (which becomes A x T) to get T x T attention scores

- In first matrix each row is the query, in second matrix each column is the key

- In effect this makes each token's query vector get dot producted with the key vector for every token, to essentially match up and answer questions / create token pairings

- Dot product is a measure of how similar two vectors are to each other, makes sense in the context of the matrix multiplication described above

- The tokens whose queries and keys match up have a higher value coming out of the dot product, and essentially are important to each other

#### Why dot product represents similarity

- For example vectors [1, 0] and [0, 1] when plotted are completely perpendicular (they are not close at all and their dot product is 0)

- Vectors [3, 2] and [2, 3] when plotted actually are close together, and have a dot product score of 12 (3 * 2 + 2 * 3)

- Vectors [3, 3] and [3, 3] which are exactly identical have an even higher dot product of 18

#### Softmax normalization

- After generating T x T tensor we apply softmax (which kind of acts like a multi-dimensional sigmoid) to squash everything to be between 0 and 1

- It will also make everything positive (because it uses exponential function) and sum to 1 (because each value is divided by the total vector sum)

- Is applied to every row in the T x T tensor of attention scores, so every row sums to 1 and every entry in each row is positive and between 0 and 1

- Given a row, which corresponds to a token, each column in that row represents probability of the column token's relevance to the row's token

#### Final layer output

- Once we have the normalized T x T scores, we multiply it by V (instead of the T x C input like the simple aggregation example)

- This is done because every token will also actually emit a value vector of size attention_dim (also learned and trained with a linear layer)

- This adds another level of complexity to the model: if the query is what a token searching for and the key is what the token actually has, we want the value to represent what is the token actually willing to share

- There are various pieces of information associated with every token (in the key) but the value is what information is actually relevant and that I actually want to share

- We don't want to actually share everything to the entire unmasked (T x C) input

- We instead allow V to be learned so the model can understand for every token what information is actually relevant to share with the other tokens

- This lets the model decouple where to look (K and Q) from what actual content is passed forward (V)

- Matrix multiplication becomes lookup/aggregation of each weighting of token relationships for a sequence (in T x T matrix of scores) with how much each of those tokens are actually willing to share (in T x A matrix V) in attention_dim vector space

- Essentially results in T x A output of what each token needs to pay attention to based on all tokens that came before it

#### Why divide by sqrt(dk) before we apply softmax

- dk = attention_dim

- This is just a scale factor researchers found to prevent NNs suffer from exploding/vanishing gradient (where values of derivatives during training either get way too big or way too small)