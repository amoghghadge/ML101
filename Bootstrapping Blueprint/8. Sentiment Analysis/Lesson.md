### Sentiment Analysis

- Given text, we want to output if it is positive or negative

- Uses embeddings

### Embeddings

- Learning a vector representation of every word/token in our total set of words through training

- Happens after words get tokenized and encoded into integers, these integers need to then map to a vector so the model understands its actual meaning

- Embedding dimension represents size of vector we learn for each token - the higher this is the more complex of a relationship our model can pick up on

- The numbers in an embedding vector represents the weights, and these are at first randomly intialized, but then updated over training as we minimize loss

- After training, plotting embeddings will cause words that are similar to be close to each other (shows model has learned some sort of meaning for every token)

### Sentiment analysis NN Model Architecture

- Input is B x T where B is batch size (how many examples we're independently processing in parallel) and T is length of longest sentence (tensor is padded with 0s to be rectangular)

- Then we have an embedding layer (nn.Embedding(vocab_size, embedding_dimension)) that outputs a B x T x embedding dimension tensor, because each individual token gets converted to an embedding vector of size embedding dimension - referred to as lookup table in documentation

- Then we do an averaging to get a B x embedding dimension tensor (we are average the meaning of each word in the sentence to get the overall meaning)

- Then we apply a linear layer with a single neuron (out_features is 1) to get a tensor of B x 1: for every single element/sentence we now have a single number to interpret how positive or negative that sentence is

- Finally we have a sigmoid layer to transform each output into a number between 0 and 1 where 1 is completely positive and 0 is completely negative

### Lookup table / embedding layer output explained

One way:
- First one hot encode the input tokens: each row will represent an input token and the number of columns is vocab size. For each row every column will be 0 except the index that token the row represents maps to (which will have a value of 1). Size of this tensor is T x vocab_size.

- Then lookup table will be vocab_size x embedding_dim. Each row contains learned/trained feature representation of row's respective token

- When you multiply these two matrices together (row dot column) you end up extracting the feature representation for each input token: result is T x embedding_dim

Extension:
- Use connected neural network layers where the first layer has vocab_size neurons (where every neuron is either 1 or 0 depending on whether or not we have that token in our input) and next layer has embedding_dim neurons

- nn.Embedding is actually a wrapper built on top of nn.Linear where in_features is vocab_size and out_features is embedding_dim