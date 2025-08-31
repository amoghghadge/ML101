### Transformer Block

#### Add

- Refers to skip / residual connections

- We have some arbitrary layer (linear, attention, etc) that takes in input x and gives some output

- Instead of just using the output of the layer, we also let some portion of x bypass the layer and get added to the layer's output

- The forward method for a layer incorporating skip connections would return layer(x) + x (the forward method of the layer with x added to it)

- The model will learn the right weights/biases for the layer to send the right proportion of x forward, instead of completely transforming x we want to retain its original identity and pass most of it / incorporate it into the output for the layer

- Through training the model figures out how much of x should be passed through / sent through this layer

- We need to actually add x because it solves the exploding/vanishing gradient/derivative problem - in a NN with tons of layers the gradients become super big or small, causing parameters to either change too much or change too little

- If you have some function as the sum of two other functions, f(x) = g(x) + h(x), the derivative is f'(x) = g'(x) + h'(x), and using an additive term instead of a multiplication helps prevent compounding effects causing a drastic increase or decrease while calculating the necessary gradients for the layer


#### Norm

- Refers to layer normalization, its own module in pytorch instantiated with nn.LayerNorm(embedding_dim) and embedding_dim is the dimension along which it will normalize

- Normalizing means recentering our data to revolve around the mean, looking like a bell curve

- Done by taking every data point, subtracting the mean, and dividing by the standard deviation: (x - H) / sigma

- We actually also multiply the previous formula by gamma and add some number beta: gamma and beta are adjustable and learnable across the iterations of gradient descent to let the NN have some learning capacity and prevent the layer output from just being a strict, deterministic formula

- Improves performance because the NN starts off with random parameters but if there's a drastic shift in the nature of the data during training, as measured by mean and standard deviation, it becomes much slower to train and not as effective to converge. So centering the activation / layer output around the mean with some adjustability makes it easier to train

- The data fed into the layer is T x C (context length x embedding dim), and the data gets normalized along the embedding dimension to normalize the embedding / feature vector for each time step / token


#### Masked Multi-Headed Attention

- The masked part means because the model is predicting the next token in sequence, it's not allowed to look at future tokens

- Given a sequence like "Write me a poem", it multiple training examples: given "write", "me" comes next; given "write me", "a" comes next; given "write me a", "poem" comes next

- Masked means if the model was tasked with predicting "poem" and had the context of "write me a", our code will mask out the word "poem" so the model can't see the answer before making its prediction

- Attention lets the tokens talk to each other to figure out which pair of tokens are important or not to then generate a new feature vector that's a weighted aggregation of the previous tokens based on their importance

- Multi-Headed component refers to how the model concatenate's the result of multiple attention heads / layers, each operating on the same entire input


#### Feed Forward

- Traditional / Vanilla Fully Connected Neural Network

- Only contains linear layers (with an arbitrary number of nodes/neurons), nonlinear activations (like ReLU), and dropout

- First having communication in attention layers and then having computation in vanilla neural network is highly effective for model's performance / stability to predict the next token


#### Final Linear & Softmax

- These are used to get an interpretable prediction

- Linear layer is nn.Linear(att_dim, vocab_size) because for every single token in the input sequence we need to predict what token comes next out of all possible tokens (vocab size)

- Final output is a vector of size vocab_size where we interpret each entry in the vector as the probability that the corresponding index's token comes next in the sequence

- Softmax is what squashes the values to be between 0 and 1 as well as all sum up to 1, allowing us to interpret the vector as a series of probabilities

- All these parts make up the decoder transformer