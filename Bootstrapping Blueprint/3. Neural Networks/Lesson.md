### Neural Networks

- Input attributes are first layer in neural network

- Each node in a hidden layer performs linear regression on the output of the previous layer's nodes and applies a non-linear function (usually sigmoid)

- Each node independently is learning its own parameters (w1, w2, ..., wn, b) through training

- Matrix size of parameters for each layer will have number of inputs (ws) as rows and number of nodes in layer as number of columns

- Might be opposite if you decide to process by doing matrix times input from previous layer (matrix would be number of nodes as rows, number of inputs as columns, and matrix youre multiplying with would be number of inputs as rows and 1 column, results in a vertical vector - one column - of what the output value of each node should be)