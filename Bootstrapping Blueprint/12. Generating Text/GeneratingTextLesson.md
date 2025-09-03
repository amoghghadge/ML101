### Generating Text from LLMs

- Let's treat the model as a black box, some trained GPT (takes in tensor of B x T - B is batch size, independent requests we're processing in parallel, which we'll say is 1 for generating text, and T is length of input sequence; outputs a tensor of B x T x V where V is vocabulary size, number of unique words/tokens model recognizes, because for each time step we are predicting which token comes next)

- We generate text by sampling from the output probability distribution for which token comes next, appending the chosen token to the original input, and then calling the model again on this new input and repeating this entire process over and over

- For the coding exercise, instead of passing in an instruction, we pass in a start token (a dummy token that tells the model to start generating text). This is how text is generated from language models when they're in their pre trained state, before we fine tune them on a Q & A dataset to talk back and forth with the model (but it still does well)

- Start token / initial context will be 0 in a given tensor

- After calling the model once, output will be 1 x 1 x V (assuming a batch size of 1 and T equals 1)

- After calling model twice, output will be B x 2 x V; after 3 times outpit will be B x 3 x V

- We only care about the final time step (the next token predicted in the sequence based on )