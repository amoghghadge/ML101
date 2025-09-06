### Decoder Transformer High Level Architecture

- Take in some sequence of words ("write me a poem") and outputs vector of size vocabulary_size (how many different tokens can the model recognize) where every entry is a number between 0 and 1 representing a probability that the respective token comes next in the sequence

- Given these probabilities, we sample which token comes next and append it to the starting sequence, then this gets fed back into the model/decoder again and the process repeats


### Pretraining Transformers

- Pretraining means you feed in (over some number of iterations) a giant body of text / corpus (like all of wikipedia / all of internet)

- Within a given sentence / body of text, there are many training examples for the model to learn from. If we passed in "write me a poem" (assuming it was somewhere on the internet and made it into the training dataset), it contains a ton of example: given the context "write", "me" should come next; given the context "write me", "a" should come next; given the context "write me a", "poem" should comes next. The same process of breaking down sequences into examples is repeated for all sequence in the corpus


### Using a Pretrained Model

- They're trained to be really good next token predictors

- You wouldn't give it a command like "write me a poem", instead you would pass in "here is a poem: " and the model just wants to keep completing what comes next, so it would start generating a poem (allowing us to achieve the desired functionality without additionally training the model to respond to commands and give the appropriate answer)


### Fine-Tuning

- Additional training of a model with a different dataset

- For ChatGPT, the pre-training dataset might've been the entire internet, but the fine-tining dataset was a smaller question and answer dataset (consisting of conversations between real humans, where the humans respond to the questions in the dataset in a logical way)

- Further iterations of gradient descent / training (after pretraining is over) using the new examples so the model learns to predict the next token based on the smaller dataset

- Then model will be able to respond to questions and mimic human answers

- Fine tuning is just switching the dataset and doing further training

- We could then feed in "write me a poem" and the model would respond to these types of questions and commands

- Fine-tuning also has other use cases like changing the fundamental behavior of the model (could make the model always talk like the president, without prompting it to, by using a new dataset like a large corpus of various conversations/speeches by the president - the model would retain its ability to understand english langauge, but be specialized to predict the next token in these customized sequences)

- Can also be used to inject new knowledge. Sometimes ChatGPT doesn't know about specific new information because it wasn't originally there in the dataset when the model got trained. Effective for company knowledge that just needs to be injected one time and won't change over time

- Another use case of fine-tuning is specialization (like making GPT particularly good at math problems or leetcode / competitive programming). We can choose a dataset that is highly specific to our specialization task / use case, and its performance would increase as it would learn from the examples to better predict the next token for that use case


### Tokenization

- The process of converting strings into numbers. The model needs to process text, but the models only understand numbers to do matrix multiplications and vector calculations. They need to encode character / word / however we choose to break up text as numbers

- We need a consistent mapping where every single token always gets assigned the same integer, and then feed these numbers into the model

- Word level tokenization means every single word be a different number in our vocabulary / mapping / dictionary, character level tokenization so every single character will be its own entry (model will read in one character at a time and also generate one character at a time) - spaces would also be a certain character in the mapping for the model to seperate the words in its response and be comprehensible

- What's actually used by chat gpt and top language models is subword level tokenization


### Subword Level Tokenization

- Word level tokenization may not be optimal because the responses given by our model could be bland as the model predicts entire words at once, leads to more generic output

- A lot of words have the same prefix / suffix (they start with the same few characters and/or end with the same few characters), so we shouldn't restrict ourselves to generate entire words at a time

- Going down to character level and generating one character at a time gives the model a large chance to generate different / diverse words. For example after generating a "b" and "r", it could still keep going down many different paths and generating words like "bruh" or "brother" or "brain". The model is not deterministic - it won't always output the same thing. We deal with probabilities so something different can be chosen for each token generation. 

- Character model is however a bit redundant, because we know so many words will have the same few groups of characters at the start or end (prefixes / suffixes), along with common roots for words

- Subword level tokenization takes an approach in between word level and character level tokenization - taking a few common characters at a time (common prefixes, common suffixes, and common roots) to be stored as tokens in our vocabulary. This avoids redundancy while still allowing diverse words to be generated

- https://platform.openai.com/tokenizer is a playground to see how a tokenizer chunks up a sequence of words or sentence into tokens

- If we feed in "this is a bruh moment", it gets broken up as "this", "is", "a", "br", "uh", "moment"

- Many words start / end with "br" / "uh"


### Gradient Descent Review

- Algorithm used to train neural networks. At its core is a minimization algorithm - used to find the minimum of a function

- We use it to find the minimum of the error / loss function so our model can learn the relationship within our training data

- Is an iterative algorithm to approximate the minimization of our loss function

- At every iteration, we update the weights of our model (the parameters influencing our model's prediction)

- We update the weights by doing the new_weight = old_weight - alpha * d

- Alpha (learning rate) is usually a constant like 0.001 or 0.01 to reflect how fast / drastically we want to update / change our weights at every iteration. Small learning rate takes many iterations to converge / training process to complete. High learning rate causes weights to change too much every iteration and not end up with a well performing model

- d is a 