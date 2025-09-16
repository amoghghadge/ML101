### Low Rank Adaptation

- State of the art technique for fine tuning LLMs in a practical way


### Problems with full fine-tuning

- We would need to load in our full pre-trained model + our fine-tuning dataset, and continuing iterations of training, letting the model adjust potentially all parameters

- Extremely expensive due to how large the models are (GPT-3 has 175 billion parameters, around 1 TB in space), requiring a significant amount of GPUs

- For deploying these models to production, it would be extremely hard to switch between different states of these large language models (may have many different models for different specializations like HR, Coding, and Design) due to them being on the scale of terabytes, so loading them into memory and quickly changing them is very infeasable


### LoRA

- LoRA says let's not do full fine-tuning and instead consider if we even need to update all the parameters in the model

- Every layer (embedding, attention, feed forward, linear, normalization) has tons of learnable parameters, but not all of them are important for the model to learn the intricacies and nuances of the fine-tuning dataset

- We only allow the model to update some of the parameters during the additional training

- The most important layer in a transformer allowing it to understand language and generate text so well is the attention layer (multi-headed attention), so we can just freeze the parameters in the other layers that came from the pre-trained state of the model

- Only the weights inside the attention layer can be updated via gradient descent

- Even within the attention layer, the query key and value linear layers have so many numbers within their matrices - not all of those need to be updated either

- As a refresher, linear layers have each node perform a linear regression based on all the previous layer's features. The layer is represented in code as a matrix with the number of input features as the number of rows and the number of nodes in the layer as the number of cols. Each row dotted with a col represents linear regression being performed for the col's respective node in the layer

- We can actually train a fraction of all the parameters in the weights matrix and still achieve the same performance

- Say we have a weight matrix W of size d x d (for a single linear layer in an LLM, d^2 can easily be in the millions)

- We can rewrite W as the product of two smaller matricies, B and A, where B is d x r and A is r x d (B x A still results in shape of d x d) - r (rank) is a much smaller number than d that we can choose on our own

- W has d^2 parameters, but B & A will have a total of 2 * r * d paramters. The lower of a value we can get away with for r, the less parameters we have to train in total

- If d was 10 and r was 2, originally we would have 100 weights, but B and A together would only have 40 weights, causing us to only have to train 40% of the original layer (during the fine-tuning we only update the values in B and A)

- Our new weights W after we've trained B and A through fine-tuning is just the initial old W (W_0) + B matrix multiplied with A. We use this for future inference (for testing after finishing fine-tuning)


### Does it work well?

- Actually works incredibly well

- We only need to train a small fraction of the total number of parameters

- We can save a ton of money and time (don't need days or weeks to train these models on GPUs)

- We get nearly identical performance (may get 90 or 95% performance compared to if we had just done full fine-tuning of an LLM)

- Lets people without access to super expensive GPUs actually fine-tune LLMs and get them to do what we need with low amounts of compute


### Model Switching

- Since the changes we make to the base model are much smaller than completely re-training the model again, we can actually cache our LoRA modules (B and A matrices) in RAM

- The data transfer for switching between LLM states is just a data transfer between RAM and VRAM

- Since RAM is significantly larger than VRAM, we don't need to read from disk every single time we want to load in a different LoRA model (we just cache a bunch of them in RAM)

- We can start with a base model (general language model but not specialized or fine-tuned). With LoRA we fine-tune the model on English, French, Chinese, Math, answering medical questions, code, code completion, code translation, etc. Because these models / modules are only slightly different from each other, switching between base model and one of the specializations is easy (the only difference is adding or subtracting the matrix B x A from the additional / initial weight matrix W)

- Add BA to go from base model to specialization, subtract BA to go from specialization back to base model

- When deploying to production, we only have to load the base model once, then cache the LoRA changes in RAM to be able to switch back and forth between different LLM states just by doing the addition or subtraction of BA (while avoiding reading the fine-tuned versions from disk every time)


### PEFT Library

- Just like we have transformers library and tokenizer library, there is an open source parameter efficient fine tuning (PEFT) library on the hugging face community

- There's a class in the PEFT library called LoraConfig where we can specify the different options like which layers of the transformer do we want to target / allow to get updated, our value for r (the rank), etc.

- Then there's extremely useful utility function called get_peft_model() which takes in the LoraConfig and the base / pre-trained model that we're going to be fine-tuning

- The above function returns an instance of PeftModel, which is a normal PyTorch model / neural network that we can then do our training loop / gradient descent on and minimze the loss on our new dataset

- Rest of code again makes use of trainer class and calls trainer.train()