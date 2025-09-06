### Hugging Face

- Open source ML community with important libraries, datasets, and models

- They have dataset and transformers libraries, we don't have to completely build a model from scratch using PyTorch like we did

- Hugging Face libraries are wrappers on top of PyTorch (like transformers and tokenizers), provide more abstraction to build larger and larger models

- The models on Hugging Face represent the saved weights of a neural network, you can download these matricies representing the layers of the neural network to use the model and run inference

- With a few lines of code you can load in the model into PyTorch or any native python environment and use the model / play around with it as well as modify it / train it some more for fine-tuning on new data


### Transformers Library

- No longer need to rely on GPT class we wrote earlier

- This library is an implementation of transformers which comes with other useful functions like tokenizers, allowing us to build larger and larger models

- pip install transformers datasets

- Gives access to a fantastic implementation of the transformer neural network architecture, which we can use for training models from scratch, import pretrained models, use these models for inference / testing, and use them for fine-tuning

- Library supports many many models in PyTorch


### Tokenizers Library

- Point of tokenization is to encode strings which can be characters or words as integers that these language models can understand

- When we take the weights of open-source pretrained models online, we need to make sure we're using the same tokenizer for testing the model or fine-tuning it

- When we load in our new dataset of sentences or strings, the exact same tokenizer that the model uses / was originally trained on must be used, allowing the text to get broken up in a consistent way

- If we don't, a different mapping / encoding of these characters / words to integers would be used, causing the model to have meaningless token outputs or even get extremely confused when trying to further train / fine-tune

- This library lets us abstract away the tokenization process, we can just call simple functions like encode sentence and tokenize my sentence, also provides an extremely fast implementation (because it's implemented in Rust which is extremely fast, and it's a highly optimized implementation meant to take advantage of GPUs)

- We will use this library instead of manually doing the tokenization ourselves (like we did in the previous problems)

- Will also automatically use the right tokenizer for the model, we only have to specify a few things