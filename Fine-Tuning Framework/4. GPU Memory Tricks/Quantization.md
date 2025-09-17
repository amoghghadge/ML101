### Quantization

- Another state of the art strategy to fine-tune LLMs on straightforward hardware

- We are going to run into memory issues

- In an LLM with billions of parameters / weights, if each weight is a 32 bit floating number, 32 bits * 100 billion results in a crazy amount of data and memory. Even in GPT-3 the weights are over 1 TB

- To save on memory, we can use less precision. Instead of using floating point 32 bit, what if we used 16 bit or 8 bit or 4 bit?

- The way floating point numbers are stored is 3 components: the mantissa, the exponent, and always 1 bit for the sign

- Say we have a number like 1.2 x 10^3. 1.2 is the mantissa and 3 is the exponent

- In 8 bit or 4 bit, there are way less bits for the mantissa and exponent, making the model a lot less precise in its internal calculations used to predict the next token

- There will be a slight decline in performance, but researchers found if we can use more parameters with less precision, we can still get pretty good results (80-90% of the original performance without needing super expensive GPUs)


### Tradeoff between model size and bit size

- A model with 1 billion parameters but 16 bit quantization has the same amount of memory as a model with 4 billion parameters and 4 bit quantization. The second model is a lot less precise even though it's a lot larger of a neural network. Researches have found its significant better virtually all the time to have the model with more parameters but just accept less precision. 

- We're going to use 4 bit quantization on a 7 billion parameter model (Llama 2 from Meta)

### How does this work in code

- We will leverage a technique called qLoRA, which integrates the quantization techniques with LoRA

- It's LoRA so we only focus on training a couple layers of the network (primarily attention layers) and we decompose that weight matrix into 2 matrices based on r (B and A)

- We can use the Bits and Bytes library from the hugging face community to do the heavy lifting of fine tuning models for us

- We just need to define a BitsandBytesConfig object and pass it into our model when we instantiate it

- The only things we tell it are to use 4 bit quantization, the actual bnb (bits and bytes) 4 bit quantization type (a new data type created in the paper called a normalized floating point type, still 4 bits but changed the way the data is represented at the binary level), along with the bnb 4 bit compute type (the magic behind qLoRA and why it works so well)

- The compute type should ideally use something much larger than 4 bits (like 8, 16, or even 32 bits). This means when we store the weights / parameters of the model, we just use the 4 bit quantization type because of our memory restriction. However when we actually do the forward passes through the network during fine-tuning (getting the model predictions) as well as the backward passes (only calculating gradients / doing backward pass for a couple layers due to LoRA) we dequantize / decompress from 4 bit to some higher dimensional data type to give us a lot more precision in these calculations that are then used to update the parameters inside our rank matrices (B and A) and ultimately fine-tune the model to improve the performance of the network. Once we're done with these calculations, the data is actually requantized / compressed back into the 4 bit state

- As a result the overall memory usage during training as well as during inference when using these fine-tuned saved model weights (which are now each saved as 4 bits) is low, and we still retain a solid amount of performance