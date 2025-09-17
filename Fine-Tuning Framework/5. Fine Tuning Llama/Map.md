### Map function

- some_dataset.map(some_function)

- some_dataset is an actual dataset object that we download from the transformers library (maybe yelp reviews, amazon product reviews, etc.) that follows the attributes that hugging face / transformers library prescribes (has a training key, a testing key - like a dictionary)

- On this dataset we call the map function, which takes in some other function that we define in the code

- This is going to apply our some_function to every single element in our dataset

- If we're training on some sort of text-based dataset, this function could do something like pad all the sentences (so some senteces that are longer or shorter all have a consistent length before feeding them into the model)

- some_function will apply a variety of data processing techniques (one of the most relevant ones being pad, but other ones will be explained later)

- Again some_function is applied to every single entry / datapoint in the some_dataset

- Instead of manually calling some_function ourselves on every element in the dataset, we want to use map because it's going to take advantage of parallel computing and actually be a lot faster, making a difference when we're dealing with datasets consisting of hundreds of thousands of data points