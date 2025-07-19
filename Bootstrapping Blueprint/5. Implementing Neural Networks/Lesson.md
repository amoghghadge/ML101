### Dropout

- Solves the problem of overfitting: when training performance/accuracy is greater than testing performance/accuracy (error goes down every iteration, but prediction on data it's never seen before is horrible) - caused when model is too complex (too many layers, too many nodes in each layer) and it memorizes irrelevant details/noise in the training data

- nn.Dropout(p) makes it so at every iteration of training, each node independently in the layer you apply it to get turned off (its output/activation is set to 0) with probability p

- When a node in a layer gets turned off, its connections from the previous layer are severed (the output/activate gets set to 0)

- It reduces the complexity of our model - with randomeness we delete some nodes in a given layer to make our model less complex and make it stupider

- Takes our giant NN and dropping it on the ground to knock some of its screws loose

- Decreases its ability to learn really intricate noise in the training data, makes it just focus on big picture instead of memorizing specific noise to make our testing accuracy will go up

- Increases perfomance of our NNs especially as they get deeper and deeper (more and more layers)