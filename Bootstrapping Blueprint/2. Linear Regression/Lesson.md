### Linear Regression

- Foundation of neural networks

- Regression: not a fixed number of classes, prediction could be any number on an infinite scale

- Linear: The relationship between data points and actual answer is linear
    - h(x, y, z) = w1 * x + w2 * y + w3 * z + b
    - Training process adjusts w1, w2, w3, and b

### Psuedocode

```
for num_iterations:
    get_model_pred()
    get_error()
    get_derivatives()
    update_weights()
```

we want get_error() to get very close to 0

Error function: MSE (Mean Squared Error)
- sum over all points(prediction_i - truth_i)^2 / num of points
- average makes it independent on number of example
- squaring gets rid of negative differences
- absolute value's derivatives have issues and are weirder to computer

Uses matrix multiplication to produce each persons prediction

Say you have 3 features:
- Would do N x 3 matrix of observations (n data points, each with 3 features) times 3 x 1 matrix (weights for the model) produce N x 1 vector (N predictions from the model)