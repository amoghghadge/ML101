import numpy as np
from numpy.typing import NDArray

# Helpful functions:
# https://numpy.org/doc/stable/reference/generated/numpy.matmul.html
# https://numpy.org/doc/stable/reference/generated/numpy.mean.html
# https://numpy.org/doc/stable/reference/generated/numpy.square.html

class Solution:
    def get_model_prediction(
        self,
        X: NDArray[np.float64],
        weights: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        # X is an Nx3 NumPy array
        # weights is a 3x1 NumPy array
        # HINT: np.matmul() will be useful
        # return np.round(your_answer, 5)
        ans = np.matmul(X, weights)
        return np.round(ans, 5)

    def get_error(
        self,
        model_prediction: NDArray[np.float64],
        ground_truth: NDArray[np.float64]
    ) -> float:
        # model_prediction is an Nx1 NumPy array
        # ground_truth is an Nx1 NumPy array
        # HINT: np.mean(), np.square() will be useful
        # return round(your_answer, 5)
        diff = model_prediction - ground_truth
        squared = np.square(diff)
        avg = np.mean(squared)
        return round(avg, 5)
    
ans = Solution()

# 2 x 3 matrix
X = np.array([
    [0.3745401188473625, 0.9507143064099162, 0.7319939418114051],
    [0.5986584841970366, 0.156018640442436, 0.15599452033620265],
])
# 3 x 1 matrix
weights = np.array([
    [1.0],
    [2.0],
    [3.0],
])
# 2 x 1 matrix
print(ans.get_model_prediction(X, weights))
# [[4.47195], [1.37868]]

# 3 x 1 matrix
predictions = np.array([
    [0.37454012],
    [0.95071431],
    [0.73199394]
])
# 3 x 1 matrix
labels = np.array([
    [0.59865848],
    [0.15601864],
    [0.15599452]
])
# Float value
print(ans.get_error(predictions, labels))
# 0.33785