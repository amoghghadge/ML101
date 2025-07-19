import numpy as np
from numpy.typing import NDArray

class Solution:
    def get_derivative(
        self,
        model_prediction: NDArray[np.float64],
        ground_truth: NDArray[np.float64],
        N: int,
        X: NDArray[np.float64],
        desired_weight: int
    ) -> float:
        # note that N is just len(X)
        return -2 * np.dot(ground_truth - model_prediction, X[:, desired_weight]) / N

    def get_model_prediction(
        self,
        X: NDArray[np.float64],
        weights: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        return np.squeeze(np.matmul(X, weights))

    learning_rate = 0.01

    def train_model(
        self, 
        X: NDArray[np.float64], 
        Y: NDArray[np.float64], 
        num_iterations: int, 
        initial_weights: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        # you will need to call get_derivative() for each weight
        # and update each one separately based on the learning rate!
        # return np.round(your_answer, 5)
        curr_weights = initial_weights
        N = len(X)
        M = len(initial_weights)

        for _ in range(num_iterations):
            pred = self.get_model_prediction(X, curr_weights)

            # find out partial derivate of error function with respect to each weight
            for i in range(M):
                curr_weights[i] -= self.learning_rate * self.get_derivative(pred, Y, N, X, i)

        return np.round(curr_weights, 5)
    
class Solution2:
    def get_derivative(
        self,
        model_prediction: NDArray[np.float64],
        ground_truth: NDArray[np.float64],
        N: int,
        X: NDArray[np.float64],
    ) -> float:
        # note that N is just len(X)
        return -2 * np.matmul(ground_truth - model_prediction, X) / N

    def get_model_prediction(
        self,
        X: NDArray[np.float64],
        weights: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        return np.squeeze(np.matmul(X, weights))

    learning_rate = 0.01

    def train_model(
        self, 
        X: NDArray[np.float64], 
        Y: NDArray[np.float64], 
        num_iterations: int, 
        initial_weights: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        # you will need to call get_derivative() for each weight
        # and update each one separately based on the learning rate!
        # return np.round(your_answer, 5)
        curr_weights = initial_weights
        N = len(X)
        M = len(initial_weights)

        for _ in range(num_iterations):
            pred = self.get_model_prediction(X, curr_weights)

            # find out partial derivate of error function with respect to each weight
            curr_weights -= self.learning_rate * self.get_derivative(pred, Y, N, X)

        return np.round(curr_weights, 5)
    
ans = Solution()

X = np.array([
    [1, 2, 3],
    [1, 1, 1],
])
Y = np.array([6, 3])
num_iterations = 10
initial_weights = np.array([0.2, 0.1,0.6])

print(ans.train_model(X, Y, num_iterations, initial_weights))