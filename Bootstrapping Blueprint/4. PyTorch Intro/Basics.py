import torch
import torch.nn
from torchtyping import TensorType

# Helpful functions:
# https://pytorch.org/docs/stable/generated/torch.reshape.html
# https://pytorch.org/docs/stable/generated/torch.mean.html
# https://pytorch.org/docs/stable/generated/torch.cat.html
# https://pytorch.org/docs/stable/generated/torch.nn.functional.mse_loss.html

# Round your answers to 4 decimal places using torch.round(input_tensor, decimals = 4)
class Solution:
    def reshape(self, to_reshape: TensorType[float]) -> TensorType[float]:
        # torch.reshape() will be useful - check out the documentation
        return torch.round(torch.reshape(to_reshape, (to_reshape.shape[0] * to_reshape.shape[1] // 2, 2)), decimals=4)
        # or return torch.round(torch.reshape(to_reshape, (-1, 2)), decimals=4)
        # -1 means to automatically figure out number of rows needed to preserve all data given we only want 2 cols

    def average(self, to_avg: TensorType[float]) -> TensorType[float]:
        # torch.mean() will be useful - check out the documentation
        return torch.round(torch.mean(to_avg, dim=0), decimals=4)

    def concatenate(self, cat_one: TensorType[float], cat_two: TensorType[float]) -> TensorType[float]:
        # torch.cat() will be useful - check out the documentation
        return torch.round(torch.cat((cat_one, cat_two), 1), decimals=4)

    def get_loss(self, prediction: TensorType[float], target: TensorType[float]) -> TensorType[float]:
        # torch.nn.functional.mse_loss() will be useful - check out the documentation
        return torch.round(torch.nn.functional.mse_loss(prediction, target), decimals=4)

# We use these functions because they take advantage of parallel processing and can operate on our input simultaneously

ans = Solution()

# 3 x 4 tensor
to_reshape = torch.tensor([
  [1.0, 1.0, 1.0, 1.0],
  [1.0, 1.0, 1.0, 1.0],
  [1.0, 1.0, 1.0, 1.0]
])
print(ans.reshape(to_reshape))
# 6 x 2 tensor
# tensor([[1., 1.], [1., 1.], [1., 1.], [1., 1.], [1., 1.], [1., 1.]])

# 2 x 3 tensor
to_avg = torch.tensor([
  [0.8088, 1.2614, -1.4371],
  [-0.0056, -0.2050, -0.7201]
])
# 1 x 3 tensor
print(ans.average(to_avg))
# tensor([0.4016, 0.5282, -1.0786])

# 2 x 3 tensor
cat_one = torch.tensor([
  [1.0, 1.0, 1.0],
  [1.0, 1.0, 1.0]
])
# 2 x 2 tensor
cat_two = torch.tensor([
  [1.0, 1.0],
  [1.0, 1.0]
])
print(ans.concatenate(cat_one, cat_two))
# 2 x 5 tensor
# tensor([[1., 1., 1., 1., 1.], [1., 1., 1., 1., 1.]])

prediction = torch.tensor([0.0, 1.0, 0.0, 1.0, 1.0])
target = torch.tensor([1.0, 1.0, 0.0, 0.0, 0.0])
print(ans.get_loss(prediction, target))
# tensor(0.6000) - can use .item() to get just the value