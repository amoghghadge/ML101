import torch
from typing import List, Tuple

class Solution:
    def batch_loader(self, raw_dataset: str, context_length: int, batch_size: int) -> Tuple[List[List[str]]]:
        # You must start by generating batch_size different random indices in the appropriate range
        # using a single call to torch.randint()
        torch.manual_seed(0)
        tokens = raw_dataset.split()
        idxs = torch.randint(0, len(tokens) - context_length, (batch_size,))    # the upper value for randint is exclusive, correctly keeps all y's words in range
        X = [[tokens[i] for i in range(idx.item(), idx.item() + context_length)] for idx in idxs]
        Y = [[tokens[i] for i in range(idx.item() + 1, idx.item() + 1 + context_length)] for idx in idxs]
        return (X, Y)
        # requires len(tokens) > context_length, as for any index you need to be able to get context_length tokens after it
        # X and Y both have dimension batch_size x context_length (B x T)

ans = Solution()

raw_dataset='Once upon a time on a GPU far far away there was an algorithm'
context_length=3
batch_size=2

print(ans.batch_loader(raw_dataset, context_length, batch_size))
# X and Y both are 2 x 3
# ([['there', 'was', 'an'], ['far', 'far', 'away']], [['was', 'an', 'algorithm'], ['far', 'away', 'there']])