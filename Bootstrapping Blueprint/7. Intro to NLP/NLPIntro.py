import torch
import torch.nn as nn
from torchtyping import TensorType
from typing import List

# This is the process of tokenization, arbitrarily encoding words into integers
# With these encoded tokens, the next step is to get an embedding vector that represents the actual semantic meaning of the word
# Embedding vectors are randomly initialized then trained and updated over many iterations to learn the relationship between words in a vectorized space

# torch.tensor(python_list) returns a Python list as a tensor
class Solution:
    def get_dataset(self, positive: List[str], negative: List[str]) -> TensorType[float]:
        strings, words = positive + negative, set()
        for s in strings:
            for w in s.split():
                words.add(w)
        ordered = list(words)
        ordered.sort()
        
        mapping = {}
        for w in ordered:
            mapping[w] = len(mapping) + 1
        print(mapping)
        
        tensors = []
        for s in strings:
            tensors.append(torch.tensor([mapping[w] for w in s.split()]))
        return torch.nn.utils.rnn.pad_sequence(tensors, batch_first=True)
    
    def better_get_dataset(self, positive: List[str], negative: List[str]) -> TensorType[float]:
        def getWords(s):
            ans, word = [], []
            for i in range(len(s)):
                if s[i].isalnum():
                    word.append(s[i])
                else:
                    if len(word):
                        ans.append("".join(word))
                    word = []
            return ans

        strings, words = positive + negative, set()
        for s in strings:
            for w in getWords(s):
                words.add(w)
        ordered = list(words)
        ordered.sort()
        
        mapping = {}
        for w in ordered:
            mapping[w] = len(mapping) + 1
        print(mapping)
        
        tensors = []
        for s in strings:
            tensors.append(torch.tensor([mapping[w] for w in getWords(s)]))
        return torch.nn.utils.rnn.pad_sequence(tensors, batch_first=True)