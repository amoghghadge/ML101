import torch
import torch.nn as nn
from torchtyping import TensorType

class Solution:
    def generate(self, model, new_chars: int, context: TensorType[int], context_length: int, int_to_char: dict) -> str:
        # 1. Use torch.multinomial() to choose the next token.
        #    This function simulates a weighted draw from a given list of probabilities
        #    It's similar to picking marbles out of a bag.
        # 2. the given model's output is BEFORE softmax is applied,
        #    and the forward() output has shape batch X time X vocab_size
        # 3. Do not alter the code below, only add to it. This is for maintaining reproducibility in testing.

        generator = torch.manual_seed(0)
        initial_state = generator.get_state()
        for i in range(new_chars):
            model_input = context[:, -context_length:]
            # ensures input is B x T
            model_output = model(model_input)
            # output is B x T x V
            final_token = model_output[:, -1, :]
            # final_token is now B x V, extracted last vector V for each sequence T across all B examples
            token_probs = nn.functional.softmax(final_token, dim=1)
            # softmax applied so numbers represent probabilities, still B x V
            # The line where you call torch.multinomial(). Pass in the generator as well.
            generator.set_state(initial_state)
            chosen_tokens = torch.multinomial(token_probs, 1, generator=generator)
            # chosen_tokens is B x 1
            context = torch.cat((context, chosen_tokens), 1)
            # context now has new generated token for all B examples

        ans = []
        for batch in context:
            tokens = batch.tolist()
            # could skip first token here to exclude start token from overall generated text
            chars = [int_to_char[c] for c in tokens]
            ans.append("".join(chars))
        print(ans)
        # just returning first example for sake of coding problem
        return ans[0][1:]

        # Once your code passes the test, check out the Colab link and hit Run to see your code generate new Drake lyrics!
        # Your code's output, ran in this sandbox will be boring because of the computational limits in this sandbox

# Input:
# model=GPT(104, 128, 252, 6, 6)
# new_chars=1
# context=torch.zeros(1, 1, dtype=int)
# context_length=128
# int_to_char={0: '\n', 1: ' ', 2: '!', 3: '"', 5: '%', 6: '&', 7: "'", 8: '(', 9: ')', 10: '*', 11: '+', 12: ',', 13: '-', 14: '.', 15: '/', 16: '0', 17: '1', 18: '2', 19: '3', 20: '4', 21: '5', 22: '6', 23: '7', 24: '8', 25: '9', 26: ':', 27: ';', 28: '?', 29: 'A', 30: 'B', 31: 'C', 32: 'D', 33: 'E', 34: 'F', 35: 'G', 36: 'H', 37: 'I', 38: 'J', 39: 'K', 40: 'L', 41: 'M', 42: 'N', 43: 'O', 44: 'P', 45: 'Q', 46: 'R', 47: 'S', 48: 'T', 49: 'U', 50: 'V', 51: 'W', 52: 'X', 53: 'Y', 54: 'Z', 55: '[', 56: ']', 57: '_', 58: 'a', 59: 'b', 60: 'c', 61: 'd', 62: 'e', 63: 'f', 64: 'g', 65: 'h', 66: 'i', 67: 'j', 68: 'k', 69: 'l', 70: 'm', 71: 'n', 72: 'o', 73: 'p', 74: 'q', 75: 'r', 76: 's', 77: 't', 78: 'u', 79: 'v', 80: 'w', 81: 'x', 82: 'y', 83: 'z', 84: '{', 85: '|', 86: '}', 87: 'à', 88: 'á', 89: 'è', 90: 'é', 91: 'ë', 92: 'ñ', 93: 'ó', 94: 'ú', 95: '\u2005', 96: '–', 97: '—', 98: '‘', 99: '’', 100: '“', 101: '”', 102: '…', 103: '\u205f'}

# stdout:
# ['\nè']

# Output:
# è
