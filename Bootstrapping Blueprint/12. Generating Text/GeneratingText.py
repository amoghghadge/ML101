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
            print(batch)
        # Once your code passes the test, check out the Colab link and hit Run to see your code generate new Drake lyrics!
        # Your code's output, ran in this sandbox will be boring because of the computational limits in this sandbox
