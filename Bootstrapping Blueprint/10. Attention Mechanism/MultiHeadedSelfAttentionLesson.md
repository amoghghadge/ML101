### Multi Headed Attention

- We can use multiple heads of self attention that operate in parallel with their own seperate weights being trained

- We can concatenate the result of each head together for the output of multi headed attention

- Taking the overall attention dim divided by the number of heads you want to use gives you the head size, or the individual attention dim for each head of self attention

- The total number of parameters across all heads stay the same because the number of parameters and attention dim for each head's key, query, and value layers shrinks

- This gives better results because each head of attention gets to operate on the input seperately, so it can learn its own different things/nuances/patterns

- Each head of attention gets specialized at learning some component of the entire langauge during training

- For example in BERT, one head of attention was specialized at looking at direct objects of verbs, and another head was specialized at looking at indirect objects of verbs

- These large language models have different heads / components of its neural network that specialize at learning different parts of a langauge / grammar