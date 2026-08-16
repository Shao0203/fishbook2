import numpy as np
from common.util import preprocess, create_co_matrix


text = 'You say goodbye and I say hello.'
corpus, word_to_id, id_to_word = preprocess(text)
co_matrix = create_co_matrix(corpus, len(word_to_id))
print(co_matrix)
# [[0 1 0 0 0 0 0]
#  [1 0 1 0 1 1 0]
#  [0 1 0 1 0 0 0]
#  [0 0 1 0 1 0 0]
#  [0 1 0 1 0 0 0]
#  [0 1 0 0 0 0 1]
#  [0 0 0 0 0 1 0]]
