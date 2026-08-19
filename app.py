import numpy as np
from common.layers import MatMul


c = np.array([[0, 0, 1, 0, 0, 0, 0]])
W = np.random.randn(7, 3)
h1 = np.dot(c, W)
layer = MatMul(W)
h2 = layer.forward(c)

idx = np.argmax(c, axis=1)  # 2
W_idx = W[idx]

# print(h1)  # [[ 0.04888457 -0.98980983  0.2845656 ]]
# print(h2)  # [[ 0.04888457 -0.98980983  0.2845656 ]]
# print(h1 == h2)     # [[ True  True  True]]
# print(h1 == W_idx)  # [[ True  True  True]]
print(np.array_equal(h1, h2) and np.array_equal(h2, W_idx))
print(np.all(h1 == h2) and np.all(h2 == W_idx))
