import numpy as np


a = np.random.randn(3).astype('f')
print(a.dtype)
b = np.random.randn(3).astype(np.float32)
print(b.dtype)
c = np.random.randn(3).astype(np.float16)
print(c.dtype)
