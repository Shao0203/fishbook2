from common.functions import *
import numpy as np
import collections


# ===== select multi rows from matrix
w = np.arange(21).reshape(7, 3)
idx = np.array([1, 0, 3, 0])
# print(w, '\n\n', w[idx])


# ===== Embedding EmbeddingDot SigmoidWithLoss
class Embedding:
    def __init__(self, W):
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.idx = None

    def forward(self, idx):
        W, = self.params
        self.idx = idx
        return W[idx]

    def backward(self, dout):
        dW, = self.grads
        dW[...] = 0
        np.add.at(dW, self.idx, dout)
        return None


class SigmoidWithLoss:
    def __init__(self):
        self.params, self.grads = [], []
        self.y = None
        self.t = None

    def forward(self, x, t):
        self.y = 1 / (1 + np.exp(-x))
        self.t = t
        loss = cross_entropy_error(np.c_[1-self.y, self.y], self.t)
        return loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) * dout / batch_size
        return dx


class EmbeddingDot:
    def __init__(self, W):
        self.embed = Embedding(W)
        self.params = self.embed.params
        self.grads = self.embed.grads
        self.cache = None

    def forward(self, h, idx):
        target_W = self.embed.forward(idx)
        self.cache = (h, target_W)
        out = np.sum(target_W * h, axis=1)
        return out

    def backward(self, dout):
        h, target_W = self.cache
        dout = dout.reshape(dout.shape[0], 1)
        dtarget_W = dout * h
        self.embed.backward(dtarget_W)
        dh = dout * target_W
        return dh
