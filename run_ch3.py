import numpy as np
from common.layers import MatMul, SoftmaxWithLoss
from common.trainer import Trainer
from common.optimizer import Adam
from common.util import preprocess, create_contexts_target, convert_one_hot


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
# print(np.array_equal(h1, h2) and np.array_equal(h2, W_idx))
# print(np.all(h1 == h2) and np.all(h2 == W_idx))


# ===== implement CBOW forward
c0 = np.array([[1, 0, 0, 0, 0, 0, 0]])  # (1, 7)
c1 = np.array([[0, 0, 1, 0, 0, 0, 0]])
W_in = np.random.randn(7, 3)
W_out = np.random.randn(3, 7)

in_layer0 = MatMul(W_in)
in_layer1 = MatMul(W_in)
out_layer = MatMul(W_out)
loss_layer = SoftmaxWithLoss()

h0 = in_layer0.forward(c0)
h1 = in_layer1.forward(c1)
h = 0.5 * (h0 + h1)  # (1, 3)
s = out_layer.forward(h)  # (1, 7)
# print(s) # [[-0.14142793  0.01381959 -0.172009    0.05532062  0.24662128 -0.15750454 -0.22891428]]


# ===== CBOW training prepare contexts + target + one_hot
def create_contexts_target(corpus, window_size=1):
    target = corpus[window_size:-window_size]
    contexts = []
    for idx in range(window_size, len(corpus)-window_size):
        cs = []
        for t in range(-window_size, window_size + 1):
            if t == 0:
                continue
            cs.append(corpus[idx + t])
        contexts.append(cs)
    return np.array(contexts), np.array(target)


def convert_one_hot(corpus, vocab_size):
    N = corpus.shape[0]
    if corpus.ndim == 1:
        one_hot = np.zeros((N, vocab_size), dtype=np.int32)
        for idx, word_id in enumerate(corpus):
            one_hot[idx, word_id] = 1
    elif corpus.ndim == 2:
        C = corpus.shape[1]
        one_hot = np.zeros((N, C, vocab_size), dtype=np.int32)
        for idx_0, word_ids in enumerate(corpus):
            for idx_1, word_id in enumerate(word_ids):
                one_hot[idx_0, idx_1, word_id] = 1
    return one_hot


# ===== SimpleCBOW 上下文 → 目标词
class SimpleCBOW:
    def __init__(self, vocab_size, hidden_size):
        V, H = vocab_size, hidden_size
        W_in = 0.01 * np.random.randn(V, H).astype('f')
        W_out = 0.01 * np.random.randn(H, V).astype('f')

        # layers
        self.in_layer0 = MatMul(W_in)
        self.in_layer1 = MatMul(W_in)
        self.out_layer = MatMul(W_out)
        self.loss_layer = SoftmaxWithLoss()

        # params, grads, word_vecs
        self.params, self.grads = [], []
        layers = [self.in_layer0, self.in_layer1, self.out_layer]
        for layer in layers:
            self.params += layer.params
            self.grads += layer.grads
        self.word_vecs = W_in

    def forward(self, contexts, target):
        h0 = self.in_layer0.forward(contexts[:, 0])
        h1 = self.in_layer1.forward(contexts[:, 1])
        h = 0.5 * (h0 + h1)
        score = self.out_layer.forward(h)
        loss = self.loss_layer.forward(score, target)
        return loss

    def backward(self, dout=1):
        ds = self.loss_layer.backward(dout)
        da = self.out_layer.backward(ds)
        da *= 0.5
        self.in_layer1.backward(da)
        self.in_layer0.backward(da)
        return None


# ===== SimpleCBOW training
text = 'You say goodbye and I say hello.'
corpus, word_to_id, id_to_word = preprocess(text)
vocab_size = len(word_to_id)
contexts, target = create_contexts_target(corpus)  # (6, 2) (6,)
contexts_one_hot = convert_one_hot(contexts, vocab_size)  # (6,2,7)
target_one_hot = convert_one_hot(target, vocab_size)  # (6, 7)

'''contexts / target 
[[0 2] [1 3] [2 4] [3 1] [4 5] [1 6]]
[1 2 3 4 1 5]

contexts_one_hot 
[[[1 0 0 0 0 0 0]  [0 0 1 0 0 0 0]]
 [[0 1 0 0 0 0 0]  [0 0 0 1 0 0 0]]
 [[0 0 1 0 0 0 0]  [0 0 0 0 1 0 0]]
 [[0 0 0 1 0 0 0]  [0 1 0 0 0 0 0]]
 [[0 0 0 0 1 0 0]  [0 0 0 0 0 1 0]]
 [[0 1 0 0 0 0 0]  [0 0 0 0 0 0 1]]]

 target_one_hot
 [[0 1 0 0 0 0 0]
 [0 0 1 0 0 0 0]
 [0 0 0 1 0 0 0]
 [0 0 0 0 1 0 0]
 [0 1 0 0 0 0 0]
 [0 0 0 0 0 1 0]]
'''

model = SimpleCBOW(vocab_size, hidden_size=5)
optimizer = Adam()
trainer = Trainer(model, optimizer)
trainer.fit(contexts_one_hot, target_one_hot, max_epoch=1000, batch_size=3)
trainer.plot()

word_vecs = model.word_vecs  # 密集向量 | 分布式表示 | W_in
for word_id, word in id_to_word.items():
    print(word, word_vecs[word_id])
'''
you [ 0.9209978 -0.9494278  1.1313007 -1.6351631  0.8810054]
say [-1.143435    1.1416619  -0.34699273 -1.3318226  -1.1345822 ]
goodbye [ 1.0880073  -1.050468    0.64392406 -0.06327208  1.1197919 ]
and [-0.75663835  0.7442101  -1.8513232  -1.283631   -0.7002805 ]
i [ 1.1202984  -1.0565563   0.635471   -0.04566751  1.1259419 ]
hello [ 0.9278471 -0.946142   1.1327729 -1.6373544  0.9061884]
. [-1.1775548   1.1744554   1.7243804  -0.85353583 -1.1529981 ]
'''

# ===== SimpleSkipGram 目标词 → 上下文


class SimpleSkipGram:
    def __init__(self, vocab_size, hidden_size):
        V, H = vocab_size, hidden_size
        W_in = 0.01 * np.random.randn(V, H).astype('f')
        W_out = 0.01 * np.random.randn(H, V).astype('f')

        self.in_layer = MatMul(W_in)
        self.out_layer = MatMul(W_out)
        self.loss_layer1 = SoftmaxWithLoss()
        self.loss_layer2 = SoftmaxWithLoss()

        self.params, self.grads = [], []
        layers = [self.in_layer, self.out_layer]
        for layer in layers:
            self.params += layer.params
            self.grads += layer.grads
        self.word_vecs = W_in

    def forward(self, contexts, target):
        h = self.in_layer.forward(target)
        s = self.out_layer.forward(h)
        l1 = self.loss_layer1.forward(s, contexts[:, 0])
        l2 = self.loss_layer2.forward(s, contexts[:, 1])
        loss = l1 + l2
        return loss

    def backward(self, dout=1):
        dl2 = self.loss_layer2.backward(dout)
        dl1 = self.loss_layer1.backward(dout)
        ds = dl1 + dl2
        dh = self.out_layer.backward(ds)
        self.in_layer.backward(dh)
        return None


# ===== SimpleSkipGram training
skip_model = SimpleSkipGram(vocab_size, 5)
skip_trainer = Trainer(skip_model, optimizer)
skip_trainer.fit(contexts_one_hot, target_one_hot, max_epoch=1000, batch_size=3)
skip_trainer.plot()

skip_word_vecs = skip_model.word_vecs
for word_id, word in id_to_word.items():
    print(word, np.round(word_vecs[word_id], 5))
'''
you [ 0.00145 -0.01262 -0.00888  0.00992 -0.00607]
say [-0.01538 -0.00059 -0.00509 -0.00874 -0.00124]
goodbye [-0.0011  -0.00306 -0.0148   0.0123   0.00667]
and [-0.00161 -0.01841  0.00981 -0.00421 -0.01631]
i [-0.00802 -0.00991 -0.00044 -0.00415 -0.01768]
hello [ 0.00876 -0.00167 -0.00086 -0.00894 -0.00369]
. [ 0.00227 -0.00751  0.00614  0.00042  0.00083]
'''
