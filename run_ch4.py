from common.functions import *
import numpy as np
import collections
import pickle
from common.trainer import Trainer
from common.optimizer import Adam
from common.util import create_contexts_target, most_similar, analogy
from dataset import ptb


# ===== select multi rows from matrix
w = np.arange(21).reshape(7, 3)
idx = np.array([1, 0, 3, 0])
# print(w, '\n\n', w[idx])

# ===== Sampling
'''
words = ['you', 'say', 'goodbye', 'I', 'hello', '.']
p = [0.5, 0.1, 0.05, 0.2, 0.05, 0.1]
print(np.random.choice(10))
print(np.random.choice(words, size=5, replace=False))
print(np.random.choice(words, p=p))

p = [0.7, 0.29, 0.01]
new_p = np.power(p, 0.75)
new_p /= np.sum(new_p)
print(p, '\n', new_p)

corpus = np.array([0, 1, 2, 3, 4, 1, 2, 3])
sampler = UnigramSampler(corpus, 0.75, 2)
target = np.array([1, 3, 0])
negative_sample = sampler.get_negative_sample(target)
# print(negative_sample) # [[2 4] [0 2] [1 3]] # 3 row 2 col = 3 target 2 sample
'''


# ===== Embedding EmbeddingDot SigmoidWithLoss UnigramSampler NegativeSamplingLoss CBOW SkipGram
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


class UnigramSampler:
    def __init__(self, corpus, power, sample_size):
        self.sample_size = sample_size
        self.vocab_size = None
        self.word_p = None

        # counts = {i: list(corpus).count(i) for i in set(corpus)}
        counts = collections.Counter(corpus)
        vocab_size = len(counts)
        self.vocab_size = vocab_size

        self.word_p = np.zeros(vocab_size)
        for i in range(vocab_size):
            self.word_p[i] = counts[i]
        self.word_p = np.power(self.word_p, power)
        self.word_p /= np.sum(self.word_p)

    def get_negative_sample(self, target):
        batch_size = target.shape[0]
        negative_sample = np.zeros((batch_size, self.sample_size), dtype=np.int32)
        for i in range(batch_size):
            p = self.word_p.copy()
            target_idx = target[i]
            p[target_idx] = 0
            p /= p.sum()
            negative_sample[i, :] = np.random.choice(self.vocab_size, self.sample_size, False, p)
        return negative_sample


class NegativeSamplingLoss:
    def __init__(self, W, corpus, power=0.75, sample_size=5):
        self.sample_size = sample_size
        self.sampler = UnigramSampler(corpus, power, sample_size)
        self.loss_layers = [SigmoidWithLoss() for _ in range(sample_size+1)]
        self.embed_dot_layers = [EmbeddingDot(W) for _ in range(sample_size+1)]

        self.params, self.grads = [], []
        for layer in self.embed_dot_layers:
            self.params += layer.params
            self.grads += layer.grads

    def forward(self, h, target):
        negative_sample = self.sampler.get_negative_sample(target)
        batch_size = target.shape[0]
        correct_label = np.ones(batch_size, dtype=np.int32)
        negative_label = np.zeros(batch_size, dtype=np.int32)

        # correct forward pass
        score = self.embed_dot_layers[0].forward(h, target)
        loss = self.loss_layers[0].forward(score, correct_label)

        # negative forward pass
        for i in range(self.sample_size):
            negative_target = negative_sample[:, i]
            score = self.embed_dot_layers[1+i].forward(h, negative_target)
            loss += self.loss_layers[1+i].forward(score, negative_label)

        return loss

    def backward(self, dout=1):
        dh = 0
        for l0, l1 in zip(self.loss_layers, self.embed_dot_layers):
            dscore = l0.backward(dout)
            dh += l1.backward(dscore)
        return dh


class CBOW:
    def __init__(self, vocab_size, hidden_size, window_size, corpus):
        V, H = vocab_size, hidden_size
        W_in = 0.01 * np.random.randn(V, H).astype('f')
        W_out = 0.01 * np.random.randn(V, H).astype('f')

        # layers
        self.in_layers = [Embedding(W_in) for _ in range(2*window_size)]
        self.ns_loss = NegativeSamplingLoss(W_out, corpus, power=0.75, sample_size=5)

        # params, grads, word_vecs
        self.params, self.grads = [], []
        layers = self.in_layers + [self.ns_loss]
        for layer in layers:
            self.params += layer.params
            self.grads += layer.grads

        self.word_vecs = W_in

    def forward(self, contexts, target):
        h = 0
        for i, layer in enumerate(self.in_layers):
            h += layer.forward(contexts[:, i])
        h *= 1 / len(self.in_layers)
        loss = self.ns_loss.forward(h, target)
        return loss

    def backward(self, dout=1):
        dout = self.ns_loss.backward(dout)
        dout *= 1 / len(self.in_layers)
        for layer in self.in_layers:
            layer.backward(dout)
        return None


class SkipGram:
    def __init__(self, vocab_size, hidden_size, window_size, corpus):
        V, H = vocab_size, hidden_size
        W_in = 0.01 * np.random.randn(V, H).astype('f')
        W_out = 0.01 * np.random.randn(V, H).astype('f')

        # layers
        self.in_layer = Embedding(W_in)
        self.loss_layers = [NegativeSamplingLoss(W_out, corpus) for _ in range(2*window_size)]

        # params, grads, word_vecs
        self.params, self.grads = [], []
        layers = [self.in_layer] + self.loss_layers
        for layer in layers:
            self.params += layer.params
            self.grads += layer.grads

        self.word_vecs = W_in

    def forward(self, contexts, target):
        h = self.in_layer.forward(target)
        loss = 0
        for i, layer in enumerate(self.loss_layers):
            loss += layer.forward(h, contexts[:, i])
        return loss

    def backward(self, dout=1):
        dh = 0
        for layer in self.loss_layers:
            dh += layer.backward(dout)
        self.in_layer.backward(dh)
        return None


# ===== CBOW training
'''
# hyper-params
window_size = 5
hidden_size = 100
batch_size = 100
max_epoch = 10
# load data
corpus, word_to_id, id_to_word = ptb.load_data('train')  # len: (929589, 10000, 10000)
vocab_size = len(word_to_id)  # 10000
contexts, target = create_contexts_target(corpus, window_size)  # len: (929579, 929579)
# create model
model = SkipGram(vocab_size, hidden_size, window_size, corpus)
model = CBOW(vocab_size, hidden_size, window_size, corpus)
optimizer = Adam()
trainer = Trainer(model, optimizer)
# start training
trainer.fit(contexts, target, max_epoch, batch_size)
trainer.plot()
# save data
params = {
    'word_vecs': model.word_vecs.astype(np.float16),
    'word_to_id': word_to_id,
    'id_to_word': id_to_word,
}
pkl_file = 'cbow_params.pkl'  # or skipgram_params.pkl
with open(pkl_file, 'wb') as f:
    pickle.dump(params, f, -1)
'''

# ===== Evaluate the model
pkl_file = 'cbow_params.pkl'  # or 'skipgram_params.pkl'
with open(pkl_file, 'rb') as f:
    params = pickle.load(f)
    word_vecs, word_to_id, id_to_word = params.values()
# print(word_vecs.shape)  # (10000, 100)
# print(word_vecs[0][:10])  # 第 0 个词的前 10 维
querys = ['you', 'year', 'car', 'toyota']
# for query in querys:
#     most_similar(query, word_to_id, id_to_word, word_vecs)
'''[query] you
i: 0.72802734375
we: 0.70751953125
your: 0.6318359375
they: 0.62451171875
anything: 0.58154296875

[query] year
month: 0.85693359375
week: 0.7763671875
summer: 0.75927734375
spring: 0.732421875
decade: 0.65185546875

[query] car
truck: 0.64111328125
window: 0.59716796875
auto: 0.5849609375
luxury: 0.5751953125
merkur: 0.5400390625

[query] toyota
honda: 0.669921875
minicomputers: 0.642578125
nissan: 0.63818359375
seita: 0.63232421875
f-14: 0.6240234375 '''

# analogy task
analogy('king', 'man', 'queen',  word_to_id, id_to_word, word_vecs)
analogy('take', 'took', 'go',  word_to_id, id_to_word, word_vecs)
analogy('car', 'cars', 'child',  word_to_id, id_to_word, word_vecs)
analogy('good', 'better', 'bad',  word_to_id, id_to_word, word_vecs)
