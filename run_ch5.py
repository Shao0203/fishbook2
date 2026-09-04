from common.time_layers import *
from common.optimizer import SGD
from common.trainer import RnnlmTrainer
from dataset import ptb
import matplotlib.pyplot as plt


class SimpleRnnlm:
    def __init__(self, vocab_size, wordvec_size, hidden_size):
        V, D, H = vocab_size, wordvec_size, hidden_size
        rn = np.random.randn

        embed_W = (rn(V, D) / 100).astype('f')          # (10000, 100)
        rnn_Wx = (rn(D, H) / np.sqrt(D)).astype('f')    # (100, 128)
        rnn_Wh = (rn(H, H) / np.sqrt(H)).astype('f')    # (128, 128)
        rnn_b = np.zeros(H).astype('f')                 # (128,)
        affine_W = (rn(H, V) / np.sqrt(H)).astype('f')  # (128, 10000)
        affine_b = np.zeros(V).astype('f')              # (10000,)

        self.layers = [
            TimeEmbedding(embed_W),
            TimeRNN(rnn_Wx, rnn_Wh, rnn_b, stateful=True),
            TimeAffine(affine_W, affine_b)
        ]
        self.loss_layer = TimeSoftmaxWithLoss()
        self.rnn_layer = self.layers[1]

        self.params, self.grads = [], []
        for layer in self.layers:
            self.params += layer.params
            self.grads += layer.grads

    def forward(self, xs, ts):
        for layer in self.layers:
            xs = layer.forward(xs)
        loss = self.loss_layer.forward(xs, ts)
        return loss

    def backward(self, dout=1):
        dout = self.loss_layer.backward(dout)
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout

    def reset_state(self):
        self.rnn_layer.reset_state()


def train_rnn():
    # ===== hyper-params
    batch_size = 10
    wordvec_size = 100
    hidden_size = 100
    time_size = 5
    lr = 0.1
    max_epoch = 100
    # ===== load data to use the first 1000 items of it
    corpus, word_to_id, id_to_word = ptb.load_data('train')
    corpus = corpus[:1000]
    vocab_size = int(max(corpus) + 1)
    xs = corpus[:-1]
    ts = corpus[1:]
    data_size = len(xs)  # 999
    # ===== training params
    iter_per_epoch = data_size // (batch_size * time_size)  # 19
    time_idx, total_loss, loss_count = 0, 0, 0
    ppl_list = []
    optimizer = SGD(lr)
    model = SimpleRnnlm(vocab_size, wordvec_size, hidden_size)
    # ===== calculate start index for each mini_batch
    jump = data_size // batch_size
    offsets = [i * jump for i in range(batch_size)]
    # ===== training process
    for epoch in range(max_epoch):
        for iter in range(iter_per_epoch):
            # get mini_batch
            batch_x = np.empty((batch_size, time_size), dtype='i')
            batch_t = np.empty((batch_size, time_size), dtype='i')
            for col in range(time_size):
                for row, offset in enumerate(offsets):
                    batch_x[row, col] = xs[(offset + time_idx) % data_size]
                    batch_t[row, col] = ts[(offset + time_idx) % data_size]
                time_idx += 1
            # calc gradients and update params
            loss = model.forward(batch_x, batch_t)
            model.backward()
            optimizer.update(model.params, model.grads)
            total_loss += loss
            loss_count += 1
        # evaluate perplexity for each epoch
        ppl = np.exp(total_loss / loss_count)
        ppl_list.append(ppl)
        total_loss, loss_count = 0, 0
        print(f'epoch: {epoch+1} | perplexity: {ppl:.4f}')
    # ===== plot the result
    x = np.arange(len(ppl_list))
    plt.plot(x, ppl_list, label='train')
    plt.xlabel('epochs')
    plt.ylabel('perplexity')
    plt.show()


def use_trainer():
    wordvec_size, hidden_size = 100, 100
    max_epoch = 100
    time_size = 5
    batch_size = 10

    corpus, _, _ = ptb.load_data()
    corpus = corpus[:1000]
    vocab_size = int(max(corpus) + 1)
    xs = corpus[:-1]
    ts = corpus[1:]

    model = SimpleRnnlm(vocab_size, wordvec_size, hidden_size)
    optimizer = SGD(lr=0.1)
    trainer = RnnlmTrainer(model, optimizer)
    trainer.fit(xs, ts, max_epoch, batch_size, time_size)
    trainer.plot()


use_trainer()
