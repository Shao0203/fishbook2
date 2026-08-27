# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
from dataset.spiral import load_data
from common.trainer import Trainer
from common.optimizer import SGD
from common.layers import Affine, Sigmoid, SoftmaxWithLoss


class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size):
        I, H, O = input_size, hidden_size, output_size
        W1 = 0.01 * np.random.randn(I, H)
        b1 = np.zeros(H)
        W2 = 0.01 * np.random.randn(H, O)
        b2 = np.zeros(O)

        self.layers = [Affine(W1, b1), Sigmoid(), Affine(W2, b2)]
        self.loss_layer = SoftmaxWithLoss()

        self.params, self.grads = [], []
        for layer in self.layers:
            self.params += layer.params
            self.grads += layer.grads

    def predict(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def forward(self, x, t):
        score = self.predict(x)
        loss = self.loss_layer.forward(score, t)
        return loss

    def backward(self, dout=1):
        dout = self.loss_layer.backward(dout)
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout


# ===== use trainer
x, t = load_data()
model = TwoLayerNet(2, 10, 3)
optimizer = SGD(1.0)
trainer = Trainer(model, optimizer)
trainer.fit(x, t, 300, 30, eval_interval=10)
trainer.plot()


# ===== spiral show
x, t = load_data()
print('x', x.shape)
print('t', t.shape)
# 绘制数据点
N = 100
CLS_NUM = 3
markers = ['o', 'x', '^']
for i in range(CLS_NUM):
    plt.scatter(x[i*N:(i+1)*N, 0], x[i*N:(i+1)*N, 1], s=40, marker=markers[i])
plt.show()


# ===== spiral training
x, t = load_data()
model = TwoLayerNet(2, 10, 3)
optimizer = SGD(lr=1)
max_epoch = 300
data_size = x.shape[0]
batch_size = 30
iter_per_epoch = data_size // batch_size
total_loss, loss_count, loss_list = 0, 0, []

for epoch in range(max_epoch):
    idx = np.random.permutation(data_size)
    x = x[idx]
    t = t[idx]

    for iter in range(iter_per_epoch):
        batch_x = x[iter*batch_size: (iter+1)*batch_size]
        batch_t = t[iter*batch_size: (iter+1)*batch_size]

        loss = model.forward(batch_x, batch_t)
        model.backward()
        optimizer.update(model.params, model.grads)

        total_loss += loss
        loss_count += 1

        if (iter+1) % iter_per_epoch == 0:
            avg_loss = total_loss / loss_count
            loss_list.append(avg_loss)
            print(f'epoch {epoch+1} | loss {avg_loss:.2f}')
            total_loss, loss_count = 0, 0

# Plot the results of loss value and data points shape
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ax1.plot(np.arange(len(loss_list)), loss_list, label='train')
ax1.set(xlabel='epoch', ylabel='loss', title='Training Loss')
ax1.legend()
ax1.grid(True, alpha=0.3)
# 右子图：决策边界 + 数据点
h = 0.001
x_min, x_max = x[:, 0].min() - .1, x[:, 0].max() + .1
y_min, y_max = x[:, 1].min() - .1, x[:, 1].max() + .1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
X = np.c_[xx.ravel(), yy.ravel()]
predict_cls = np.argmax(model.predict(X), axis=1)
ax2.contourf(xx, yy, predict_cls.reshape(xx.shape))
ax2.axis('off')
# 绘制数据点
x_data, t_data = load_data()
N, CLS_NUM = 100, 3
markers = ['o', 'x', '^']
for i in range(CLS_NUM):
    ax2.scatter(x_data[i*N:(i+1)*N, 0], x_data[i*N:(i+1)*N, 1], s=40, marker=markers[i])
ax2.set_title('Decision Boundary')
plt.tight_layout()
plt.show()
