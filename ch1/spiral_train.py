import numpy as np
import matplotlib.pyplot as plt
from ch1.two_layer_net import TwoLayerNet
from dataset.spiral import load_data
from common.optimizer import SGD


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

# 绘制学习结果
plt.plot(np.arange(len(loss_list)), loss_list, label='train')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()

# 绘制决策边界
h = 0.001
x_min, x_max = x[:, 0].min() - .1, x[:, 0].max() + .1
y_min, y_max = x[:, 1].min() - .1, x[:, 1].max() + .1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
X = np.c_[xx.ravel(), yy.ravel()]
score = model.predict(X)
predict_cls = np.argmax(score, axis=1)
Z = predict_cls.reshape(xx.shape)
plt.contourf(xx, yy, Z)
plt.axis('off')

# 绘制数据点
x, t = load_data()
N = 100
CLS_NUM = 3
markers = ['o', 'x', '^']
for i in range(CLS_NUM):
    plt.scatter(x[i*N:(i+1)*N, 0], x[i*N:(i+1)*N, 1], s=40, marker=markers[i])
plt.show()
