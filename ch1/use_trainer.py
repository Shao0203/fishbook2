from ch1.two_layer_net import TwoLayerNet
from dataset.spiral import load_data
from common.optimizer import SGD
from common.trainer import Trainer


x, t = load_data()
model = TwoLayerNet(2, 10, 3)
optimizer = SGD(1.0)
trainer = Trainer(model, optimizer)
trainer.fit(x, t, 300, 30, eval_interval=10)
trainer.plot()
