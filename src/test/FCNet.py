import math
import time
from typing import List

import numpy as npy
import torch
from torch import nn
from torch.optim import Adam

from model.FCNet import FCNet

import torch.nn.functional as F

torch.set_default_tensor_type(torch.DoubleTensor)

model = FCNet(4, [(7, None), (1, nn.Sigmoid())], F.leaky_relu, hidden_dims=[32, 32])
optimizer = Adam(params=model.parameters(), lr=0.001, betas=(0.9, 0.99), weight_decay=1e-5, eps=1e-6)


def sigmoid(x: float) -> float:
    return 1 / (math.exp(-x) + 1)


def tanh(x: float) -> float:
    return math.tanh(x)


def Cmap1(x0: float, x1: float, x2: float, x3: float, noise: float = 0) -> List[float]:
    return [
        2 * x0 + 0.25,
        x1 - 0.5 * x0 - 0.25,
        x0 + x1 + x2 + x3 + noise * 0.0625,
        1.5 * x3 - 0.5 * x1 - noise / 6.0 + 0.1,
        x3 * x1,
        x2 - 0.125 * x3,
        x1 * x1 - x0 * x3 * 0.5
    ]


def Cmap2(x0: float, x1: float, x2: float, x3: float, noise: float = 0) -> List[float]:
    return [abs(sigmoid(x0 * 10 - x3 - 1) * tanh(x1 * 2 + x2 * 5 + noise * 0.5))]


dataX = npy.random.randn(256, 4)
noises = npy.random.randn(256)
dataY1 = torch.tensor(npy.array([
    Cmap1(dataX[i][0], dataX[i][1], dataX[i][2], dataX[i][3], noises[i]) for i in range(dataX.shape[0])
]), dtype=torch.float64, device="cuda")
dataY2 = torch.tensor(npy.array([
    Cmap2(dataX[i][0], dataX[i][1], dataX[i][2], dataX[i][3], noises[i]) for i in range(dataX.shape[0])
]), dtype=torch.float64, device="cuda")
dataX = torch.tensor(dataX, dtype=torch.float64)

start_time = time.time()
for e in range(1024):
    preds = model.forward(dataX)
    loss1 = F.mse_loss(preds[0], dataY1)
    loss2 = F.binary_cross_entropy(preds[1], dataY2)
    loss = loss1 * 7 + loss2
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if e % 64 == 0:
        print("epoch = %d, loss = %f" % (e, loss.cpu().item()))
end_time = time.time()
print("Time cost: " + str(end_time - start_time) + " seconds")

testX = [0.1, 0.2, 0.3, 0.4]
testY1 = Cmap1(*testX)
testY2 = Cmap2(*testX)
print("Input: " + str(testX))
print("Expect output: " + str(testY1 + testY2))
preds = model.forward(torch.tensor([testX], dtype=torch.float64, device="cuda"))
preds_list = preds[0].cpu().tolist()[0] + preds[1].cpu().tolist()[0]
print("FCNet output: " + str(preds_list))
loss1 = F.mse_loss(preds[0], torch.tensor([testY1], dtype=torch.float64, device="cuda")).cpu().item()
loss2 = F.binary_cross_entropy(preds[1], torch.tensor([testY2], dtype=torch.float64, device="cuda")).cpu().item()
print("Loss = " + str(loss1 * 7 + loss2))
