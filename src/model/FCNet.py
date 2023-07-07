# -*- coding: utf-8 -*-

# @author 刘冬煜
# @desc 全连接网络的统一类，请按需继承或调用
# @date 2023/7/1

from typing import List, Callable, Tuple

import torch
import torch.nn.functional as F
from torch import nn, Tensor


class FCNet(nn.Module):
    input_channels: int
    activation: Callable[[Tensor], Tensor]
    hiddens: nn.Sequential
    outputs: List[nn.Module]
    device: torch.device

    # 构造函数，参数包括输入维度、输出维度与损失（可能有多个）、每层激活函数、隐含层维度、计算设备（cpu/cuda）
    # @see test.FCNet.model
    def __init__(self, input_channels: int,
                 outputs: List[Tuple[int, nn.Module]],
                 activation: Callable[[Tensor], Tensor] = F.relu,
                 hidden_dims: List[int] = None, device: torch.device = "cuda"):
        super(FCNet, self).__init__()
        self.input_channels = input_channels
        self.device = device
        if hidden_dims is None:
            hidden_dims = [256, 256]
        self.activation = activation
        self.outputs = []
        hiddens = []
        last_dim = input_channels
        for d in hidden_dims:
            hiddens.append(nn.Linear(last_dim, d, device=device))
            last_dim = d
        self.hiddens = nn.Sequential(*hiddens)
        for od, m in outputs:
            if m is None:
                self.outputs.append(nn.Linear(last_dim, od, device=device))
            else:
                self.outputs.append(nn.Sequential(nn.Linear(last_dim, od, device=device), m))

    # 前向传播，仅传入输入（一个batch下的）向量即可
    def forward(self, x: Tensor) -> List[Tensor]:
        x = x.to(self.device)
        for h in self.hiddens:
            x = self.activation(h(x))
        return [o(x) for o in self.outputs]
