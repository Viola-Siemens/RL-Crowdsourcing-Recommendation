from abc import ABC, abstractmethod
from typing import Callable, Tuple

from torch.optim import Optimizer, SGD

from data.Environment import Environment
from model.FCNet import FCNet

_ret = (0, 0)

_temp_net = FCNet(1, [(1, None)])


def set_ret(e, reward, entropy):
    global _ret
    _ret = (reward, entropy)


class ReinforcementAlgorithm(ABC):
    net: FCNet

    def __init__(self, net: FCNet):
        # 继承后的类的构造函数的签名请设置为：__init__(self)
        self.net = net

    @abstractmethod
    def train(self, env: Environment, optimizer: Optimizer, epochs: int,
              logger: Callable[[int, float, float], None], **kwargs) -> None:
        # 算法核心需要训练的部分，env为算法需要交互的环境，optimizer为优化器，epochs为训练总轮数，
        # logger为输出函数（传入的参数分别为当前epoch、reward、entropy值），其余参数如有必要在kwargs中传递
        pass

    @abstractmethod
    def get_algorithm_name(self) -> str:
        # 算法的名字，即三种方法中的哪一种，请统一用全小写、无空格的字符串
        pass

    def test(self, env: Environment) -> Tuple[float, float]:
        global _ret
        env.set_test(True)
        self.train(env, SGD(_temp_net.parameters(), lr=0), 1, set_ret)
        env.set_test(False)
        return _ret
