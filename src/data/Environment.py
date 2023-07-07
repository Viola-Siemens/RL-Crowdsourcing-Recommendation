from torch import Tensor

from data.Action import Action


class Environment:
    done: bool

    def __init__(self):
        # TODO 构造函数

        self.done = False

    def reset(self) -> None:
        # TODO 初始化环境状态
        pass

    def sample(self) -> Action:
        # TODO 随机选择一个可行的行为
        pass

    def perform(self, action: Action) -> float:
        # TODO 执行行为获得奖励值
        pass

    def get_state(self) -> Tensor:
        # TODO 得到一个可以作为网络输入的state
        pass