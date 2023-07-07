from typing import Callable

from torch.optim import Optimizer

from data.Environment import Environment
from model.ReinforcementAlgorithm import ReinforcementAlgorithm


class QLearning(ReinforcementAlgorithm):
    def train(self, env: Environment, optimizer: Optimizer, epochs: int,
              logger: Callable[[int, float], None], **kwargs) -> None:
        # TODO
        pass

    def get_algorithm_name(self) -> str:
        return "value-based"
