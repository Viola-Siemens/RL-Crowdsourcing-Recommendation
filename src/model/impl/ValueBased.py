from typing import Callable

from torch.optim import Optimizer

from data.Environment import Environment
from model.FCNet import FCNet
from model.ReinforcementAlgorithm import ReinforcementAlgorithm

n_gram = 5


class QLearning(ReinforcementAlgorithm):
    def __init__(self, env: Environment):
        net = FCNet(env.get_state_dim() * n_gram, [(env.get_output_dim(), None)])  # TODO
        super().__init__(net)

    def train(self, env: Environment, optimizer: Optimizer, epochs: int,
              logger: Callable[[int, float, float], None], **kwargs) -> None:
        # TODO
        pass

    def get_algorithm_name(self) -> str:
        return "value-based"
