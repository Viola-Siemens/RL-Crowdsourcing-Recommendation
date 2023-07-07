from typing import Callable

from torch.optim import Optimizer

from data.Environment import Environment
from model.FCNet import FCNet
from model.ReinforcementAlgorithm import ReinforcementAlgorithm


class ActorCritic(ReinforcementAlgorithm):
    def __init__(self):
        net = FCNet(4, [(8, None)])  # TODO
        super().__init__(net)

    def train(self, env: Environment, optimizer: Optimizer, epochs: int,
              logger: Callable[[int, float], None], **kwargs) -> None:
        # TODO
        pass

    def get_algorithm_name(self) -> str:
        return "actor-critic"
