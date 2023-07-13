from typing import Callable

import numpy as np
import torch
from torch import Tensor
from torch import nn
from torch.optim import Optimizer

from data.Action import Action
from data.Environment import Environment
from model.FCNet import FCNet
from model.ReinforcementAlgorithm import ReinforcementAlgorithm

n_gram = 5


class QLearning(ReinforcementAlgorithm):
    def __init__(self, env: Environment):
        policy_net = FCNet(env.get_state_dim() * n_gram, [(env.get_output_dim(), None)])

        super().__init__(policy_net)
        self.policy_net = policy_net
        self.gamma = 0.5
        self.epsilon = 0.9
        self.epsilon_decay = 0.99
        self.epsilon_min = 0.01
        self.loss_fn = nn.MSELoss()
        self.action_size = env.get_output_dim()

    def train(self, env: Environment, optimizer: Optimizer, epochs: int,
              logger: Callable[[int, float, float], None], **kwargs) -> None:
        for e in range(epochs):
            env.reset()
            loss = 0
            rewards = []
            it = 0
            while not env.is_done():
                it += 1
                hist = env.get_history_states(n_gram - 1)
                hist.append(env.get_state())
                state = hist
                state = Tensor(np.array(state).reshape(-1))

                # actions = self.policy_net.forward(state)[0]
                # action = Action(int(random.choice(actions.tolist())))
                action = env.sample()
                reward = env.perform(action)

                rewards.append(reward)

                hist = env.get_history_states(n_gram - 1)
                hist.append(env.get_state())
                next_state = hist
                next_state = Tensor(np.array(next_state).reshape(-1))
                next_actions = self.policy_net.forward(next_state)[0]
                next_action = Action(int(torch.max(next_actions).item()))

                target = reward + self.gamma * (next_action.get()/1807)
                loss_temp = ((target - (action.get())/1807) ** 2) / 2

                loss += loss_temp

            loss /= it
            optimizer.zero_grad()
            loss = torch.tensor(loss, requires_grad=True)
            loss.backward()
            optimizer.step()
            logger(e, float(np.mean(rewards)), 0)

    def get_algorithm_name(self) -> str:
        return "value-based"
