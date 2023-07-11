from typing import Callable

from torch.optim import Optimizer
from torch.distributions import Categorical
from torch import Tensor

from data.Environment import Environment
from data.Action import Action
from model.FCNet import FCNet
from model.ReinforcementAlgorithm import ReinforcementAlgorithm

import numpy as npy
import torch.nn.functional as F
import torch

n_gram = 5


class REINFORCE(ReinforcementAlgorithm):
    def __init__(self, env: Environment):
        # state_size * n_gram, action_size
        net = FCNet(env.get_state_dim() * n_gram, [(env.get_output_dim(), None)])  # TODO
        super().__init__(net)

    def train(self, env: Environment, optimizer: Optimizer, epochs: int,
              logger: Callable[[int, float, float], None], **kwargs) -> None:
        for e in range(epochs):
            env.reset()
            log_probs = []
            rewards = []
            for t in range(100):
                action_value, log_prob = self.select_action(env)
                log_probs.append(log_prob)
                action = Action(action_value)
                reward = env.perform(action)
                rewards.append(reward)
                if env.is_done():
                    break
            
            n_steps = len(rewards)
            loss = 0
            gamma = 1.0
            R = 0
            for t in reversed(range(n_steps)):
                R =  gamma * R + rewards[t]
                loss = loss - log_probs[t] * R
            loss = loss/n_steps

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            
            logger(e, npy.mean(rewards), log_probs[0])

            

    def select_action(self, env: Environment):
        hist = env.get_history_states(n_gram)
        hist = Tensor(npy.array(hist).reshape(-1))
        x = self.net(hist)[0]
        probs = F.softmax(x)
        m = Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action)
    

    def get_algorithm_name(self) -> str:
        return "policy-based"
