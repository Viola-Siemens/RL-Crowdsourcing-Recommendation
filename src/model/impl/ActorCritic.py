from typing import Callable

import numpy as npy
import torch
from torch import nn
from torch.distributions import Categorical
from torch.optim import Optimizer

from data.Action import Action
from data.Environment import Environment
from model.FCNet import FCNet
from model.ReinforcementAlgorithm import ReinforcementAlgorithm

n_gram = 5


class ActorCritic(ReinforcementAlgorithm):
    def __init__(self, env: Environment):
        net = FCNet(env.get_state_dim() * n_gram, [(env.get_output_dim(), nn.Softmax(dim=1)), (1, None)])  # TODO
        super().__init__(net)

    def train(self, env: Environment, optimizer: Optimizer, epochs: int,
              logger: Callable[[int, float, float], None], **kwargs) -> None:
        gamma = kwargs.get('gamma', 0.99)

        for epoch in range(epochs):
            env.reset()
            history_state = env.get_history_states(n_gram - 1)
            state = env.get_state()  # 获取初始状态
            history_state.append(state)
            done = False
            total_reward = 0
            count = 0

            entropies = []

            while not done:
                count += 1
                state_tensor = torch.tensor(npy.array(history_state).reshape(-1), dtype=torch.float32).unsqueeze(0)
                
                action_probs, value = self.net(state_tensor)
                # action_probs_tensor = torch.tensor(action_probs[0], dtype=torch.float32, requires_grad=True)
                action_probs_tensor = action_probs.clone().detach()
                dist = Categorical(action_probs_tensor)
                action = dist.sample()
                
                reward = env.perform(Action(action.item()))

                next_state = env.get_state()
                history_state = env.get_history_states(n_gram - 1)
                history_state.append(next_state)
                done = env.is_done()
                total_reward += reward

                next_state_tensor = torch.tensor(npy.array(history_state).reshape(-1), dtype=torch.float32).unsqueeze(0)
                next_action_probs, next_value = self.net(next_state_tensor)

                target = reward + gamma * next_value.item() * (1 - done)
                advantage = torch.tensor(target - value.item(), dtype=torch.float32, requires_grad=True)

                # loss and update
                critic_loss = advantage.pow(2).mean()
                log_probs = torch.log(action_probs_tensor.squeeze(0))[action.item()]
                actor_loss = -(log_probs * advantage.detach()).mean()
                actor_loss.requires_grad_(True)
                loss = actor_loss + critic_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                entropies.append(dist.log_prob(torch.tensor(action.item(), device="cuda")))
            

            logger(epoch, total_reward / count, entropies[-1])


    def get_algorithm_name(self) -> str:
        return "actor-critic"
