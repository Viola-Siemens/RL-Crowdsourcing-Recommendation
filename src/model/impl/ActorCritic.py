from typing import Callable

from torch.optim import Optimizer

from data.Environment import Environment
from model.FCNet import FCNet
from model.ReinforcementAlgorithm import ReinforcementAlgorithm
import torch
import torch.nn as nn
from torch.distributions import Categorical
from data.Action import Action



n_gram = 5

class ActorCritic(ReinforcementAlgorithm):
    def __init__(self, env: Environment):
        net = FCNet(env.get_state_dim() * n_gram, [(env.get_output_dim(), None)])  # TODO
        super().__init__(net)
        self.env = env
        self.state_dim = env.get_state_dim()
        self.action_dim = env.get_output_dim()

        self.actor = FCNet(self.state_dim, [(self.action_dim, nn.Softmax(dim=1))])
        self.critic = FCNet(self.state_dim, [(1, None)])

    def train(self, env: Environment, optimizer: Optimizer, epochs: int,
              logger: Callable[[int, float, float], None], **kwargs) -> None:
        gamma = kwargs.get('gamma', 0.99)

        for epoch in range(epochs):
            env.reset()
            state = env.get_state()  # 获取初始状态
            done = False
            total_reward = 0
            count = 0

            while not done:
                count += 1
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                # Actor network
                action_probs = self.actor(state_tensor)
                action_probs_tensor = torch.tensor(action_probs[0], dtype=torch.float32, requires_grad=True)
                action = Action(torch.multinomial(action_probs_tensor, 1).item())

                dist = Categorical(action_probs_tensor)

                reward = self.env.perform(action)
                next_state = self.env.get_state()
                done = self.env.is_done()
                total_reward += reward

                next_state_tensor = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)

                # Critic network
                value = self.critic(state_tensor)[0].item()
                next_value = self.critic(next_state_tensor)[0].item()

                target = reward + gamma * next_value * (1 - done)
                advantage = torch.tensor(target - value, dtype=torch.float32, requires_grad=True)

                # Critic loss and update
                critic_loss = advantage.pow(2).mean()
                optimizer.zero_grad()
                critic_loss.backward()
                optimizer.step()

                # Actor loss and update
                log_probs = torch.log(action_probs_tensor.squeeze(0))[action.get()]
                actor_loss = -(log_probs * advantage.detach()).mean()
                optimizer.zero_grad()
                actor_loss.backward()
                optimizer.step()

                state = next_state

            logger(epoch, total_reward/count, -dist.entropy().item())

    def get_algorithm_name(self) -> str:
        return "actor-critic"