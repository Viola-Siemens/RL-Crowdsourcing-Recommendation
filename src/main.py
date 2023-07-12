import argparse
import math
from typing import List

from matplotlib import pyplot as plt
from torch.optim import Adam, RMSprop, Adagrad, SGD

from data.Environment import Environment
from model.impl.ActorCritic import ActorCritic
from model.impl.PolicyBased import REINFORCE
from model.impl.ValueBased import QLearning

env = Environment()
algorithms = {}
optimizers = {
    "adam": lambda x, lr: Adam(params=x.parameters(), lr=lr),
    "rmsprop": lambda x, lr: RMSprop(params=x.parameters(), lr=lr),
    "adagrad": lambda x, lr: Adagrad(params=x.parameters(), lr=lr),
    "sgd": lambda x, lr: SGD(params=x.parameters(), lr=lr),
}

for model in [ActorCritic(env), REINFORCE(env), QLearning(env)]:
    algorithms[model.get_algorithm_name()] = model

parser = argparse.ArgumentParser()
parser.add_argument("--algo", type=str, choices=algorithms.keys(), help="Reinforcement Learning Algorithm")
parser.add_argument("--epochs", type=int, help="Total Epoch of Training")
parser.add_argument("--reward_type", type=str, choices=['w', 'r', 'rn1', 'rn2'], help="Type of Reward")
parser.add_argument("--optimizer", type=str, choices=optimizers.keys(), help="Optimizer of Training")
parser.add_argument("--lr", type=float, default=0.001, help="Learning Rate for optimizer")


# 训练数据直接解压在src/resources中，如"src/resources/project_list.csv"、"src/resources/entry/entry_19393_24.txt"等

def logger(epoch, reward, entropy):
    print("epoch = %d, reward = %f, entropy = %f\n" % (epoch, reward, entropy))
    rewards.append(reward)
    entropies.append(entropy)


def C(n, k):
    return math.factorial(n) // (math.factorial(k) * math.factorial(n - k))


def smooth(lst: List[float]) -> List[float]:
    n = 32
    lst = [lst[0] for _ in range(n // 2)] + lst
    ret = []
    kernel = [(C(n, i) / 2 ** n) for i in range(n + 1)]
    for i in range(0, len(lst) - n):
        v = 0
        for j in range(n + 1):
            v += kernel[j] * lst[i + j]
        ret.append(v)
    return ret


if __name__ == "__main__":
    # parser.print_help()
    args = parser.parse_args()
    model = algorithms.get(args.algo)
    env.set_reward_type(args.reward_type)
    rewards = []
    entropies = []
    model.train(
        env,
        optimizers[args.optimizer](model.net, args.lr),
        args.epochs,
        logger
    )
    fig = plt.figure(figsize=(10, 6))
    ax1 = fig.add_subplot(111)
    smooth_rewards = smooth(rewards)
    ax1.plot(rewards, color='b', alpha=0.4)
    line1, = ax1.plot(smooth_rewards, label="reward", color='b')
    ax2 = ax1.twinx()
    smooth_entropies = smooth(entropies)
    ax2.plot(entropies, color='r', alpha=0.4)
    line2, = ax2.plot(smooth_entropies, label="entropy", color='r')
    plt.legend(handles=[line1, line2])
    plt.show()
