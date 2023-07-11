import argparse

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
    ax1.plot(rewards, label="reward")
    ax2 = ax1.twinx()
    ax2.plot(entropies, label="entropy")
    plt.show()
