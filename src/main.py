import argparse

from torch.optim import Adam, RMSprop, Adagrad, SGD

from data.Environment import Environment
from model.impl.ActorCritic import ActorCritic
from model.impl.PolicyBased import PPO
from model.impl.ValueBased import QLearning

env = Environment()
algorithms = {}
optimizers = {
    "adam": lambda x, lr: Adam(params=x.parameters(), lr=lr),
    "rmsprop": lambda x, lr: RMSprop(params=x.parameters(), lr=lr),
    "adagrad": lambda x, lr: Adagrad(params=x.parameters(), lr=lr),
    "sgd": lambda x, lr: SGD(params=x.parameters(), lr=lr),
}

for model in [ActorCritic(env), PPO(env), QLearning(env)]:
    algorithms[model.get_algorithm_name()] = model

parser = argparse.ArgumentParser()
parser.add_argument("--algo", type=str, choices=algorithms.keys(), help="Reinforcement Learning Algorithm")
parser.add_argument("--epochs", type=int, help="Total Epoch of Training")
parser.add_argument("--reward_type", type=str, choices=['w', 'r', 'rn1', 'rn2'], help="Type of Reward")
parser.add_argument("--optimizer", type=str, choices=optimizers.keys(), help="Optimizer of Training")
parser.add_argument("--lr", type=float, default=0.001, help="Learning Rate for optimizer")

# 训练数据直接解压在src/resources中，如"src/resources/project_list.csv"、"src/resources/entry/entry_19393_24.txt"等

if __name__ == "__main__":
    # parser.print_help()
    args = parser.parse_args()
    model = algorithms.get(args.algo)
    env.set_reward_type(args.reward_type)
    model.train(
        env,
        optimizers[args.optimizer](model.net.parameters(), args.lr),
        args.epochs,
        lambda e, reward, entropy: print("epoch = %d, reward = %f, entropy = %f" % (e, reward, entropy))
    )
