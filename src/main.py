import argparse

from model.impl.ActorCritic import ActorCritic
from model.impl.PolicyBased import PPO
from model.impl.ValueBased import QLearning

algorithms = {}

for model in [ActorCritic(), PPO(), QLearning()]:
    algorithms[model.get_algorithm_name()] = model

parser = argparse.ArgumentParser()
parser.add_argument("--algo", type=str, choices=algorithms.keys(), help="Reinforcement Learning Algorithm")
parser.add_argument("--epochs", type=int, help="Total Epoch of Training")

# 训练数据直接解压在src/resources中，如"src/resources/project_list.csv"、"src/resources/entry/entry_19393_24.txt"等

if __name__ == "__main__":
    # TODO
    parser.print_help()
