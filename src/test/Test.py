import math
from typing import List

from matplotlib import pyplot as plt
from torch.optim import RMSprop

from data.Environment import Environment
from model.impl.ActorCritic import ActorCritic
from model.impl.PolicyBased import REINFORCE
from model.impl.ValueBased import QLearning

rewards = []
entropies = []


def logger(epoch, reward, entropy):
    print("epoch = %d, reward = %f, entropy = %f" % (epoch, reward, entropy))
    rewards.append(reward)
    entropies.append(entropy)


def C(n, k):
    return math.factorial(n) // (math.factorial(k) * math.factorial(n - k))


def smooth(lst: List[float]) -> List[float]:
    n = 18
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
    env = Environment()
    plt.figure(figsize=(16, 12))
    reward_types = ["w", "r", "rn1", "rn2"]
    for r in range(len(reward_types)):
        print("Reward Type: '%s'" % reward_types[r])
        env.set_reward_type(reward_types[r])
        models = [ActorCritic(env), REINFORCE(env), QLearning(env)]
        for m in range(len(models)):
            print("Training %s" % models[m].get_algorithm_name())
            rewards = []
            entropies = []
            models[m].train(
                env,
                RMSprop(params=models[m].net.parameters(), lr=0.01, weight_decay=1e-6, eps=1e-8),
                128,
                logger
            )
            ax1 = plt.subplot(3, 4, r + m * 4 + 1)
            smooth_rewards = smooth(rewards)
            ax1.plot(rewards, color='b', alpha=0.3)
            line1, = ax1.plot(smooth_rewards, label="reward", color='b')
            ax2 = ax1.twinx()
            smooth_entropies = smooth(entropies)
            ax2.plot(entropies, color='r', alpha=0.3)
            line2, = ax2.plot(smooth_entropies, label="entropy", color='r')
            plt.legend(handles=[line1, line2])
            dev_r, dev_e = models[m].test(env)
            print("Dev reward = %f, Dev Entropy = %f" % (dev_r, dev_e))

    plt.tight_layout()
    plt.show()
