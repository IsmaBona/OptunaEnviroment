import os
import gymnasium as gym
import random

from deustorl.common import *
from deustorl.qlearning import QLearning
from deustorl.helpers import DiscretizedObservationWrapper

def test(algo, n_steps= 60000, **kwargs):
    epsilon_greedy_policy = EpsilonGreedyPolicy(epsilon=0.3)
    start_time = time.time()
    algo.learn(epsilon_greedy_policy, n_steps, **kwargs)
    print("----- {:0.4f} secs. -----".format(time.time() - start_time))

    return evaluate_policy(algo.env, algo.q_table, max_policy, n_episodes=100, verbose=False)

os.system("rm -rf ./logs/")

env_name = "Acrobot-v1"
env = DiscretizedObservationWrapper(gym.make(env_name), n_bins=10)

seed = 20
random.seed(seed)
env.reset(seed=seed)

n_rounds = 10
n_steps_per_round = 3_000_000

print("Testing Best Trial")
total_reward = 0
for _ in range(n_rounds):
    algo = QLearning(env)
    avg_reward, _ = test(algo, n_steps=n_steps_per_round, discount_rate=0.9500000000000001, \
                         lr=0.08529125979056351, lrdecay=0.9500000000000001, \
                         n_episodes_decay=100)
    total_reward += avg_reward
print("------\r\nAverage reward over {} rounds: {:.4f}\r\n------".format(n_rounds, total_reward/n_rounds))
input("Press any key to train the next trial...")

print("Testing Second Best Trial")
total_reward = 0
for _ in range(n_rounds):
    algo = QLearning(env)
    avg_reward, _ = test(algo, n_steps=n_steps_per_round, discount_rate=0.9500000000000001, \
                         lr=0.0994615145947944, lrdecay=0.97, \
                         n_episodes_decay=100)
    total_reward += avg_reward
print("------\r\nAverage reward over {} rounds: {:.4f}\r\n------".format(n_rounds, total_reward/n_rounds))
input("Press any key to train the next trial...")

print("Testing Third Best Trial")
total_reward = 0
for _ in range(n_rounds):
    algo = QLearning(env)
    avg_reward, _ = test(algo, n_steps=n_steps_per_round, discount_rate=0.9500000000000001, \
                         lr=0.06342416842598908, lrdecay=0.9500000000000001, \
                         n_episodes_decay=100)
    total_reward += avg_reward
print("------\r\nAverage reward over {} rounds: {:.4f}\r\n------".format(n_rounds, total_reward/n_rounds))