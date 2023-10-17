import random
import numpy as np

import gym
from gym import wrappers
from gym.envs.registration import register


#print([env for env in gym.envs.registry.all()])

fl_env = gym.make('FrozenLake-v0')

# Reset the state
state = fl_env.reset()

register(
    id='FrozenLakeNotSlippery-v0',
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name': '4x4', 'is_slippery': False},
    max_episode_steps=100,
    reward_threshold=0.78,  # optimum = .8196
)

fl_ns_env = gym.make('FrozenLakeNotSlippery-v0')

# Reset the state
state = fl_ns_env.reset()

import sys, os
sys.path.append(os.path.abspath('./'))

from alcs import ACS2, ACS2Configuration


CLASSIFIER_LENGTH = 16  # Because we are operating in 4x4 grid
POSSIBLE_ACTIONS = fl_env.action_space.n  # 4


def one_hot_encode(state):
    vec = ['0' for i in range(CLASSIFIER_LENGTH)]
    vec[state] = 'X'
    return ''.join(vec)
    
# We assume if the final state was with number 15 that the algorithm found the reward. Otherwise not
def collect_env_metrics(env):
    return {'found_reward': env.env.s == 15}

def print_performance(population, metrics):
    population.sort(key=lambda cl: -cl.fitness)
    population_count = len(population)
    reliable_count = len([cl for cl in population if cl.is_reliable()])
    successful_trials = sum(m['environment']['found_reward'] for m in metrics) #m['found_reward'] for m in metrics)

    print("Number of classifiers: {}".format(population_count))
    print("Number of reliable classifiers: {}".format(reliable_count))
    print("Percentage of successul trials: {:.2f}%".format(successful_trials / EXPLOIT_TRIALS  * 100))
    print("\nTop 10 classifiers:")
    for cl in population[:10]:
        print("{!r} \tq: {:.2f} \tr: {:.2f} \tir: {:.2f} \texp: {}".format(cl, cl.q, cl.r, cl.ir, cl.exp))



cfg = ACS2Configuration(
    classifier_length=CLASSIFIER_LENGTH,
    number_of_possible_actions=POSSIBLE_ACTIONS,
    perception_mapper_fcn=one_hot_encode,
    environment_metrics_fcn=collect_env_metrics,
    theta_i=0.3,
    epsilon=0.7)

# print(cfg)


EXPLORE_TRIALS = 2000
EXPLOIT_TRIALS = 100

def perform_experiment(cfg, env):
    # explore phase
    agent = ACS2(cfg)
    population_explore, metrics_explore = agent.explore(env, EXPLORE_TRIALS)
    
    # exploit phase, reinitialize agent with population above
    agent = ACS2(cfg, population=population_explore)
    population_exploit, metrics_exploit = agent.exploit(env, EXPLOIT_TRIALS)
    
    return (population_explore, metrics_explore), (population_exploit, metrics_exploit)




explore_results, exploit_results = perform_experiment(cfg, fl_env)

#print(explore_results[1])

# # exploration
# print_performance(explore_results[0], explore_results[1])

# # exploitation
# print_performance(exploit_results[0], exploit_results[1])


print(print(f"\n\n\nScore: {np.mean([m['agent']['fitness'] for m in explore_results[1] if m['environment']['found_reward']])}\n\n\n"))
