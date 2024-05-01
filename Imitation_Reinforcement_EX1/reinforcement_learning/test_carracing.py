from __future__ import print_function

import gym
import os
import json
from datetime import datetime
from agent.dqn_agent import DQNAgent
from train_carracing import run_episode
from agent.networks import *
import numpy as np

np.random.seed(0)

if __name__ == "__main__":

    env = gym.make("CarRacing-v0").unwrapped

    history_length = 0
    num_actions = 5
    
    Q_network = CNN(history_length=history_length, action_dim=num_actions)
    Q_network_target = CNN(history_length=history_length, action_dim=num_actions)
    agent = DQNAgent(Q_network, Q_network_target, num_actions,
                     epsilon=0.1, gamma=0.95, tau=0.01, lr=0.0001, batch_size=256,
                     buffer_size=int(1e5))
    agent.load(os.path.join("./models_carracing", "dqn_agent.pt"))

    n_test_episodes = 15

    episode_rewards = []
    for i in range(n_test_episodes):
        stats = run_episode(
            env, agent,     skip_frames=0, 
            deterministic=True, do_training=False, rendering=False
        )
        episode_rewards.append(stats.episode_reward)
        print("test episode: ", i, "reward: ", stats.episode_reward)

    # save results in a dictionary and write them into a .json file
    results = dict()
    results["episode_rewards"] = episode_rewards
    results["mean"] = np.array(episode_rewards).mean()
    results["std"] = np.array(episode_rewards).std()

    if not os.path.exists("./results"):
        os.mkdir("./results")

    fname = "./results/carracing_results_dqn-%s.json" % datetime.now().strftime(
        "%Y%m%d-%H%M%S"
    )
    fh = open(fname, "w")
    json.dump(results, fh)

    env.close()
    print("... finished")
