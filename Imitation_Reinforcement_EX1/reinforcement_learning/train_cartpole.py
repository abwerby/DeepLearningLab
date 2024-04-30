import sys

sys.path.append(".")

import os
import numpy as np
import time
import gym
import itertools as it
from agent.dqn_agent import DQNAgent
from torch.utils.tensorboard import SummaryWriter
from agent.networks import MLP
from utils import EpisodeStats


def run_episode(
    env, agent, deterministic, do_training=True, rendering=False, max_timesteps=1000
):
    """
    This methods runs one episode for a gym environment.
    deterministic == True => agent executes only greedy actions according the Q function approximator (no random actions).
    do_training == True => train agent
    """

    stats = EpisodeStats()  # save statistics like episode reward or action usage
    state = env.reset()

    step = 0
    while True:

        action_id = agent.act(state=state, deterministic=deterministic)
        next_state, reward, terminal, info = env.step(action_id)

        if do_training:
            agent.train(state, action_id, next_state, reward, terminal)

        stats.step(reward, action_id)

        state = next_state

        if rendering:
            env.render()

        if terminal or step > max_timesteps:
            break

        step += 1

    return stats


def train_online(
    env,
    agent,
    num_episodes,
    eval_cycle=20,
    num_eval_episodes=5,
    model_dir="./models_cartpole",
    tensorboard_dir="./tensorboard",
):
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    print("[INFO] train agent")
    time_ = time.strftime("%Y-%m-%d_%H-%M-%S")
    expermints_name = f"RL-Cart_time_{time_}"
    writer = SummaryWriter(tensorboard_dir + "/" + expermints_name)
    
    # training
    for i in range(num_episodes):
        stats = run_episode(env, agent, deterministic=False, do_training=True)
        # write episode data to tensorboard
        writer.add_scalar('Reward/train', stats.episode_reward, i)        
        print("episode: ", i, "reward: ", stats.episode_reward)
        if i % eval_cycle == 0:
           for j in range(num_eval_episodes):
                stats = run_episode(env, agent, deterministic=True, do_training=False)
                writer.add_scalar('Reward/eval', stats.episode_reward, i + j)
                print("eval episode: ", j, "reward: ", stats.episode_reward)
        # store model.
        if i % eval_cycle == 0 or i >= (num_episodes - 1):
            agent.save(os.path.join(model_dir, "dqn_agent.pt"))

    # close environment and tensorboard writer
    env.close()
    writer.flush()
    writer.close()

if __name__ == "__main__":


    num_eval_episodes = 5  # evaluate on 5 episodes
    eval_cycle = 20  # evaluate every 10 episodes

    env = gym.make("CartPole-v0").unwrapped

    # state dimension (4,) and 2 actions
    state_dim = 4
    num_actions = 2
    # create Q network and target network
    Q_network = MLP(state_dim, num_actions)
    Q_network_target = MLP(state_dim, num_actions)
    # create agent
    agent = DQNAgent(Q_network, Q_network_target, num_actions,
                     epsilon=0.1, gamma=0.99, tau=0.01, lr=0.001, batch_size=256,
                     buffer_size=1e6)
    # train agent
    train_online(env, agent, num_episodes=1000,
                 eval_cycle=eval_cycle, num_eval_episodes=num_eval_episodes)