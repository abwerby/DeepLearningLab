# export DISPLAY=:0
import os
import sys

sys.path.append(".")

import numpy as np
import gym
import time 
from torch.utils.tensorboard import SummaryWriter
from utils import EpisodeStats, rgb2gray
from utils import *
from agent.dqn_agent import DQNAgent
from agent.networks import CNN


def run_episode(
    env,
    agent,
    deterministic,
    skip_frames=0,
    do_training=True,
    rendering=False,
    max_timesteps=1000,
    history_length=0,
):
    """
    This methods runs one episode for a gym environment.
    deterministic == True => agent executes only greedy actions according the Q function approximator (no random actions).
    do_training == True => train agent
    """

    stats = EpisodeStats()

    # Save history
    image_hist = []

    step = 0
    state = env.reset()

    # fix bug of corrupted states without rendering in gym environment
    env.viewer.window.dispatch_events()

    # append image history to first state
    state = state_preprocessing(state)
    image_hist.extend([state] * (history_length + 1))
    state = np.array(image_hist).reshape(history_length + 1, 96, 96)

    while True:

        # TODO: get action_id from agent
        # Hint: adapt the probabilities of the 5 actions for random sampling so that the agent explores properly.
        action_id = agent.act(state, deterministic)
        action = id_to_action(action_id)
        
        # Hint: frame skipping might help you to get better results.
        reward = 0
        for _ in range(skip_frames + 1):
            next_state, r, terminal, info = env.step(action)
            reward += r

            if rendering:
                env.render()

            if terminal:
                break

        next_state = state_preprocessing(next_state)
        image_hist.append(next_state)
        image_hist.pop(0)
        next_state = np.array(image_hist).reshape(history_length + 1, 96, 96)

        if do_training:
            state = state.reshape(history_length + 1, 96, 96)
            agent.train(state, action_id, next_state, reward, terminal)

        stats.step(reward, action_id)

        state = next_state

        if terminal or (step * (skip_frames + 1)) > max_timesteps:
            break

        step += 1

    return stats


def train_online(
    env,
    agent,
    num_episodes,
    history_length=0,
    model_dir="./models_carracing",
    tensorboard_dir="./tensorboard",
):

    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    print("[INFO] train agent")
    time_ = time.strftime("%Y-%m-%d_%H-%M-%S")
    expermints_name = f"RL-Carracing_time_{time_}"
    writer = SummaryWriter(tensorboard_dir + "/" + expermints_name)
    writer.add_hparams({"history_length": history_length}, {})
    

    for i in range(num_episodes):
        # Hint: you can keep the episodes short in the beginning by changing max_timesteps (otherwise the car will spend most of the time out of the track)
        max_timesteps = int(200 * (i / 10 + 1))
        if max_timesteps > 1000: max_timesteps = 1000
        print("max_timesteps: %d" % max_timesteps)
        stats = run_episode(
            env,
            agent,
            history_length=history_length,
            max_timesteps=max_timesteps,
            rendering=False,
            deterministic=False,
            do_training=True,
        )
        # write episode data to tensorboard
        print("episode: ", i, "reward: ", stats.episode_reward)
        writer.add_scalar('Reward/train', stats.episode_reward, i)

        # TODO: evaluate your agent every 'eval_cycle' episodes using run_episode(env, agent, deterministic=True, do_training=False) to
        # check its performance with greedy actions only. You can also use tensorboard to plot the mean episode reward.
        eval_cycle = 20
        if i % eval_cycle == 0:
           for j in range(num_eval_episodes):
                stats = run_episode(env, agent, deterministic=True, do_training=False)
                writer.add_scalar('Reward/eval', stats.episode_reward, i + j)
                print("eval episode: ", j, "reward: ", stats.episode_reward)

        # store model.
        if i % eval_cycle == 0 or (i >= num_episodes - 1):
            agent.save(os.path.join(model_dir, "dqn_agent.pt"))

    env.close()
    writer.flush()
    writer.close()


def state_preprocessing(state):
    return rgb2gray(state).reshape(96, 96) / 255.0


if __name__ == "__main__":

    num_eval_episodes = 5
    eval_cycle = 20

    env = gym.make("CarRacing-v0").unwrapped

    # state dimension (4,) and 2 actions
    history_length = 0
    num_actions = 2
    
    Q_network = CNN(history_length=history_length, action_dim=num_actions)
    Q_network_target = CNN(history_length=history_length, action_dim=num_actions)
    agent = DQNAgent(Q_network, Q_network_target, num_actions,
                     epsilon=0.1, gamma=0.99, tau=0.01, lr=0.001, batch_size=256,
                     buffer_size=1e6)
    train_online(
        env, agent, num_episodes=1000, history_length=0, model_dir="./models_carracing"
    )
