import numpy as np
import torch
import torch.optim as optim
from agent.replay_buffer import ReplayBuffer


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


class DQNAgent:

    def __init__(
        self,
        Q,
        Q_target,
        num_actions,
        gamma=0.95,
        batch_size=32,
        epsilon=0.01,
        tau=0.01,
        lr=1e-4,
        buffer_size=100,
    ):
        """
        Q-Learning agent for off-policy TD control using Function Approximation.
        Finds the optimal greedy policy while following an epsilon-greedy policy.

        Args:
           Q: Action-Value function estimator (Neural Network)
           Q_target: Slowly updated target network to calculate the targets.
           num_actions: Number of actions of the environment.
           gamma: discount factor of future rewards.
           batch_size: Number of samples per batch.
           tau: indicates the speed of adjustment of the slowly updated target network.
           epsilon: Chance to sample a random action. Float betwen 0 and 1.
           lr: learning rate of the optimizer
           buffer_size: size of the replay buffer
        """
        # setup networks
        self.Q = Q.cuda()
        self.Q_target = Q_target.cuda()
        self.Q_target.load_state_dict(self.Q.state_dict())

        # define replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size)

        # parameters
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.epsilon = epsilon

        self.loss_function = torch.nn.MSELoss()
        self.optimizer = optim.Adam(self.Q.parameters(), lr=lr)

        self.num_actions = num_actions

    def train(self, state, action, next_state, reward, terminal):
        """
        This method stores a transition to the replay buffer and updates the Q networks.
        """
        # add current transition to replay buffer
        self.replay_buffer.add_transition(state, action, next_state, reward, terminal)
        # sample next batch
        batch_states, batch_actions, batch_next_states, batch_rewards, batch_dones = self.replay_buffer.next_batch(self.batch_size)
        # convert to tensors
        states_tensors = [torch.tensor(state).float() for state in batch_states]
        states_tensor = torch.stack(states_tensors).cuda()
        actions_tensor = torch.tensor(batch_actions).unsqueeze(1).cuda()
        rewards_tensor = torch.tensor(batch_rewards).unsqueeze(1).cuda()
        next_states_tensors = [torch.tensor(next_state).float() for next_state in batch_next_states]
        next_states_tensor = torch.stack(next_states_tensors).cuda()
        dones_tensor = torch.tensor(batch_dones).unsqueeze(1).float().cuda()
        # compute td targets and loss
        qvalues = self.Q(states_tensor) 
        next_qvalues = self.Q_target(next_states_tensor)
        qvalue_selected = qvalues.gather(1, actions_tensor) 
        expected_qvalues = rewards_tensor + (self.gamma * torch.max(next_qvalues, dim=1, keepdim=True)[0] * (1 - dones_tensor))
        loss = self.loss_function(qvalue_selected.float(), expected_qvalues.float())
        # optimize the Q network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # call soft update for target network
        soft_update(self.Q_target, self.Q, self.tau)
        
        
    def act(self, state, deterministic):
        """
        This method creates an epsilon-greedy policy based on the Q-function approximator and epsilon (probability to select a random action)
        Args:
            state: current state input
            deterministic:  if True, the agent should execute the argmax action (False in training, True in evaluation)
        Returns:
            action id
        """
        r = np.random.uniform()
        if deterministic or r > self.epsilon:
            action_id = self.Q(torch.tensor(state).cuda().float().unsqueeze(0)).argmax().item()
        else:
            # Hint for the exploration in CarRacing: sampling the action from a uniform distribution will probably not work.
            # You can sample the agents actions with different probabilities (need to sum up to 1) so that the agent will prefer to accelerate or going straight.
            # To see how the agent explores, turn the rendering in the training on and look what the agent is doing.
            action_id = np.random.choice(self.num_actions)
        return action_id

    def save(self, file_name):
        torch.save(self.Q.state_dict(), file_name)

    def load(self, file_name):
        self.Q.load_state_dict(torch.load(file_name))
        self.Q_target.load_state_dict(torch.load(file_name))
