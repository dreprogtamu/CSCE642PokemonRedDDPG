import random
from copy import deepcopy
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.optim import Adam
from torch.distributions.normal import Normal

class QNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes):
        super().__init__()
        sizes = [obs_dim + act_dim] + hidden_sizes + [1]
        self.layers = nn.ModuleList()
        for i in range(len(sizes) - 1):
            self.layers.append(nn.Linear(sizes[i], sizes[i + 1]))

    def forward(self, obs, act):
        x = torch.cat([obs, act], dim=-1)
        for i in range(len(self.layers) - 1):
            x = F.relu(self.layers[i](x))
        return self.layers[-1](x).squeeze(dim=-1)

class PolicyNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim, act_lim, hidden_sizes):
        super().__init__()
        sizes = [obs_dim] + hidden_sizes + [act_dim]
        self.act_lim = act_lim
        self.layers = nn.ModuleList()
        for i in range(len(sizes) - 1):
            self.layers.append(nn.Linear(sizes[i], sizes[i + 1]))

    def forward(self, obs):
        x = obs
        for i in range(len(self.layers) - 1):
            x = F.relu(self.layers[i](x))
        return self.act_lim * torch.tanh(self.layers[-1](x))

class ActorCriticNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim, act_lim, hidden_sizes):
        super().__init__()
        self.q = QNetwork(obs_dim, act_dim, hidden_sizes)
        self.pi = PolicyNetwork(obs_dim, act_dim, act_lim, hidden_sizes)

class DDPG:
    def __init__(self, policy, env, verbose=0, **kwargs):
        self.env = env
        self.verbose = verbose
        self.options = {
            'gamma': kwargs.get('gamma', 0.99),
            'batch_size': kwargs.get('batch_size', 64),
            'steps': kwargs.get('steps', 1000),
            'replay_memory_size': kwargs.get('replay_memory_size', 1000000),
            'layers': kwargs.get('layers', [256, 256]),
            'alpha': kwargs.get('alpha', 1e-3)
        }
        # Create actor-critic network
        self.actor_critic = ActorCriticNetwork(
            env.observation_space.shape[0],
            env.action_space.shape[0],
            env.action_space.high[0],
            self.options['layers'],
        )
        # Create target actor-critic network
        self.target_actor_critic = deepcopy(self.actor_critic)

        self.optimizer_q = Adam(self.actor_critic.q.parameters(), lr=self.options['alpha'])
        self.optimizer_pi = Adam(self.actor_critic.pi.parameters(), lr=self.options['alpha'])

        # Freeze target actor critic network parameters
        for param in self.target_actor_critic.parameters():
            param.requires_grad = False

        # Replay buffer
        self.replay_memory = deque(maxlen=self.options['replay_memory_size'])

        if self.verbose > 0:
            print("DDPG initialized with options:", self.options)

    @staticmethod
    def load(path, env, **kwargs):
        """
        Loads a saved DDPG model.
        """
        model = DDPG('CnnPolicy', env, **kwargs)
        checkpoint = torch.load(path)
        model.actor_critic.load_state_dict(checkpoint['actor_critic'])
        model.target_actor_critic.load_state_dict(checkpoint['target_actor_critic'])
        return model

    def save(self, path):
        """
        Saves the DDPG model.
        """
        torch.save({
            'actor_critic': self.actor_critic.state_dict(),
            'target_actor_critic': self.target_actor_critic.state_dict()
        }, path)

    @torch.no_grad()
    def select_action(self, state):
        """
        Selects an action given state.

         Returns:
            The selected action.
        """
        state = torch.as_tensor(state, dtype=torch.float32)
        mu = self.actor_critic.pi(state)
        m = Normal(torch.zeros(self.env.action_space.shape[0]), torch.ones(self.env.action_space.shape[0]))
        noise_scale = 0.1
        action_limit = self.env.action_space.high[0]
        action = mu + noise_scale * m.sample()
        return torch.clip(action, -action_limit, action_limit).numpy()

    def learn(self, total_timesteps, callback=None):
        """
        Runs the training process for the specified number of timesteps.
        """
        num_episodes = total_timesteps // self.options['steps']
        for episode in range(num_episodes):
            self.train_episode()
            if callback:
                callback.on_step()

    def train_episode(self):
        """
        Runs a single episode of the DDPG algorithm.
        """
        state, _ = self.env.reset()
        for _ in range(self.options['steps']):
            action = self.select_action(state)
            next_state, reward, done, _ = self.env.step(action)
            self.memorize(state, action, reward, next_state, done)
            state = next_state

            if len(self.replay_memory) >= self.options['batch_size']:
                self.replay()
                self.update_target_networks()

            if done:
                break

    def memorize(self, state, action, reward, next_state, done):
        """
        Adds transitions to the replay buffer.
        """
        self.replay_memory.append((state, action, reward, next_state, done))

    @torch.no_grad()
    def update_target_networks(self, tau=0.995):
        """
        Copy params from actor_critic to target_actor_critic using Polyak averaging.
        """
        for param, param_targ in zip(self.actor_critic.parameters(), self.target_actor_critic.parameters()):
            param_targ.data.mul_(tau)
            param_targ.data.add_((1 - tau) * param.data)

    def replay(self):
        """
        Samples transitions from the replay memory and updates actor_critic network.
        """
        if len(self.replay_memory) > self.options['batch_size']:
            minibatch = random.sample(self.replay_memory, self.options['batch_size'])
            states, actions, rewards, next_states, dones = map(np.array, zip(*minibatch))

            # Convert numpy arrays to torch tensors
            states = torch.as_tensor(states, dtype=torch.float32)
            actions = torch.as_tensor(actions, dtype=torch.float32)
            rewards = torch.as_tensor(rewards, dtype=torch.float32)
            next_states = torch.as_tensor(next_states, dtype=torch.float32)
            dones = torch.as_tensor(dones, dtype=torch.float32)

            # Current Q-values
            current_q = self.actor_critic.q(states, actions)
            # Target Q-values
            target_q = self.compute_target_values(next_states, rewards, dones)

            # Optimize critic network
            loss_q = self.q_loss(current_q, target_q).mean()
            self.optimizer_q.zero_grad()
            loss_q.backward()
            self.optimizer_q.step()

            # Optimize actor network
            loss_pi = self.pi_loss(states).mean()
            self.optimizer_pi.zero_grad()
            loss_pi.backward()
            self.optimizer_pi.step()

    def compute_target_values(self, next_states, rewards, dones):
        """
        Computes the target q values.

        Returns:
            The target q value (as a tensor).
        """
        next_actions = self.target_actor_critic.pi(next_states)
        target_q_values = self.target_actor_critic.q(next_states, next_actions)
        gamma = self.options['gamma']
        targets = rewards + gamma * (1 - dones) * target_q_values
        return targets

    def q_loss(self, current_q, target_q):
        """
        The q loss function.

        Returns:
            The unreduced loss (as a tensor).
        """
        return F.mse_loss(current_q, target_q, reduction='none')

    def pi_loss(self, states):
        """
        The policy gradient loss function.

        Returns:
            The unreduced loss (as a tensor).
        """
        actions = self.actor_critic.pi(states)
        loss_pi = -self.actor_critic.q(states, actions)
        return loss_pi
