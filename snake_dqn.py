import torch
import torch.nn as nn
import torch.nn.functional as F

from game_handler import GameHandler

from collections import deque

import random

class DQN(nn.Module):
    def __init__(self, state_feature_count, h1_node_count, action_feature_count):
        super().__init__()

        device = torch.device("gpu" if torch.cuda.is_available() else "cpu")

        self.fc1 = nn.Linear(state_feature_count, h1_node_count, device=device)
        self.fc2 = nn.Linear(h1_node_count, action_feature_count)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x


class ReplayMemory():
    def __init__(self, maxlen):
        self.memory = deque([], maxlen=maxlen)

    def append(self, x):
        self.memory.append(x)

    def sample(self, sample_size):
        return random.sample(self.memory, sample_size)

    def __len__(self):
        return len(self.memory)


class SnakeDQL():
    def __init__(self, game_handlers, memory_size, h1_node_count=16, action_feature_count=4, learning_rate=0.01):
        self.game_handlers = game_handlers

        self.state_feature_count = GameHandler.get_state_size()
        self.h1_node_count = h1_node_count
        self.action_feature_count = action_feature_count

        self.learning_rate = learning_rate

        self.optimizer = None

        # exploration vs exploitation parameter
        self.epsilon = 1

        self.memory = ReplayMemory(memory_size)



    def train(self, episodes):
        policy_dqn = DQN(self.state_feature_count, self.h1_node_count, self.action_feature_count)
        target_dqn = DQN(self.state_feature_count, self.h1_node_count, self.action_feature_count)

        # synchronize the two networks
        target_dqn.load_state_dict(policy_dqn.state_dict())

        self.optimizer = torch.optim.Adam(policy_dqn.parameters(), lr=self.learning_rate)

