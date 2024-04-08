import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from game_handler import GameHandler

from collections import deque

import random

memory = (torch.tensor(()), int, torch.tensor(()), int, bool)

class DQN(nn.Module):
    def __init__(self, state_feature_count, h1_node_count, action_feature_count):
        super().__init__()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.fc1 = nn.Linear(state_feature_count, h1_node_count, device=device)
        self.fc2 = nn.Linear(h1_node_count, action_feature_count)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x


class ReplayMemory():
    def __init__(self, maxlen):
        self.memory = [None] * maxlen
        self.max_len = maxlen
        self.index = 0
        self.full = False

    def append(self, x: memory):
        self.memory[self.index] = x
        self.index += 1
        if self.index >= self.max_len:
            self.index = 0
            self.full = True

    def get_iterator(self, batch_size):
        if not self.full and self.index < batch_size:
            return []
        if self.full:
            dataset = TensorDataset(torch.Tensor(self.memory))
        else:
            dataset = TensorDataset(torch.Tensor(self.memory[:self.index-1]))
        return DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    def __len__(self):
        return len(self.memory)


class SnakeDQL:
    def __init__(self, memory_size=1024, h1_node_count=16, action_feature_count=4,
                 learning_rate=0.01, batch_size=32, discount_factor=0.9, network_sync_rate=10, epsilon_decay=0.001):

        self.state_feature_count = GameHandler.get_state_size()
        self.h1_node_count = h1_node_count
        self.action_feature_count = action_feature_count
        self.action_space = GameHandler.get_action_space()

        self.learning_rate = learning_rate
        self.batch_size = batch_size

        self.policy_dqn = DQN(self.state_feature_count, self.h1_node_count, self.action_feature_count)
        self.target_dqn = DQN(self.state_feature_count, self.h1_node_count, self.action_feature_count)

        # synchronize the two networks, will be repeated every (network_sync_rate) batches trained
        self.target_dqn.load_state_dict(self.policy_dqn.state_dict())
        self.network_sync_rate = network_sync_rate

        self.optimizer = torch.optim.Adam(self.policy_dqn.parameters(), lr=self.learning_rate)

        self.loss_fn = nn.MSELoss

        # exploration vs exploitation parameter
        self.epsilon = 1
        self.epsilon_decay = epsilon_decay

        # parameter which describes value of future rewards
        self.discount_factor = discount_factor

        # keeping track of how many batches the model were trained on
        self.trained_batches = 0

        self.memory = ReplayMemory(memory_size)


    def __call__(self, x):
        if random.random() <= self.epsilon:
            return random.randint(0, self.action_feature_count-1)
        else:
            return self.policy_dqn.forward(x).argmax()

    def memorize(self, mem: memory):
        state, action, new_state, reward, terminated = mem

        self.memory.append((state, action, new_state, reward, terminated))

    def train(self):

        for mini_batch in self.memory.get_iterator(self.batch_size):
            self.optimize(mini_batch)
            self.epsilon = max(0.01, self.epsilon - self.epsilon_decay)
            self.trained_batches += 1
            # sync the two dqn's according to network_sync_rate
            if self.trained_batches % self.network_sync_rate == 0:
                self.target_dqn.load_state_dict(self.policy_dqn.state_dict())

    def optimize(self, mini_batch):

        current_q_list = []
        target_q_list = []

        for state, action, new_state, reward, terminated in mini_batch:

            if terminated:
                target = torch.FloatTensor([reward])
            else:
                with torch.no_grad():
                    target = reward + self.discount_factor * self.target_dqn(new_state).max()

            # predicted q-values from model of current state
            current_q = self.policy_dqn(state)
            current_q_list.append(current_q)

            # updated q-values for current state
            target_q = self.target_dqn(state)
            target_q[action] = target
            target_q_list.append(target_q)

        loss = self.loss_fn(torch.stack(current_q_list), torch.stack(target_q_list))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()