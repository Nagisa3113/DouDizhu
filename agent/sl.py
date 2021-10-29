import random
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import collections

Transition = collections.namedtuple('Transition', 'info_state action_probs')


class SLAgent(object):

    def __init__(self, para, num_actions=4, state_shape=None, device=None):
        self.use_raw = False
        self._num_actions = num_actions
        self._state_shape = state_shape
        self.device = device

        self._layer_sizes = para.hidden_layers_sizes + [num_actions]
        self._batch_size = para.batch_size
        self._train_every = para.train_every
        self._sl_learning_rate = para.learning_rate
        self._min_buffer_size_to_learn = para.min_buffer_size_to_learn
        self._reservoir_buffer = ReservoirBuffer(para.reservoir_buffer_capacity)

        self.total_t = 0

        self._build_model()

    def _build_model(self):
        policy_network = AveragePolicyNetwork(self._num_actions, self._state_shape, self._layer_sizes)
        policy_network = policy_network.to(self.device)
        self.policy_network = policy_network
        self.policy_network.eval()

        # xavier init
        for p in self.policy_network.parameters():
            if len(p.data.shape) > 1:
                nn.init.xavier_uniform_(p.data)

        # configure optimizer
        self.policy_network_optimizer = torch.optim.Adam(self.policy_network.parameters(), lr=self._sl_learning_rate)

    def feed(self, ts):
        self.total_t += 1
        if self.total_t > 0 and self.total_t % self._train_every == 0 and \
                len(self._reservoir_buffer) >= self._min_buffer_size_to_learn:
            sl_loss = self.train()
            print('\rINFO - Step {}, sl-loss: {}'.format(self.total_t, sl_loss), end='')

    def step(self, info_state):
        info_state = np.expand_dims(info_state, axis=0)
        info_state = torch.from_numpy(info_state).float().to(self.device)

        with torch.no_grad():
            log_action_probs = self.policy_network(info_state).cpu().numpy()

        action_probs = np.exp(log_action_probs)[0]

        return action_probs

    def add_transition(self, state, probs):
        transition = Transition(
            info_state=state,
            action_probs=probs)
        self._reservoir_buffer.add(transition)

    def train(self):
        if (len(self._reservoir_buffer) < self._batch_size or
                len(self._reservoir_buffer) < self._min_buffer_size_to_learn):
            return None

        transitions = self._reservoir_buffer.sample(self._batch_size)
        info_states = [t.info_state for t in transitions]
        action_probs = [t.action_probs for t in transitions]

        self.policy_network_optimizer.zero_grad()
        self.policy_network.train()

        # (batch, state_size)
        info_states = torch.from_numpy(np.array(info_states)).float().to(self.device)

        # (batch, num_actions)
        eval_action_probs = torch.from_numpy(np.array(action_probs)).float().to(self.device)

        # (batch, num_actions)
        log_forecast_action_probs = self.policy_network(info_states)

        ce_loss = - (eval_action_probs * log_forecast_action_probs).sum(dim=-1).mean()
        ce_loss.backward()

        self.policy_network_optimizer.step()
        ce_loss = ce_loss.item()
        self.policy_network.eval()

        return ce_loss

    def set_device(self, device):
        self.device = device


class AveragePolicyNetwork(nn.Module):

    def __init__(self, num_actions=2, state_shape=None, mlp_layers=None):
        ''' Initialize the policy network.  It's just a bunch of ReLU
        layers with no activation on the final one, initialized with
        Xavier (sonnet.nets.MLP and tensorflow defaults)
        '''
        super(AveragePolicyNetwork, self).__init__()

        self.num_actions = num_actions
        self.state_shape = state_shape
        self.mlp_layers = mlp_layers

        # set up mlp w/ relu activations
        layer_dims = [np.prod(self.state_shape)] + self.mlp_layers
        mlp = [nn.Flatten()]
        mlp.append(nn.BatchNorm1d(layer_dims[0]))
        for i in range(len(layer_dims) - 1):
            mlp.append(nn.Linear(layer_dims[i], layer_dims[i + 1]))
            if i != len(layer_dims) - 2:  # all but final have relu
                mlp.append(nn.ReLU())
        self.mlp = nn.Sequential(*mlp)

    def forward(self, s):
        logits = self.mlp(s)
        log_action_probs = F.log_softmax(logits, dim=-1)
        return log_action_probs


class ReservoirBuffer(object):

    def __init__(self, reservoir_buffer_capacity):
        self._reservoir_buffer_capacity = reservoir_buffer_capacity
        self._data = []
        self._add_calls = 0

    def add(self, element):
        if len(self._data) < self._reservoir_buffer_capacity:
            self._data.append(element)
        else:
            idx = np.random.randint(0, self._add_calls + 1)
            if idx < self._reservoir_buffer_capacity:
                self._data[idx] = element
        self._add_calls += 1

    def sample(self, num_samples):
        if len(self._data) < num_samples:
            raise ValueError("{} elements could not be sampled from size {}".format(
                num_samples, len(self._data)))
        return random.sample(self._data, num_samples)

    def clear(self):
        self._data = []
        self._add_calls = 0

    def __len__(self):
        return len(self._data)

    def __iter__(self):
        return iter(self._data)
