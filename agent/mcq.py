import random
import numpy as np
import torch
import torch.nn as nn
from collections import namedtuple
from copy import deepcopy
from rlcard.utils.utils import remove_illegal

Transition = namedtuple('Transition', ['state', 'action', 'reward', 'next_state', 'legal_actions', 'done', 'target'])


class MCQAgent(object):

    def __init__(self, para, num_actions=2, state_shape=None, device=None):
        self.use_raw = False
        self.replay_memory_init_size = para.replay_memory_init_size
        self.update_target_estimator_every = para.update_target_estimator_every
        self.discount_factor = para.discount_factor
        self.epsilon_decay_steps = para.epsilon_decay_steps
        self.batch_size = para.batch_size
        self.num_actions = num_actions
        self.train_every = para.train_every

        self.device = device

        self.total_t = 0
        self.train_t = 0

        # The epsilon decay scheduler
        self.epsilons = np.linspace(para.epsilon_start, para.epsilon_end, para.epsilon_decay_steps)

        self.q_estimator = Estimator(num_actions=num_actions, learning_rate=para.learning_rate,
                                     state_shape=state_shape, mlp_layers=para.mlp_layers, device=self.device)
        self.target_estimator = Estimator(num_actions=num_actions, learning_rate=para.learning_rate,
                                          state_shape=state_shape, mlp_layers=para.mlp_layers, device=self.device)

        self.memory = Memory(para.replay_memory_size, para.batch_size)

    def feed(self, ts):
        (state, action, reward, next_state, done, target) = tuple(ts)
        self.feed_memory(state['obs'], action, reward, next_state['obs'], list(state['legal_actions'].keys()), done,
                         target)
        self.total_t += 1
        tmp = self.total_t - self.replay_memory_init_size
        if tmp >= 0 and tmp % self.train_every == 0:
            self.train()

    def step(self, state):
        q_values = self.predict(state)
        epsilon = self.epsilons[min(self.total_t, self.epsilon_decay_steps - 1)]
        legal_actions = list(state['legal_actions'].keys())
        probs = np.ones(len(legal_actions), dtype=float) * epsilon / len(legal_actions)
        best_action_idx = legal_actions.index(np.argmax(q_values))
        probs[best_action_idx] += (1.0 - epsilon)
        action_idx = np.random.choice(np.arange(len(probs)), p=probs)

        return legal_actions[action_idx]

    def eval_step(self, state):
        q_values = self.predict(state)
        best_action = np.argmax(q_values)

        info = {}
        info['values'] = {state['raw_legal_actions'][i]: float(q_values[list(state['legal_actions'].keys())[i]]) for i
                          in range(len(state['legal_actions']))}

        return best_action, info

    def predict(self, state):
        q_values = self.q_estimator.predict_nograd(np.expand_dims(state['obs'], 0))[0]
        masked_q_values = -np.inf * np.ones(self.num_actions, dtype=float)
        legal_actions = list(state['legal_actions'].keys())
        masked_q_values[legal_actions] = q_values[legal_actions]

        return masked_q_values

    def train(self):

        state_batch, action_batch, reward_batch, next_state_batch, legal_actions_batch, done_batch, target_batch = self.memory.sample()

        # # Calculate best next actions using Q-network (Double DQN)
        # q_values_next = self.q_estimator.predict_nograd(next_state_batch)
        # legal_actions = []
        # for b in range(self.batch_size):
        #     legal_actions.extend([i + b * self.num_actions for i in legal_actions_batch[b]])
        # masked_q_values = -np.inf * np.ones(self.num_actions * self.batch_size, dtype=float)
        # masked_q_values[legal_actions] = q_values_next.flatten()[legal_actions]
        # masked_q_values = masked_q_values.reshape((self.batch_size, self.num_actions))
        # best_actions = np.argmax(masked_q_values, axis=1)
        #
        # # Evaluate best next actions using Target-network (Double DQN)
        # q_values_next_target = self.target_estimator.predict_nograd(next_state_batch)
        # target_batch = reward_batch + np.invert(done_batch).astype(np.float32) * \
        #                self.discount_factor * q_values_next_target[np.arange(self.batch_size), best_actions]

        # Perform gradient descent update
        state_batch = np.array(state_batch)

        loss = self.q_estimator.update(state_batch, action_batch, target_batch)
        print('\rINFO - Step {}, rl-loss: {}'.format(self.total_t, loss), end='')

        # Update the target estimator
        if self.train_t % self.update_target_estimator_every == 0:
            self.target_estimator = deepcopy(self.q_estimator)
            print("\nINFO - Copied model parameters to target network.")

        self.train_t += 1

    def feed_memory(self, state, action, reward, next_state, legal_actions, done, target):
        self.memory.save(state, action, reward, next_state, legal_actions, done, target)

    def set_device(self, device):
        self.device = device
        self.q_estimator.device = device
        self.target_estimator.device = device


class Estimator(object):
    '''
    Approximate clone of rlcard.agents.dqn_agent.Estimator that
    uses PyTorch instead of Tensorflow.  All methods input/output np.ndarray.

    Q-Value Estimator neural network.
    This network is used for both the Q-Network and the Target Network.
    '''

    def __init__(self, num_actions=2, learning_rate=0.001, state_shape=None, mlp_layers=None, device=None):

        self.num_actions = num_actions
        self.learning_rate = learning_rate
        self.state_shape = state_shape
        self.mlp_layers = mlp_layers
        self.device = device

        # set up Q model and place it in eval mode
        qnet = EstimatorNetwork(num_actions, state_shape, mlp_layers)
        qnet = qnet.to(self.device)
        self.qnet = qnet
        self.qnet.eval()

        # initialize the weights using Xavier init
        for p in self.qnet.parameters():
            if len(p.data.shape) > 1:
                nn.init.xavier_uniform_(p.data)

        # set up loss function
        self.mse_loss = nn.MSELoss(reduction='mean')
        rms_optimizer = torch.optim.RMSprop(
            qnet.parameters(),
            lr=0.0001,
            momentum=0,
            eps=1e-5,
            alpha=0.99)
        adam_optimizer = torch.optim.Adam(self.qnet.parameters(), lr=self.learning_rate)
        # set up optimizer
        self.optimizer = rms_optimizer

    def predict_nograd(self, s):
        ''' Predicts action values, but prediction is not included
            in the computation graph.  It is used to predict optimal next
            actions in the Double-DQN algorithm.

        Args:
          s (np.ndarray): (batch, state_len)

        Returns:
          np.ndarray of shape (batch_size, NUM_VALID_ACTIONS) containing the estimated
          action values.
        '''
        with torch.no_grad():
            s = torch.from_numpy(s).float().to(self.device)
            q_as = self.qnet(s).cpu().numpy()
        return q_as

    def update(self, s, a, y):
        ''' Updates the estimator towards the given targets.
            In this case y is the target-network estimated
            value of the Q-network optimal actions, which
            is labeled y in Algorithm 1 of Minh et al. (2015)

        Args:
          s (np.ndarray): (batch, state_shape) state representation
          a (np.ndarray): (batch,) integer sampled actions
          y (np.ndarray): (batch,) value of optimal actions according to Q-target

        Returns:
          The calculated loss on the batch.
        '''
        self.optimizer.zero_grad()

        self.qnet.train()

        s = torch.from_numpy(s).float().to(self.device)
        a = torch.from_numpy(a).long().to(self.device)
        y = torch.from_numpy(y).float().to(self.device)

        # (batch, state_shape) -> (batch, num_actions)
        q_as = self.qnet(s)

        # (batch, num_actions) -> (batch, )
        Q = torch.gather(q_as, dim=-1, index=a.unsqueeze(-1)).squeeze(-1)

        # update model
        batch_loss = self.mse_loss(Q, y)
        batch_loss.backward()
        self.optimizer.step()
        batch_loss = batch_loss.item()

        self.qnet.eval()

        return batch_loss


class EstimatorNetwork(nn.Module):

    def __init__(self, num_actions=2, state_shape=None, mlp_layers=None):
        super(EstimatorNetwork, self).__init__()

        self.num_actions = num_actions
        self.state_shape = state_shape
        self.mlp_layers = mlp_layers

        # build the Q network
        layer_dims = [np.prod(self.state_shape)] + self.mlp_layers
        va = [nn.Flatten()]
        va.append(nn.BatchNorm1d(layer_dims[0]))
        for i in range(len(layer_dims) - 1):
            va.append(nn.Linear(layer_dims[i], layer_dims[i + 1], bias=True))
            va.append(nn.Tanh())
        va.append(nn.Linear(layer_dims[-1], 1, bias=True))

        ad = [nn.Flatten()]
        ad.append(nn.BatchNorm1d(layer_dims[0]))
        for i in range(len(layer_dims) - 1):
            ad.append(nn.Linear(layer_dims[i], layer_dims[i + 1], bias=True))
            ad.append(nn.Tanh())
        ad.append(nn.Linear(layer_dims[-1], self.num_actions, bias=True))

        self.ad_layers = nn.Sequential(*va)
        self.va_layers = nn.Sequential(*ad)

    def forward(self, s):
        va = self.va_layers(s)
        ad = self.ad_layers(s)
        return va + ad - ad.mean(dim=1, keepdim=True)


class Memory(object):

    def __init__(self, memory_size, batch_size):
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.memory = []
        self.unroll_length=100

    def save(self, state, action, reward, next_state, legal_actions, done, target):
        if len(self.memory) == self.memory_size:
            self.memory.pop(0)
        transition = Transition(state, action, reward, next_state, legal_actions, done, target)
        self.memory.append(transition)

    def sample(self):
        # samples = random.sample(self.memory, self.batch_size)
        # samples = self.memory[0: self.batch_size - 1]
        # self.memory = self.memory[self.batch_size:-1]
        samples=self.memory[:]
        self.memory=[]
        return map(np.array, zip(*samples))
