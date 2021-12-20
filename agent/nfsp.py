import numpy as np
from rlcard.utils.utils import remove_illegal

from agent.dqn import DQNAgent
from agent.dueldqn import DuelDQNAgent
from agent.sl import SLAgent


class NFSPAgent(object):

    def __init__(self, args, num_actions=4, state_shape=None, device=None):
        self.use_raw = False
        self._num_actions = num_actions
        self._state_shape = state_shape
        self.device = device

        self.hyper_para = args.getpara('hyperparameters')
        self.evaluate_with = self.hyper_para.evaluate_with
        self._anticipatory_param = self.hyper_para.anticipatory_param

        self._sl_agent = SLAgent(args.getpara('sl'), num_actions=self._num_actions, state_shape=self._state_shape,
                                    device=self.device)
        self._rl_agent = DuelDQNAgent(args.getpara('rl'), num_actions=self._num_actions, state_shape=self._state_shape,
                                      device=self.device)

    def feed(self, ts):
        self._rl_agent.feed(ts)
        self._sl_agent.feed(ts)

    def step(self, state):

        self.sample_episode_policy()

        obs = state['obs']
        legal_actions = list(state['legal_actions'].keys())
        if self._mode == 'best_response':
            action = self._rl_agent.step(state)
            one_hot = np.zeros(self._num_actions)
            one_hot[action] = 1
            self._sl_agent.add_transition(obs, one_hot)

        elif self._mode == 'average_policy':
            probs = self._sl_agent.step(obs)
            probs = remove_illegal(probs, legal_actions)
            action = np.random.choice(len(probs), p=probs)

        return action

    def eval_step(self, state):
        if self.evaluate_with == 'best_response':
            action, info = self._rl_agent.eval_step(state)
        elif self.evaluate_with == 'average_policy':
            obs = state['obs']
            legal_actions = list(state['legal_actions'].keys())
            probs = self._sl_agent.step(obs)
            probs = remove_illegal(probs, legal_actions)
            action = np.random.choice(len(probs), p=probs)
            info = {}
            info['probs'] = {state['raw_legal_actions'][i]: float(probs[list(state['legal_actions'].keys())[i]]) for
                             i
                             in range(len(state['legal_actions']))}
        else:
            raise ValueError("'evaluate_with' should be either 'average_policy' or 'best_response'.")
        return action, info

    def sample_episode_policy(self):
        if np.random.rand() < self._anticipatory_param:
            self._mode = 'best_response'
        else:
            self._mode = 'average_policy'

    def set_device(self, device):
        self._sl_agent.set_device(device)
        self._rl_agent.set_device(device)
