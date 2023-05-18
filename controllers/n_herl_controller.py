from modules.agents import REGISTRY as agent_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY
from .basic_herl_controller import BasicHerlMAC
import torch
from utils.rl_utils import RunningMeanStd
import numpy as np


# This multi-agent controller shares parameters between agents
class NHERLMAC(BasicHerlMAC):
    def __init__(self, scheme, groups, args):
        super(NHERLMAC, self).__init__(scheme, groups, args)

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        # Only select actions for the selected batch elements in bs
        avail_actions = ep_batch["avail_actions"][:, t_ep]

        qvals, Z_dot_dist = self.forward(ep_batch, t_ep, test_mode=test_mode)

        chosen_actions = self.action_selector.select_action(qvals[bs], avail_actions[bs], t_env, test_mode=test_mode)

        return chosen_actions, Z_dot_dist

    def forward(self, ep_batch, t, test_mode=False):
        agent_inputs = self._build_inputs(ep_batch, t)
        avail_actions = ep_batch["avail_actions"][:, t]

        if test_mode:
            self.agent.eval()
            self.policy_app.eval()

        Z_dot, Z_dot_dist = self.policy_app.forward(agent_inputs)
        agent_outs, self.hidden_states = self.agent(agent_inputs, self.hidden_states, Z_dot)

        return agent_outs.view(ep_batch.batch_size, self.n_agents, -1), Z_dot_dist