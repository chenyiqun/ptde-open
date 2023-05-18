from modules.agents import REGISTRY as agent_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY
import torch as th

from utils.th_utils import get_parameters_num
from modules.agents.gire_agent import PolicyAppModule, SharedRNN, COPAAgent
from modules.gire.coach_net import DecCoachNet, DecCoachNet2, AppDecCoachNet, StateCoachNet, CSRLCoachNet, COPACoachNet
from modules.snc.cog import CogModule

# This multi-agent controller shares parameters between agents
class BasicSNCMAC:
    def __init__(self, scheme, groups, args):
        self.n_agents = args.n_agents
        self.args = args
        input_shape = self._get_input_shape(scheme)
        self.args.obs_input_dims = self._get_inputs_dims(scheme)
        self.args.state_dims = self._get_state_shape(scheme)
        self._build_agents(input_shape)
        self.agent_output_type = args.agent_output_type

        self.action_selector = action_REGISTRY[args.action_selector](args)
        self.save_probs = getattr(self.args, 'save_probs', False)

        self.hidden_states = None

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        # Only select actions for the selected batch elements in bs
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        agent_outputs, Z_dot = self.forward(ep_batch, t_ep, test_mode=test_mode)
        chosen_actions = self.action_selector.select_action(agent_outputs[bs], avail_actions[bs], t_env,
                                                            test_mode=test_mode)
        return chosen_actions, Z_dot

    def forward(self, ep_batch, t, test_mode=False):
        agent_inputs = self._build_inputs(ep_batch, t)
        avail_actions = ep_batch["avail_actions"][:, t]

        if test_mode:
            self.agent.eval()

        # h_out = self.shared_rnn(agent_inputs, self.hidden_states)
        agent_outs, self.hidden_states, Z_dot = self.agent(agent_inputs, self.hidden_states)

        # Softmax the agent outputs if they're policy logits
        if self.agent_output_type == "pi_logits":

            if getattr(self.args, "mask_before_softmax", True):
                # Make the logits for unavailable actions very negative to minimise their affect on the softmax
                agent_outs = agent_outs.reshape(ep_batch.batch_size * self.n_agents, -1)
                reshaped_avail_actions = avail_actions.reshape(ep_batch.batch_size * self.n_agents, -1)
                agent_outs[reshaped_avail_actions == 0] = -1e10

            agent_outs = th.nn.functional.softmax(agent_outs, dim=-1)

        return agent_outs.view(ep_batch.batch_size, self.n_agents, -1), Z_dot

    def init_hidden(self, batch_size):
        self.hidden_states = self.shared_rnn.init_hidden()
        if self.hidden_states is not None:
            self.hidden_states = self.hidden_states.unsqueeze(0).expand(batch_size, self.n_agents, -1)  # bav

    def parameters(self):
        if self.args.name == 'gire_env=8_adam_td_lambda':  # decentralized

            return list(self.shared_rnn.parameters()) + list(self.agent.parameters()) + \
                   list(self.policy_app.parameters()) + list(self.coach_net.parameters())

        elif self.args.name == 'snc_env=8_adam_td_lambda':

            return list(self.shared_rnn.parameters()) + list(self.agent.parameters()) + \
                   list(self.coach_net.parameters())

        else:  # centralized
            return list(self.shared_rnn.parameters()) + list(self.agent.parameters()) + \
                   list(self.coach_net.parameters())

    def get_parameters(self):
        return list(self.shared_rnn.parameters()) + list(self.agent.parameters()) + list(self.coach_net.parameters())

    def load_state(self, other_mac):
        self.shared_rnn.load_state_dict(other_mac.shared_rnn.state_dict())
        self.agent.load_state_dict(other_mac.agent.state_dict())
        self.coach_net.load_state_dict(other_mac.coach_net.state_dict())
        if self.args.name == 'gire_env=8_adam_td_lambda':  # decentralized
            self.policy_app.load_state_dict(other_mac.policy_app.state_dict())

    def cuda(self):
        self.shared_rnn.cuda()
        self.agent.cuda()
        self.coach_net.cuda()
        if self.args.name == 'gire_env=8_adam_td_lambda':  # decentralized
            self.policy_app.cuda()

    def save_models(self, path):
        th.save(self.shared_rnn.state_dict(), "{}/shared_rnn.th".format(path))
        th.save(self.agent.state_dict(), "{}/agent.th".format(path))
        th.save(self.coach_net.state_dict(), "{}/coach_net.th".format(path))
        if self.args.name == 'gire_env=8_adam_td_lambda':  # decentralized
            th.save(self.policy_app.state_dict(), "{}/policy_app.th".format(path))

    def load_models(self, path):
        self.shared_rnn.load_state_dict(th.load("{}/shared_rnn.th".format(path), map_location=lambda storage, loc: storage))
        self.agent.load_state_dict(th.load("{}/agent.th".format(path), map_location=lambda storage, loc: storage))
        self.coach_net.load_state_dict(th.load("{}/coach_net.th".format(path), map_location=lambda storage, loc: storage))
        if self.args.name == 'gire_env=8_adam_td_lambda':  # decentralized
            self.policy_app.load_state_dict(th.load("{}/policy_app_500000.th".format(path), map_location=lambda storage, loc: storage))

    def _build_agents(self, input_shape):

        self.shared_rnn = SharedRNN(input_shape, self.args)
        print('Shared rnn module Size: ')
        print(get_parameters_num(self.shared_rnn.parameters()))

        self.agent = agent_REGISTRY[self.args.agent](self.args)
        print('SNC agent module Size: ')
        print(get_parameters_num(self.agent.parameters()))

        if self.args.name == 'gire_env=8_adam_td_lambda' \
                or self.args.name == 'gire_z_env=8_adam_td_lambda' \
                or self.args.name == 'gire_z_vdn_env=8_adam_td_lambda' \
                or self.args.name == 'snc_env=8_adam_td_lambda':  # z
            self.coach_net = DecCoachNet(self.args)
        elif self.args.name == 'gire_s_env=8_adam_td_lambda' \
                or self.args.name == 'gire_s_vdn_env=8_adam_td_lambda':  # s
            self.coach_net = StateCoachNet(self.args)
        elif self.args.name == 'csrl_env=8_adam_td_lambda':
            self.coach_net = CSRLCoachNet(self.args)
        elif self.args.name == 'copa_env=8_adam_td_lambda':
            self.coach_net = COPACoachNet(self.args)
        print('Coach net module Size: ')
        print(get_parameters_num(self.coach_net.parameters()))

        # if self.args.name == 'snc_env=8_adam_td_lambda':
        #     self.cog = CogModule(self.args)
        #     print('Cognition module Size: ')
        #     print(get_parameters_num(self.cog.parameters()))

        if self.args.name == 'gire_env=8_adam_td_lambda':  # decentralized
            self.policy_app = PolicyAppModule(self.args)
            print('Policy approximate module Size: ')
            print(get_parameters_num(self.policy_app.parameters()))

    def _build_inputs(self, batch, t):
        # Assumes homogenous agents with flat observations.
        # Other MACs might want to e.g. delegate building inputs to each agent
        bs = batch.batch_size
        inputs = []
        inputs.append(batch["obs"][:, t])  # b1av
        if self.args.obs_last_action:
            if t == 0:
                inputs.append(th.zeros_like(batch["actions_onehot"][:, t]))
            else:
                inputs.append(batch["actions_onehot"][:, t - 1])
        if self.args.obs_agent_id:
            inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1))

        inputs = th.cat([x.reshape(bs, self.n_agents, -1) for x in inputs], dim=-1)
        return inputs

    def _get_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"]
        if self.args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0]
        if self.args.obs_agent_id:
            input_shape += self.n_agents

        return input_shape

    def _get_inputs_dims(self, scheme):  # add
        obs_dims, action_dims = scheme['obs']['vshape'], scheme['avail_actions']['vshape'][0]
        obs_input_dims = obs_dims
        if self.args.obs_last_action:
            obs_input_dims += action_dims
        if self.args.obs_agent_id:
            obs_input_dims += self.n_agents

        return obs_input_dims

    def _get_state_shape(self, scheme):

        return scheme["state"]["vshape"]
