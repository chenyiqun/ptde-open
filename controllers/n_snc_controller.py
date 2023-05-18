from .basic_snc_controller import BasicSNCMAC
import random


# This multi-agent controller shares parameters between agents
class NSNCMAC(BasicSNCMAC):
    def __init__(self, scheme, groups, args):
        super(NSNCMAC, self).__init__(scheme, groups, args)

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        # Only select actions for the selected batch elements in bs
        avail_actions = ep_batch["avail_actions"][:, t_ep]

        if self.args.name == 'snc_env=8_adam_td_lambda':  # decentralized way z'
            if test_mode:  # testing
                # print(
                #     '_______________________________________________________________________________________________, ',
                #     t_ep)
                qvals, _, _ = self.forward(ep_batch, t_ep, test_mode=test_mode)
            else:  # training
                qvals, _, _ = self.forward(ep_batch, t_ep, test_mode=test_mode)

        else:
            qvals = self.forward(ep_batch, t_ep, test_mode=test_mode)

        chosen_actions = self.action_selector.select_action(qvals[bs], avail_actions[bs], t_env, test_mode=test_mode)

        return chosen_actions

    def forward(self, ep_batch, t, test_mode=False):
        agent_inputs = self._build_inputs(ep_batch, t)
        avail_actions = ep_batch["avail_actions"][:, t]

        if test_mode:
            self.shared_rnn.eval()
            self.agent.eval()
            self.coach_net.eval()

        self.hidden_states = self.shared_rnn(agent_inputs, self.hidden_states)

        if self.args.name == 'gire_env=8_adam_td_lambda':  # decentralized way z'

            if test_mode:  # testing
                # Z_dot, _ = self.policy_app.forward(self.hidden_states)  # 1. o or tao  2. clone() or not

                # Z_dot_dist, _ = self.policy_app.forward(ep_batch, agent_inputs, t=t, return_one=False)  # test
                # agent_outs = self.agent(self.hidden_states, Z_dot_dist.rsample())

                Z_dot, _ = self.policy_app.forward(agent_inputs, test_mode=test_mode)  # 1. o or tao  2. clone() or not
                agent_outs = self.agent(self.hidden_states, Z_dot)

                return agent_outs.view(ep_batch.batch_size, self.n_agents, -1)

            else:  # training
                b, a, _ = agent_inputs.size()
                Z_dist, Z_wog_dist = self.coach_net.forward(ep_batch, agent_inputs, t=t, return_one=False)
                # Z_dot, Z_dot_dist = self.policy_app.forward(self.hidden_states)  # 1. o or tao  2. clone() or not

                # Z_dot_dist, _ = self.policy_app.forward(ep_batch, agent_inputs, t=t, return_one=False)  # test
                # agent_outs = self.agent(self.hidden_states, Z_dist.rsample().view(b, a, -1))

                Z_dot, Z_dot_dist = self.policy_app.forward(agent_inputs)  # 1. o or tao  2. clone() or not
                agent_outs = self.agent(self.hidden_states, Z_dot.view(b, a, -1))

                return agent_outs.view(ep_batch.batch_size, self.n_agents, -1), Z_dist, Z_wog_dist, Z_dot_dist  # z z(no g) z'

        elif self.args.name == 'gire_s_env=8_adam_td_lambda' or self.args.name == 'gire_s_vdn_env=8_adam_td_lambda':  # centralized way state

            s_dist = self.coach_net.forward(ep_batch, agent_inputs, t=t)

            agent_outs = self.agent(self.hidden_states, s_dist.rsample())

            return agent_outs.view(ep_batch.batch_size, self.n_agents, -1)

        elif self.args.name == 'gire_z_env=8_adam_td_lambda' or self.args.name == 'gire_z_vdn_env=8_adam_td_lambda':  # centralized way z

            z_dist = self.coach_net.forward(ep_batch, agent_inputs, t=t, return_one=True)

            agent_outs = self.agent(self.hidden_states, z_dist.rsample())

            return agent_outs.view(ep_batch.batch_size, self.n_agents, -1)

        elif self.args.name == 'csrl_env=8_adam_td_lambda':

            zit = self.coach_net.forward(ep_batch, agent_inputs, t=t)

            agent_outs = self.agent(self.hidden_states, zit)

            return agent_outs.view(ep_batch.batch_size, self.n_agents, -1)

        elif self.args.name == 'copa_env=8_adam_td_lambda':  # centralized way z

            z_dist = self.coach_net.forward(ep_batch, agent_inputs, t=t)

            agent_outs = self.agent(self.hidden_states, z_dist.rsample())

            return agent_outs.view(ep_batch.batch_size, self.n_agents, -1)

        elif self.args.name == 'snc_env=8_adam_td_lambda':  # centralized way z

            # if not test_mode:
            #     print('step:___________________________________________________________________________', t_ep)

            z_dist = self.coach_net.forward(ep_batch, agent_inputs, t=t, return_one=True)

            agent_outs, sn_i, c_dist = self.agent(self.hidden_states, z_dist.rsample(), agent_inputs)

            return agent_outs.view(ep_batch.batch_size, self.n_agents, -1), sn_i, c_dist