import copy
from components.episode_buffer import EpisodeBatch
from modules.mixers.nmix import Mixer
from envs.matrix_game import print_matrix_status
from utils.rl_utils import build_td_lambda_targets, build_q_lambda_targets
import torch as th
from torch.optim import RMSprop, Adam
import numpy as np
from utils.th_utils import get_parameters_num

import torch.nn as nn

from torch.distributions import kl_divergence
import torch.distributions as D


class NGireQLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.logger = logger
        self.n_agents = args.n_agents

        self.last_target_update_episode = 0
        self.device = th.device('cuda' if args.use_cuda else 'cpu')
        self.params = list(mac.parameters())

        # mixer
        if args.mixer == "qmix":
            self.mixer = Mixer(args)
        else:
            raise "mixer error"
        self.target_mixer = copy.deepcopy(self.mixer)
        self.params += list(self.mixer.parameters())

        print('Mixer Size: ')
        print(get_parameters_num(self.mixer.parameters()))

        self.obs_input_dims = self._get_inputs_dims(scheme)  # add
        self.args.obs_input_dims = self.obs_input_dims
        self.state_dims = self._get_state_shape(scheme)
        self.args.state_dims = self.state_dims
        self.n_agents = args.n_agents

        if self.args.optimizer == 'adam':
            self.optimiser = Adam(params=self.params, lr=args.lr, weight_decay=getattr(args, "weight_decay", 0))
        else:
            self.optimiser = RMSprop(params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)

        # if self.args.name == 'gire_env=8_adam_td_lambda':  # decentralized
        #     self.kl_params = [
        #         {"params": self.mac.policy_app.parameters(), "lr": args.lr},
        #         {"params": self.mac.coach_net.parameters(), "lr": args.lr / 100},
        #     ]
        #     self.optimiser2 = Adam(params=self.kl_params, weight_decay=getattr(args, "weight_decay", 0))
        # else:
        #     self.kl_params = [
        #         {"params": self.mac.coach_net.parameters(), "lr": args.lr},
        #     ]
        #     self.optimiser2 = Adam(params=self.kl_params, weight_decay=getattr(args, "weight_decay", 0))

        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        self.target_mac = copy.deepcopy(mac)
        self.log_stats_t = -self.args.learner_log_interval - 1
        self.train_t = 0

        # priority replay
        self.use_per = getattr(self.args, 'use_per', False)
        self.return_priority = getattr(self.args, "return_priority", False)
        if self.use_per:
            self.priority_max = float('-inf')
            self.priority_min = float('inf')

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int, per_weight=None):
        # Get the relevant quantities
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]

        # Calculate estimated Q-Values
        self.mac.shared_rnn.train()
        self.mac.agent.train()
        self.mac.coach_net.train()
        if self.args.name == 'gire_env=8_adam_td_lambda':  # decentralized
            self.mac.policy_app.train()

        mac_out = []

        if self.args.name == 'gire_env=8_adam_td_lambda':  # decentralized
            Z_dot = []
            Z_wog = []
            Z = []

        self.mac.init_hidden(batch.batch_size)

        for t in range(batch.max_seq_length):

            if self.args.name == 'gire_env=8_adam_td_lambda':  # decentralized way z'

                # z z(no g) z'
                agent_outs, z_dist, z_wog_dist, z_dot_dist = self.mac.forward(batch, t=t, test_mode=False)  # mac

                mac_out.append(agent_outs)
                Z_wog.append(z_wog_dist)
                Z_dot.append(z_dot_dist)
                Z.append(z_dist)

            else:
                agent_outs = self.mac.forward(batch, t=t, test_mode=False)

                mac_out.append(agent_outs)

        mac_out = th.stack(mac_out, dim=1)  # Concat over time

        # Pick the Q-Values for the actions taken by each agent
        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the last dim
        chosen_action_qvals_ = chosen_action_qvals

        # Calculate the Q-Values necessary for the target
        with th.no_grad():
            self.target_mac.agent.train()
            target_mac_out = []
            self.target_mac.init_hidden(batch.batch_size)
            for t in range(batch.max_seq_length):

                if self.args.name == 'gire_env=8_adam_td_lambda':  # decentralized way z'

                    target_agent_outs, _, _, _ = self.target_mac.forward(batch, t=t)
                    target_mac_out.append(target_agent_outs)

                else:
                    target_agent_outs = self.target_mac.forward(batch, t=t)
                    target_mac_out.append(target_agent_outs)

            # We don't need the first timesteps Q-Value estimate for calculating targets
            target_mac_out = th.stack(target_mac_out, dim=1)  # Concat across time

            # Max over target Q-Values/ Double q learning
            mac_out_detach = mac_out.clone().detach()
            mac_out_detach[avail_actions == 0] = -9999999
            cur_max_actions = mac_out_detach.max(dim=3, keepdim=True)[1]
            target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)

            # Calculate n-step Q-Learning targets
            target_max_qvals = self.target_mixer(target_max_qvals, batch["state"])

            if getattr(self.args, 'q_lambda', False):
                qvals = th.gather(target_mac_out, 3, batch["actions"]).squeeze(3)
                qvals = self.target_mixer(qvals, batch["state"])

                targets = build_q_lambda_targets(rewards, terminated, mask, target_max_qvals, qvals,
                                                 self.args.gamma, self.args.td_lambda)
            else:
                targets = build_td_lambda_targets(rewards, terminated, mask, target_max_qvals,
                                                  self.args.n_agents, self.args.gamma, self.args.td_lambda)

        # Mixer
        chosen_action_qvals = self.mixer(chosen_action_qvals, batch["state"][:, :-1])

        td_error = (chosen_action_qvals - targets.detach())
        td_error2 = 0.5 * td_error.pow(2)

        mask = mask.expand_as(td_error2)
        masked_td_error = td_error2 * mask

        # important sampling for PER
        if self.use_per:
            per_weight = th.from_numpy(per_weight).unsqueeze(-1).to(device=self.device)
            masked_td_error = masked_td_error.sum(1) * per_weight

        # loss function
        loss = 0

        # td-loss
        td_loss = masked_td_error.sum() / mask.sum()
        loss += td_loss

        # # kl-loss
        # if self.args.name == 'gire_env=8_adam_td_lambda':  # decentralized way z'
        #
        #     Z_ZD_kl_loss = []
        #     Z_dot_sigma_loss = []
        #     Z_sigma_loss = []
        #     MSE_loss = []
        #     for i in range(len(Z_wog)):
        #         # Z_ZD_kl_loss.append(kl_divergence(Z_wog[i], Z_dot[i]).sum(dim=-1).mean())
        #         # Z_ZD_kl_loss.append(kl_divergence(Z[i], Z_dot[i]).sum(dim=-1).mean())
        #
        #         # Z_dot_sigma_loss.append(Z_dot[i].scale.sum(dim=-1).mean())
        #         # Z_sigma_loss.append(Z[i].scale.sum(dim=-1).mean())
        #         MSE_loss.append((Z_wog[i].loc-Z_dot[i].loc).sum(dim=-1).mean()**2)
        #     zzd_loss = sum(Z_ZD_kl_loss) / len(Z_ZD_kl_loss)
        #     # z_dot_sigma_loss = sum(Z_dot_sigma_loss) / len(Z_dot_sigma_loss)
        #     # z_sigma_loss = sum(Z_sigma_loss) / len(Z_sigma_loss)
        #     # mse_loss = sum(MSE_loss) / len(MSE_loss)
        #
        #     # loss += 0.1 * (zzd_loss + 5 * z_dot_sigma_loss + z_sigma_loss)
        #     # loss += 0.01 * (z_dot_sigma_loss + z_sigma_loss + 100 * mse_loss)
        #     # loss += 0.1 * (z_dot_sigma_loss + mse_loss)
        #     # loss += mse_loss
        #     # loss += 0.01 * (zzd_loss + z_dot_sigma_loss + mse_loss)
        #     loss += 0.01 * zzd_loss
        #
        #     # # Optimise
        #     # self.optimiser2.zero_grad()
        #     # zzd_loss.backward()
        #     # # grad_norm2 = th.nn.utils.clip_grad_norm_(self.kl_params, self.args.grad_norm_clip)
        #     # self.optimiser2.step()

        # Optimise
        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()

        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss_td", td_loss.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm, t_env)

            # if self.args.name == 'gire_env=8_adam_td_lambda':  # decentralized way z'
                # self.logger.log_stat("grad_norm2", grad_norm2, t_env)

            mask_elems = mask.sum().item()
            self.logger.log_stat("td_error_abs", (masked_td_error.abs().sum().item() / mask_elems), t_env)
            self.logger.log_stat("q_taken_mean",
                                 (chosen_action_qvals * mask).sum().item() / (mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat("target_mean", (targets * mask).sum().item() / (mask_elems * self.args.n_agents),
                                 t_env)

            if self.args.name == 'gire_env=8_adam_td_lambda':
                # self.logger.log_stat("kl_loss", zzd_loss, t_env)
                # self.logger.log_stat("z_dot_sigma_loss", z_dot_sigma_loss, t_env)
                # self.logger.log_stat("z_sigma_loss", z_sigma_loss, t_env)
                # self.logger.log_stat("mse_loss", mse_loss, t_env)
                pass

            self.log_stats_t = t_env

            print('cuda number: {}, env name: {}, algorithm: {}, seed: {}'.format(self.args.cuda_num,
                                                                                  self.args.env_args['map_name'],
                                                                                  self.args.name,
                                                                                  self.args.seed))

            print('\t')

            # print estimated matrix
            if self.args.env == "one_step_matrix_game":
                print_matrix_status(batch, self.mixer, mac_out)

        # return info
        info = {}
        # calculate priority
        if self.use_per:
            if self.return_priority:
                info["td_errors_abs"] = rewards.sum(1).detach().to('cpu')
                # normalize to [0, 1]
                self.priority_max = max(th.max(info["td_errors_abs"]).item(), self.priority_max)
                self.priority_min = min(th.min(info["td_errors_abs"]).item(), self.priority_min)
                info["td_errors_abs"] = (info["td_errors_abs"] - self.priority_min) \
                                        / (self.priority_max - self.priority_min + 1e-5)
            else:
                info["td_errors_abs"] = ((td_error.abs() * mask).sum(1) \
                                         / th.sqrt(mask.sum(1))).detach().to('cpu')
        return info

    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.logger.console_logger.info("Updated target network")

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()
        if self.mixer is not None:
            self.mixer.cuda()
            self.target_mixer.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))

    def load_models(self, path):
        # may be barely used
        self.mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
        # self.optimiser.load_state_dict(th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))

    def _get_inputs_dims(self, scheme):  # add
        obs_dims, action_dims = scheme['obs']['vshape'], scheme['avail_actions']['vshape'][0]
        obs_input_dims = obs_dims
        if self.args.obs_last_action:
            obs_input_dims += action_dims
        if self.args.obs_agent_id:
            obs_input_dims += self.n_agents

        return obs_input_dims

    def _get_state_shape(self, scheme):  # add

        return scheme["state"]["vshape"]

    def _build_inputs(self, batch, t):  # add
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

    # @register_kl(Normal, Normal)
    # def _kl_normal_normal(p, q):
    #     var_ratio = (p.scale / q.scale).pow(2)
    #     t1 = ((p.loc - q.loc) / q.scale).pow(2)
    #     return 0.5 * (var_ratio + t1 - 1 - var_ratio.log())

    def get_kl_loss(self, p_mu, q_mu, p_sigma, q_sigma):
        var_ratio = (p_sigma / q_sigma).pow(2)
        t1 = ((p_mu - q_mu) / q_sigma).pow(2)

        return 0.5 * (var_ratio + t1 - 1 - var_ratio.log())