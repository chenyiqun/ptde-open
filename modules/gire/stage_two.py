from modules.agents.gire_agent import PolicyAppModule
from modules.gire.coach_net import DecCoachNet

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D


class DecCoachNet_TwoStage(DecCoachNet):
    # 由于每个智能体的obs维度不同，所有输入的维度单独列出来，而不是放在args中。
    def __init__(self, args):
        super(DecCoachNet_TwoStage, self).__init__(args)
        self.args = args

    def forward(self, state, obs):

        w1 = self.w1(obs).view(-1, self.args.state_dims, self.args.z_dims)
        b1 = self.b1(obs).view(-1, 1, self.args.z_dims)

        z_hidden = F.elu(torch.matmul(state.unsqueeze(1), w1) + b1).squeeze()

        mu = self.mu(z_hidden)
        sigma = self.sigma(z_hidden)
        sigma = torch.clamp(torch.exp(sigma), min=self.args.var_floor)  # var

        dist = D.Normal(mu.clone(), sigma.clone() ** (1 / 2))

        return dist


class PolicyAppModule_TwoStage(PolicyAppModule):
    def __init__(self, args):
        super(PolicyAppModule_TwoStage, self).__init__(args)
        self.args = args

    def forward(self, inputs):

        z_dot_hidden = F.relu(self.poli_app1(inputs), inplace=True)

        z_dot_mu = self.poli_app2(z_dot_hidden)

        # z_dot_sigma = self.poli_app3(z_dot_hidden)
        # z_dot_sigma = torch.clamp(torch.exp(z_dot_sigma), min=self.args.var_floor)  # var
        # z_dot_sigma = torch.clamp(torch.exp(z_dot_sigma), min=self.args.var_floor, max=self.args.var_floor)  # var

        # dist = D.Normal(z_dot_mu, z_dot_sigma ** (1 / 2))  # 此处1 / 2必须加括号，不然就计算为一次方再除以二。重要！！！！！

        return z_dot_mu