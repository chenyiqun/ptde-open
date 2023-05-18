import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D

from modules.snc.cog import CogModule


class SNCAgent(nn.Module):
    def __init__(self, args):
        super(SNCAgent, self).__init__()
        self.args = args

        self.fc1 = nn.Linear(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim + args.z_dims + args.c_dims, args.n_actions)
        self.cog = CogModule(self.args)

    def forward(self, input_hidden_out, z_dot, input_obs):
        b, a, e = input_hidden_out.size()

        c, c_dist, sn_i = self.cog.forward(input_obs, z_dot)

        h_oi = F.relu(self.fc1(input_hidden_out.view(-1, e)), inplace=True)

        q = self.fc2(torch.concat([h_oi, z_dot.view(b * a, -1), c.view(b * a, -1)], dim=-1))

        return q.view(b, a, -1), sn_i, c_dist


class SharedRNN(nn.Module):
    def __init__(self, obs_input_dims, args):
        super(SharedRNN, self).__init__()
        self.args = args

        self.fc1 = nn.Linear(obs_input_dims, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state):
        b, a, e = inputs.size()

        x = F.relu(self.fc1(inputs.view(-1, e)), inplace=True)
        h = self.rnn(x, hidden_state.view(b*a, -1))

        return h.view(b, a, -1)


class PolicyAppModule(nn.Module):
    def __init__(self, args):
        super(PolicyAppModule, self).__init__()
        self.args = args

        self.poli_app1 = nn.Linear(args.obs_input_dims, args.rnn_hidden_dim)  # args.obs_input_dims

        self.poli_app2 = nn.Sequential(nn.Linear(args.rnn_hidden_dim, args.z_dims),
                                       # nn.BatchNorm1d(args.z_dims),
                                       nn.LeakyReLU(),
                                       nn.Linear(args.z_dims, args.z_dims))

        self.poli_app3 = nn.Sequential(nn.Linear(args.rnn_hidden_dim, args.z_dims),
                                       # nn.BatchNorm1d(args.z_dims),
                                       nn.LeakyReLU(),
                                       nn.Linear(args.z_dims, args.z_dims))

    def forward(self, inputs, test_mode=False):
        b, a, e = inputs.size()

        z_dot_hidden = F.relu(self.poli_app1(inputs.view(-1, e)), inplace=True)
        z_dot_mu = self.poli_app2(z_dot_hidden)
        z_dot_sigma = self.poli_app3(z_dot_hidden)

        if test_mode:
            z_dot_sigma = torch.clamp(torch.exp(z_dot_sigma), min=self.args.var_floor, max=self.args.var_floor)  # var
        else:
            z_dot_sigma = torch.clamp(torch.exp(z_dot_sigma), min=self.args.var_floor)  # var
            # z_dot_sigma = torch.exp(z_dot_sigma)  # var

        dist = D.Normal(z_dot_mu, z_dot_sigma ** (1 / 2))  # 此处1 / 2必须加括号，不然就计算为一次方再除以二。重要！！！！！
        z_dot = dist.rsample()

        return z_dot.view(b, a, -1), dist
