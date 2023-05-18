import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D


class CogModule(nn.Module):
    def __init__(self, args):
        super(CogModule, self).__init__()
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

        self.SN_module = nn.Sequential(nn.Linear(args.z_dims, args.z_dims),
                                       nn.LeakyReLU(),
                                       nn.Linear(args.z_dims, args.n_agents),
                                       nn.Softmax(dim=-1))

    def forward(self, inputs, zit, test_mode=False):
        b, a, e = inputs.size()

        c_hidden = F.relu(self.poli_app1(inputs), inplace=True)
        c_mu = self.poli_app2(c_hidden)
        c_sigma = self.poli_app3(c_hidden)

        if test_mode:
            c_sigma = torch.clamp(torch.exp(c_sigma), min=self.args.var_floor, max=self.args.var_floor)  # var
        else:
            c_sigma = torch.clamp(torch.exp(c_sigma), min=self.args.var_floor)  # var
            # z_dot_sigma = torch.exp(z_dot_sigma)  # var

        c_dist_agents = []
        c_agents = []
        for i in range(a):
            c_dist_i = D.Normal(c_mu[:, i, :], c_sigma[:, i, :] ** (1 / 2))  # 此处1 / 2必须加括号，不然就计算为一次方再除以二。重要！！！！！
            c_i = c_dist_i.rsample()
            c_dist_agents.append(c_dist_i)
            c_agents.append(c_i)
        c = torch.stack(c_agents, dim=1)

        SN_i = self.SN_module(zit).view(b, a, -1)  # (b*a, -1)

        # if not test_mode:
        #     print(SN_i[0])

        return c.view(b, a, -1), c_dist_agents, SN_i