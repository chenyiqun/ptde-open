import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch.distributions import MultivariateNormal
import torch.distributions as D


# high level policy generate module
class MutualInfo(nn.Module):
    # 由于每个智能体的obs维度不同，所有输入的维度单独列出来，而不是放在args中。
    def __init__(self, args):
        super(MutualInfo, self).__init__()
        self.args = args

        self.fc1 = nn.Linear(args.obs_input_dims, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)

        self.mu = nn.Sequential(nn.Linear(args.rnn_hidden_dim, args.p_dims),
                                nn.BatchNorm1d(args.p_dims),
                                nn.LeakyReLU(),
                                nn.Linear(args.p_dims, args.p_dims))
        self.sigma = nn.Sequential(nn.Linear(args.rnn_hidden_dim, args.p_dims),
                                   nn.BatchNorm1d(args.p_dims),
                                   nn.LeakyReLU(),
                                   nn.Linear(args.p_dims, args.p_dims))

        if self.args.use_cuda:
            self.cuda()
            # self.fc1.cuda()
            # self.rnn.cuda()
            # self.fc2.cuda()
            # self.fc3.cuda()

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, obs, hidden_state):
        # 此处的obs加入了last_action和agent_idx之后
        # z的维度应该为[n_agents, coach_instruc_dims]
        b, a, e = obs.size()

        x = F.relu(self.fc1(obs.view(-1, e)))
        # h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h_out = self.rnn(x, hidden_state.view(b*a, -1))

        p_mu = self.mu(h_out)
        p_sigma = self.sigma(h_out)
        p_sigma = torch.clamp(torch.exp(p_sigma), min=self.args.var_floor)  # var

        dist = D.Normal(p_mu, p_sigma ** 1/2)
        p_i = dist.rsample()

        return h_out.view(b, a, -1), dist

        # return p_i.view(b, a, -1), h_out.view(b, a, -1), dist