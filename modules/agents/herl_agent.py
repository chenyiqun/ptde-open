import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D


class HERLAgent(nn.Module):
    def __init__(self, obs_input_dims, args):
        super(HERLAgent, self).__init__()
        self.args = args

        self.fc1 = nn.Linear(obs_input_dims, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc3 = nn.Linear(args.rnn_hidden_dim + args.z_dims, args.n_actions)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state, z_dot):
        b, a, e = inputs.size()

        x = F.relu(self.fc1(inputs.view(-1, e)), inplace=True)
        h = self.rnn(x, hidden_state.view(b*a, -1))
        h_oi = self.fc2(h)
        q = self.fc3(torch.concat([h_oi, z_dot.view(b*a, -1)], dim=-1))

        return q.view(b, a, -1), h.view(b, a, -1)


class PolicyAppModule(nn.Module):
    def __init__(self, obs_input_dims, args):
        super(PolicyAppModule, self).__init__()
        self.args = args

        self.poli_app1 = nn.Linear(obs_input_dims, args.rnn_hidden_dim)
        self.poli_app2 = nn.Linear(args.rnn_hidden_dim, args.z_dims)
        self.poli_app3 = nn.Linear(args.rnn_hidden_dim, args.z_dims)

        if args.use_cuda:
            self.poli_app1.cuda()
            self.poli_app2.cuda()
            self.poli_app3.cuda()

    def forward(self, inputs):
        b, a, e = inputs.size()

        z_dot_hidden = F.relu(self.poli_app1(inputs.view(-1, e)), inplace=True)
        z_dot_mu = F.relu(self.poli_app2(z_dot_hidden), inplace=True)
        z_dot_sigma = F.relu(self.poli_app3(z_dot_hidden), inplace=True)
        # z_dot_sigma = torch.diag_embed(z_dot_sigma)
        z_dot_sigma = torch.clamp(torch.exp(z_dot_sigma), min=self.args.var_floor)  # var

        dist = D.Normal(z_dot_mu, z_dot_sigma ** 1/2)
        z_dot = dist.rsample()

        return z_dot.view(b, a, -1), dist