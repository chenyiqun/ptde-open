import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
import copy


# mutual information module
class MutualInfo(nn.Module):
    def __init__(self, args):
        super(MutualInfo, self).__init__()
        self.args = args

        self.fc1 = nn.Linear(args.rnn_hidden_dim, args.rnn_hidden_dim)

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

    def forward(self, input_hidden_out):
        # input_hidden_out = input_hidden.clone()

        b, a, e = input_hidden_out.size()

        x = F.relu(self.fc1(input_hidden_out.view(-1, e)))

        p_mu = self.mu(x)
        p_sigma = self.sigma(x)
        p_sigma = torch.clamp(torch.exp(p_sigma), min=self.args.var_floor)  # var

        dist = D.Normal(p_mu, p_sigma ** 1/2)

        return dist