import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch.distributions import MultivariateNormal
import torch.distributions as D


# high level policy generate module
class HighLevelPolicy(nn.Module):
    # 由于每个智能体的obs维度不同，所有输入的维度单独列出来，而不是放在args中。
    def __init__(self, args):
        super(HighLevelPolicy, self).__init__()
        self.args = args
        self.n_agents = args.n_agents

        # high hypyer network
        if args.two_hyper_layers:
            self.w1 = nn.Sequential(
                nn.Linear(args.obs_input_dims, args.high_hyper_hidden_dims),
                nn.ReLU(),
                nn.Linear(args.high_hyper_hidden_dims, args.state_dims * args.z_dims)
            )
            # self.w2 = nn.Sequential(
            #     nn.Linear(args.obs_input_dims, args.high_hyper_hidden_dims),
            #     nn.ReLU(),
            #     nn.Linear(args.high_hyper_hidden_dims, args.high_mixer_hidden_dims * args.z_dims)
            # )
            # self.w3 = nn.Sequential(
            #     nn.Linear(args.obs_input_dims, args.high_hyper_hidden_dims),
            #     nn.ReLU(),
            #     nn.Linear(args.high_hyper_hidden_dims, args.high_mixer_hidden_dims * args.z_dims)
            # )
        else:
            self.w1 = nn.Sequential(
                nn.Linear(args.obs_input_dims, args.state_dims * args.z_dims)
            )
            # self.w2 = nn.Sequential(
            #     nn.Linear(args.obs_input_dims, args.high_mixer_hidden_dims * args.z_dims)
            # )
            # self.w3 = nn.Sequential(
            #     nn.Linear(args.obs_input_dims, args.high_mixer_hidden_dims * args.z_dims)
            # )

        self.b1 = nn.Sequential(
            nn.Linear(args.obs_input_dims, args.z_dims)
        )
        # self.b2 = nn.Sequential(
        #     nn.Linear(args.obs_input_dims, args.high_mixer_hidden_dims),
        #     nn.ReLU(),
        #     nn.Linear(args.high_mixer_hidden_dims, args.z_dims)
        # )
        # self.b3 = nn.Sequential(
        #     nn.Linear(args.obs_input_dims, args.high_mixer_hidden_dims),
        #     nn.ReLU(),
        #     nn.Linear(args.high_mixer_hidden_dims, args.z_dims)
        # )

        self.mu = nn.Sequential(nn.Linear(args.z_dims, args.z_dims),
                                nn.BatchNorm1d(args.z_dims),
                                nn.LeakyReLU(),
                                nn.Linear(args.z_dims, args.z_dims))
        self.sigma = nn.Sequential(nn.Linear(args.z_dims, args.z_dims),
                                   nn.BatchNorm1d(args.z_dims),
                                   nn.LeakyReLU(),
                                   nn.Linear(args.z_dims, args.z_dims))

        if self.args.use_cuda:
            self.cuda()
            # self.w1.cuda()
            # self.w2.cuda()
            # self.w3.cuda()
            # self.b1.cuda()
            # self.b2.cuda()
            # self.b3.cuda()
            # self.mu.cuda()
            # self.sigma.cuda()

    def forward(self, ep_batch, obs, t):
        state = ep_batch["state"][:, t, :]

        b, a, e = obs.size()
        obs = obs.view(b*a, e)
        state = state.unsqueeze(1).repeat(1, a, 1).view(b*a, 1, -1)

        w1 = self.w1(obs).view(-1, self.args.state_dims, self.args.z_dims)
        b1 = self.b1(obs).view(-1, 1, self.args.z_dims)

        z_hidden = F.elu(torch.matmul(state, w1) + b1).squeeze()

        mu = self.mu(z_hidden)
        sigma = self.sigma(z_hidden)
        sigma = torch.clamp(torch.exp(sigma), min=self.args.var_floor)  # var
        dist = D.Normal(mu, sigma ** (1 / 2))
        z_i = dist.rsample()

        return dist


class HighLevelPolicy2(nn.Module):
    def __init__(self, args):
        super(HighLevelPolicy2, self).__init__()
        self.args = args
        self.n_agents = args.n_agents

        self.fc1 = nn.Linear(args.state_dims, args.z_dims)

        self.mu = nn.Sequential(nn.Linear(args.z_dims, args.z_dims),
                                nn.BatchNorm1d(args.z_dims),
                                nn.LeakyReLU(),
                                nn.Linear(args.z_dims, args.z_dims))
        self.sigma = nn.Sequential(nn.Linear(args.z_dims, args.z_dims),
                                   nn.BatchNorm1d(args.z_dims),
                                   nn.LeakyReLU(),
                                   nn.Linear(args.z_dims, args.z_dims))

        if self.args.use_cuda:
            self.cuda()

    def forward(self, ep_batch, obs, t):
        state = ep_batch["state"][:, t, :]

        b, a, e = obs.size()
        state = state.unsqueeze(1).repeat(1, a, 1).view(b*a, 1, -1)

        z_hidden = self.fc1(state).squeeze()

        mu = self.mu(z_hidden)
        sigma = self.sigma(z_hidden)
        sigma = torch.clamp(torch.exp(sigma), min=self.args.var_floor)  # var
        dist = D.Normal(mu, sigma ** (1 / 2))

        return dist