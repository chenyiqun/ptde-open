import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
from .mha import MHA


# decentralized coach net (o generate w,b)
class DecCoachNet(nn.Module):
    # 由于每个智能体的obs维度不同，所有输入的维度单独列出来，而不是放在args中。
    def __init__(self, args):
        super(DecCoachNet, self).__init__()
        self.args = args
        self.n_agents = args.n_agents

        # high hypyer network
        if args.two_hyper_layers:
            self.w1 = nn.Sequential(
                nn.Linear(args.obs_input_dims, args.high_hyper_hidden_dims),
                nn.ReLU(),
                nn.Linear(args.high_hyper_hidden_dims, args.state_dims * args.z_dims)
            )
        else:
            self.w1 = nn.Sequential(
                nn.Linear(args.obs_input_dims, args.state_dims * args.z_dims)
            )

        self.b1 = nn.Sequential(
            nn.Linear(args.obs_input_dims, args.z_dims)
        )

        self.mu = nn.Sequential(nn.Linear(args.z_dims, args.z_dims),
                                # nn.BatchNorm1d(args.z_dims),
                                nn.LeakyReLU(),
                                nn.Linear(args.z_dims, args.z_dims))
        self.sigma = nn.Sequential(nn.Linear(args.z_dims, args.z_dims),
                                   # nn.BatchNorm1d(args.z_dims),
                                   nn.LeakyReLU(),
                                   nn.Linear(args.z_dims, args.z_dims))

    def forward(self, ep_batch, obs, t, return_one=False):
        state = ep_batch["state"][:, t, :]

        b, a, e = obs.size()
        obs = obs.view(b*a, e)
        state = state.unsqueeze(1).repeat(1, a, 1).view(b*a, 1, -1)

        w1 = self.w1(obs).view(-1, self.args.state_dims, self.args.z_dims)
        b1 = self.b1(obs).view(-1, 1, self.args.z_dims)

        z_hidden = F.elu(torch.matmul(state, w1) + b1).squeeze()

        mu = self.mu(z_hidden)
        sigma = self.sigma(z_hidden)
        sigma = torch.clamp(torch.exp(sigma), min=self.args.var_floor)  # max=1000*self.args.var_floor
        # sigma = torch.exp(sigma)  # var

        dist = D.Normal(mu, sigma ** (1 / 2))

        if return_one:
            return dist
        else:
            with torch.no_grad():
                dist_wog = D.Normal(mu.clone(), sigma.clone() ** (1 / 2))
            return dist, dist_wog


# decentralized coach net (o generate w,b)
class AppDecCoachNet(nn.Module):
    # 由于每个智能体的obs维度不同，所有输入的维度单独列出来，而不是放在args中。
    def __init__(self, args):
        super(AppDecCoachNet, self).__init__()
        self.args = args
        self.n_agents = args.n_agents

        # high hypyer network
        if args.two_hyper_layers:
            self.w1 = nn.Sequential(
                nn.Linear(args.obs_input_dims, args.high_hyper_hidden_dims),
                nn.ReLU(),
                nn.Linear(args.high_hyper_hidden_dims, args.obs_input_dims * args.z_dims)
            )
        else:
            self.w1 = nn.Sequential(
                nn.Linear(args.obs_input_dims, args.obs_input_dims * args.z_dims)
            )

        self.b1 = nn.Sequential(
            nn.Linear(args.obs_input_dims, args.z_dims)
        )

        self.mu = nn.Sequential(nn.Linear(args.z_dims, args.z_dims),
                                # nn.BatchNorm1d(args.z_dims),
                                nn.LeakyReLU(),
                                nn.Linear(args.z_dims, args.z_dims))
        self.sigma = nn.Sequential(nn.Linear(args.z_dims, args.z_dims),
                                   # nn.BatchNorm1d(args.z_dims),
                                   nn.LeakyReLU(),
                                   nn.Linear(args.z_dims, args.z_dims))

    def forward(self, obs):

        b, a, e = obs.size()
        obs = obs.view(b * a, 1, e)

        w1 = self.w1(obs).view(-1, self.args.obs_input_dims, self.args.z_dims)
        b1 = self.b1(obs).view(-1, 1, self.args.z_dims)

        z_hidden = F.elu(torch.matmul(obs, w1) + b1).squeeze()

        mu = self.mu(z_hidden)
        sigma = self.sigma(z_hidden)
        sigma = torch.clamp(torch.exp(sigma), min=self.args.var_floor)  # var
        # sigma = torch.exp(sigma)  # var

        dist = D.Normal(mu, sigma ** (1 / 2))

        return dist.rsample(), dist


# decentralized coach net (s generate w, b)
class DecCoachNet2(nn.Module):
    # 由于每个智能体的obs维度不同，所有输入的维度单独列出来，而不是放在args中。
    def __init__(self, args):
        super(DecCoachNet2, self).__init__()
        self.args = args
        self.n_agents = args.n_agents

        # high hypyer network
        if args.two_hyper_layers:
            self.w1 = nn.Sequential(
                nn.Linear(args.state_dims, args.high_hyper_hidden_dims),  # state_dims obs_input_dims
                nn.ReLU(),
                nn.Linear(args.high_hyper_hidden_dims, args.obs_input_dims * args.z_dims)  # obs_input_dims state_dims
            )
        else:
            self.w1 = nn.Sequential(
                nn.Linear(args.obs_input_dims, args.obs_input_dims * args.z_dims)  # obs_input_dims state_dims
            )

        self.b1 = nn.Sequential(
            nn.Linear(args.state_dims, args.z_dims)  # state_dims obs_input_dims
        )

        self.mu = nn.Sequential(nn.Linear(args.z_dims, args.z_dims),
                                # nn.BatchNorm1d(args.z_dims),
                                nn.LeakyReLU(),
                                nn.Linear(args.z_dims, args.z_dims))
        self.sigma = nn.Sequential(nn.Linear(args.z_dims, args.z_dims),
                                   # nn.BatchNorm1d(args.z_dims),
                                   nn.LeakyReLU(),
                                   nn.Linear(args.z_dims, args.z_dims))

    def forward(self, ep_batch, obs, t, return_one=False):
        state = ep_batch["state"][:, t, :]

        b, a, e = obs.size()
        obs = obs.view(b*a, 1, e)
        state = state.unsqueeze(1).repeat(1, a, 1).view(b*a, 1, -1)

        w1 = self.w1(state).view(-1, self.args.obs_input_dims, self.args.z_dims)
        b1 = self.b1(state).view(-1, 1, self.args.z_dims)

        z_hidden = F.elu(torch.matmul(obs, w1) + b1).squeeze()

        mu = self.mu(z_hidden)
        sigma = self.sigma(z_hidden)
        sigma = torch.clamp(torch.exp(sigma), min=self.args.var_floor)  # var
        # sigma = torch.exp(sigma)  # var

        dist = D.Normal(mu, sigma ** (1 / 2))

        if return_one:
            return dist
        else:
            with torch.no_grad():
                dist_wog = D.Normal(mu.clone(), sigma.clone() ** (1 / 2))
            return dist, dist_wog


class StateCoachNet(nn.Module):
    def __init__(self, args):
        super(StateCoachNet, self).__init__()
        self.args = args
        self.n_agents = args.n_agents

        self.fc1 = nn.Linear(args.state_dims, args.z_dims)

        self.mu = nn.Sequential(nn.Linear(args.z_dims, args.z_dims),
                                # nn.BatchNorm1d(args.z_dims),
                                nn.LeakyReLU(),
                                nn.Linear(args.z_dims, args.z_dims))
        self.sigma = nn.Sequential(nn.Linear(args.z_dims, args.z_dims),
                                   # nn.BatchNorm1d(args.z_dims),
                                   nn.LeakyReLU(),
                                   nn.Linear(args.z_dims, args.z_dims))

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


# decentralized coach net (o generate w,b)
class CSRLCoachNet(nn.Module):
    # 由于每个智能体的obs维度不同，所有输入的维度单独列出来，而不是放在args中。
    def __init__(self, args):
        super(CSRLCoachNet, self).__init__()
        self.args = args
        self.n_agents = args.n_agents

        # high hypyer network
        if args.two_hyper_layers:
            self.w1 = nn.Sequential(
                nn.Linear(args.obs_input_dims, args.high_hyper_hidden_dims),
                nn.ReLU(),
                nn.Linear(args.high_hyper_hidden_dims, args.state_dims * args.z_dims)
            )
        else:
            self.w1 = nn.Sequential(
                nn.Linear(args.obs_input_dims, args.state_dims * args.z_dims)
            )

        self.b1 = nn.Sequential(
            nn.Linear(args.obs_input_dims, args.z_dims)
        )

    def forward(self, ep_batch, obs, t):
        state = ep_batch["state"][:, t, :]

        b, a, e = obs.size()
        obs = obs.view(b*a, e)
        state = state.unsqueeze(1).repeat(1, a, 1).view(b*a, 1, -1)

        w1 = self.w1(obs).view(-1, self.args.state_dims, self.args.z_dims)
        b1 = self.b1(obs).view(-1, 1, self.args.z_dims)

        z_hidden = F.elu(torch.matmul(state, w1) + b1).squeeze()

        return z_hidden