from modules.gire.stage_two import PolicyAppModule_TwoStage
from modules.gire.stage_two import DecCoachNet_TwoStage

import numpy as np
import torch
import os
import argparse
# from torch.distributions import kl_divergence
from torch.optim import Adam

# import matplotlib
# matplotlib.use('AGG')
import matplotlib.pyplot as plt


def load_data(dic_path='', map_name='3s_vs_5z'):

    dims_path = os.path.join(dic_path, map_name + '_dims.csv')
    dims = np.loadtxt(dims_path, dtype=np.int, delimiter=',')

    # csv_path = os.path.join(dic_path, map_name + '.csv')
    # data = np.loadtxt(csv_path, dtype=np.float, delimiter=',')

    npy_path = os.path.join(dic_path, map_name + '.npy')
    data = np.load(npy_path)
    data = data.reshape(-1, data.shape[-1])

    # obs = data[:, :dims[0]]  # (602400, 48)
    # states = data[:, dims[0]:]  # (602400, 68)

    return data, dims


def get_path(checkpoint_path='results/models/', map_name='3s_vs_5z', args=None, seed=0):

    checkpoint_path = os.path.join(checkpoint_path, map_name, 'gire_z_env/seed_{}'.format(seed))

    timesteps = []
    timestep_to_load = 0

    if not os.path.isdir(checkpoint_path):
        print("Checkpoint directiory {} doesn't exist".format(checkpoint_path))
        return

    # Go through all files in args.checkpoint_path
    for name in os.listdir(checkpoint_path):
        full_name = os.path.join(checkpoint_path, name)
        # Check if they are dirs the names of which are numbers
        if os.path.isdir(full_name) and name.isdigit():
            timesteps.append(int(name))

    load_step = args.load_step  # 可以选择离load_step最近的，可以作为函数定义的参数。
    if load_step == 0:
        # choose the max timestep
        timestep_to_load = max(timesteps)
    else:
        # choose the timestep closest to load_step
        timestep_to_load = min(timesteps, key=lambda x: abs(x - load_step))

    model_path = os.path.join(checkpoint_path, str(timestep_to_load))

    return model_path


def get_model(args, dic_path):
    # load coach network parameters
    coach_net = DecCoachNet_TwoStage(args)
    coach_net.load_state_dict(torch.load("{}/coach_net.th".format(dic_path), map_location=lambda storage, loc: storage))
    # initialize policy approximation network
    policy_app = PolicyAppModule_TwoStage(args)

    return coach_net, policy_app


def save_model(dic_path, model, i):
    torch.save(model.state_dict(), "{}/policy_app_{}.th".format(dic_path, i))


def plot(eval_loss, eval_mse_loss, eval_l1_max_loss, i):
    # plt.plot([p[0] for p in train_loss], [p[1] for p in train_loss], c='b')
    plt.plot([p[0] for p in eval_loss], [p[1] for p in eval_loss], c='r')
    plt.plot([p[0] for p in eval_mse_loss], [p[1] for p in eval_mse_loss], c='y')
    plt.plot([p[0] for p in eval_l1_max_loss], [p[1] for p in eval_l1_max_loss], c='g')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('loss in epoch {}'.format(i))
    # plt.savefig('i.jpg')
    plt.show()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--map_name', type=str, default='3s5z_vs_3s7z', help='map name')
    parser.add_argument('--load_step', type=int, default=10000000, help='load step')  # 6001798
    parser.add_argument('--seed', type=int, default=2, help='random seed')  # [0, 2, 321]

    parser.add_argument('--cuda', type=bool, default=True, help='cuda or not')
    parser.add_argument('--high_hyper_hidden_dims', type=int, default=64, help='local hyper hidden dims')
    parser.add_argument('--z_dims', type=int, default=64, help='z dims')
    parser.add_argument('--rnn_hidden_dim', type=int, default=64, help='rnn hidden dims')
    parser.add_argument('--batch_size', type=int, default=2048, help='batch size')

    parser.add_argument('--eval_inter', type=int, default=1000, help='batch size')
    parser.add_argument('--plot_inter', type=int, default=50000, help='batch size')
    parser.add_argument('--max_step', type=int, default=500001, help='batch size')

    # parser.add_argument('--eval_inter', type=int, default=10, help='batch size')
    # parser.add_argument('--plot_inter', type=int, default=20, help='batch size')
    # parser.add_argument('--max_step', type=int, default=101, help='batch size')

    parser.add_argument('--grad_norm_clip', type=int, default=10, help='Reduce magnitude of gradients above this L2 norm')
    parser.add_argument('--var_floor', type=float, default=0.0000001, help='min sigma')  # 0.002
    parser.add_argument('--two_hyper_layers', type=bool, default=True, help='two hyper layers')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    # 获取参数
    args = get_args()

    # 设置seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # 获取数据和部分参数
    dic_path = get_path(map_name=args.map_name, args=args, seed=args.seed)
    print(dic_path)

    data, dims = load_data(dic_path=dic_path, map_name=args.map_name)
    # obs = data[:, :dims[0]]  # (602400, 48)
    # states = data[:, dims[0]:]  # (602400, 68)

    o_dim, s_dim, a_dim, n_agents = dims[0], dims[1], dims[2], dims[3]
    args.obs_input_dims = o_dim + a_dim + n_agents
    args.state_dims = s_dim
    args.n_agents = n_agents
    args.save_inter = max(int(args.max_step // 5), 5)

    # 加载模型
    coach_net, policy_app = get_model(args=args, dic_path=dic_path)
    if args.cuda:
        coach_net.cuda()
        policy_app.cuda()

    # train
    np.random.shuffle(data)
    train_data = data[:int(data.shape[0]*0.8)]
    eval_data = data[int(data.shape[0]*0.8):]

    # optimizer
    params = policy_app.parameters()
    optimizer = Adam(params, lr=1e-4, weight_decay=0.0)

    train_loss = []
    eval_loss = []
    eval_mse_loss = []
    eval_l1_max_loss = []
    last_eval, last_plot, last_save = 0, 0, 0
    for i in range(args.max_step):
        train_idxs = np.random.choice(train_data.shape[0], args.batch_size)
        epoch_train = train_data[train_idxs]
        epoch_train = torch.tensor(epoch_train, dtype=torch.float).cuda()

        epoch_z = coach_net.forward(epoch_train[:, args.obs_input_dims:], epoch_train[:, :args.obs_input_dims])
        epoch_z_dot = policy_app.forward(epoch_train[:, :args.obs_input_dims])

        loss_mse = ((epoch_z.loc - epoch_z_dot)**2).sum(dim=-1).mean()  # mse loss
        # loss_l1 = (abs(epoch_z.loc - epoch_z_dot)).sum(dim=-1).mean()  # l1 loss
        # loss_l1_max = abs(epoch_z.loc - epoch_z_dot).max(dim=1).values.mean()  # l1 max loss
        # loss = kl_divergence(epoch_z, epoch_z_dot).sum(dim=-1).mean()

        loss = loss_mse
        train_loss.append([i, loss.clone().cpu().detach().numpy()])
        # print('epoch {}, train loss {}:'.format(i, train_loss[-1]))

        # Optimise
        optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(params, args.grad_norm_clip)
        optimizer.step()

        if i - last_eval > args.eval_inter:
            last_eval = i

            eval_idxs = np.random.choice(eval_data.shape[0], args.batch_size)
            epoch_eval = eval_data[eval_idxs]
            epoch_eval = torch.tensor(epoch_eval, dtype=torch.float).cuda()

            with torch.no_grad():

                epoch_z = coach_net.forward(epoch_eval[:, args.obs_input_dims:], epoch_eval[:, :args.obs_input_dims])
                epoch_z_dot = policy_app.forward(epoch_eval[:, :args.obs_input_dims])

                loss_mse = ((epoch_z.loc - epoch_z_dot) ** 2).sum(dim=-1).mean()  # mse
                # loss_l1 = (abs(epoch_z.loc - epoch_z_dot)).sum(dim=-1).mean()  # l1 loss
                # loss_l1_max = abs(epoch_z.loc - epoch_z_dot).max(dim=1).values.mean()  # l1 max loss
                # loss = kl_divergence(epoch_z, epoch_z_dot).sum(dim=-1).mean()

                loss = loss_mse
                eval_loss.append([i, loss.clone().cpu().detach().numpy()])
                # eval_mse_loss.append([i, loss_mse.clone().cpu().detach().numpy()])
                # eval_l1_max_loss.append([i, loss_l1_max.clone().cpu().detach().numpy()])

                print('epoch {}, train loss {}:'.format(i, train_loss[-1]))
                print('epoch {}, eval loss {}:'.format(i, eval_loss[-1]))
                # print('epoch {}, eval l1 max loss {}:'.format(i, eval_l1_max_loss[-1]))
                # print('epoch {}, eval mse loss {}:'.format(i, eval_mse_loss[-1]))
                print('\t')

        if i - last_plot > args.plot_inter:
            last_plot = i

            # plot(eval_loss, eval_mse_loss, eval_l1_max_loss, i)

        if i - last_save >= args.save_inter:
            last_save = i

            save_model(dic_path, policy_app, i)