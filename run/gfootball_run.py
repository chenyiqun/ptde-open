import datetime
import os
import pprint
import time
import threading
import torch as th
from types import SimpleNamespace as SN
from utils.logging import Logger
from utils.timehelper import time_left, time_str
from os.path import dirname, abspath

from learners import REGISTRY as le_REGISTRY
from runners import REGISTRY as r_REGISTRY
from controllers import REGISTRY as mac_REGISTRY
from components.episode_buffer import ReplayBuffer
from components.transforms import OneHot

from smac.env import StarCraft2Env

import numpy as np
import csv

def get_agent_own_state_size(env_args):
    sc_env = StarCraft2Env(**env_args)
    # qatten parameter setting (only use in qatten)
    return  4 + sc_env.shield_bits_ally + sc_env.unit_type_bits

def run(_run, _config, _log):

    # all_seed = [0, 3, 2, 1, 321, 19990321]
    all_seed = [2]  # [0, 3, 2]
    for seed in all_seed:
        _config['seed'] = seed

        # check args sanity
        _config = args_sanity_check(_config, _log)

        args = SN(**_config)
        args.device = "cuda" if args.use_cuda else "cpu"

        # setup loggers
        logger = Logger(_log)

        _log.info("Experiment Parameters:")
        experiment_params = pprint.pformat(_config,
                                           indent=4,
                                           width=1)
        _log.info("\n\n" + experiment_params + "\n")

        # configure tensorboard logger
        unique_token = "{}__{}".format(args.name, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

        unique_token = unique_token.split('=')[0]

        args.unique_token = unique_token
        if args.use_tensorboard:
            tb_logs_direc = os.path.join(dirname(dirname(dirname(abspath(__file__)))), "results", "tb_logs",
                                         args.env_args['map_name'])
            tb_exp_direc = os.path.join(tb_logs_direc, "{}", "seed_" + str(args.seed)).format(unique_token)
            logger.setup_tb(tb_exp_direc)

        # sacred is on by default
        logger.setup_sacred(_run)

        # Run and train
        run_sequential(args=args, logger=logger)

    # Clean up after finishing
    print("Exiting Main")

    print("Stopping all threads")
    for t in threading.enumerate():
        if t.name != "MainThread":
            print("Thread {} is alive! Is daemon: {}".format(t.name, t.daemon))
            t.join(timeout=1)
            print("Thread joined")

    print("Exiting script")

    # Making sure framework really exits
    os._exit(os.EX_OK)


# def evaluate_sequential(args, runner):
#
#     for _ in range(args.test_nepisode):
#         runner.run(test_mode=True)
#
#     if args.save_replay:
#         runner.save_replay()
#
#     runner.close_env()


def evaluate_sequential(args, runner):

    aver_win = []
    for i in range(64):  # args.test_nepisode
        win_rate = runner.run(test_mode=True)  # True的时候返回胜率等信息。
        aver_win.append(win_rate)
        print('testing {} episode, winning rate {}'.format(i, win_rate))
    print('\t')
    print('average winning rate is {}, in random seed {}.'.format(sum(aver_win)/len(aver_win), args.seed))
    print('\t')

    if args.save_replay:
        runner.save_replay()

    runner.close_env()


def run_sequential(args, logger):

    # Init runner so we can get env info
    runner = r_REGISTRY[args.runner](args=args, logger=logger)

    # Set up schemes and groups here
    env_info = runner.get_env_info()
    args.n_agents = env_info["n_agents"]
    args.n_actions = env_info["n_actions"]
    args.state_shape = env_info["state_shape"]
    args.accumulated_episodes = getattr(args, "accumulated_episodes", None)

    if getattr(args, 'agent_own_state_size', False):
        args.agent_own_state_size = get_agent_own_state_size(args.env_args)

    # Default/Base scheme
    scheme = {
        "state": {"vshape": env_info["state_shape"]},
        "obs": {"vshape": env_info["obs_shape"], "group": "agents"},
        "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
        "avail_actions": {"vshape": (env_info["n_actions"],), "group": "agents", "dtype": th.int},
        "probs": {"vshape": (env_info["n_actions"],), "group": "agents", "dtype": th.float},
        "reward": {"vshape": (1,)},
        "terminated": {"vshape": (1,), "dtype": th.uint8},
    }
    groups = {
        "agents": args.n_agents
    }
    preprocess = {
        "actions": ("actions_onehot", [OneHot(out_dim=args.n_actions)])
    }

    buffer = ReplayBuffer(scheme, groups, args.buffer_size, env_info["episode_limit"] + 1,
                          preprocess=preprocess,
                          device="cpu" if args.buffer_cpu_only else args.device)
    # Setup multiagent controller here
    mac = mac_REGISTRY[args.mac](buffer.scheme, groups, args)

    # Give runner the scheme
    runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=mac)

    # Learner
    learner = le_REGISTRY[args.learner](mac, buffer.scheme, logger, args)

    if args.use_cuda:
        learner.cuda()

    if args.checkpoint_path != "":

        args.checkpoint_path = os.path.join(args.checkpoint_path, args.env_args['map_name'],
                                            'qmix_env/seed_{}'.format(args.seed))  # qmix_env gire_z_env

        timesteps = []
        timestep_to_load = 0

        if not os.path.isdir(args.checkpoint_path):
            logger.console_logger.info("Checkpoint directiory {} doesn't exist".format(args.checkpoint_path))
            return

        # Go through all files in args.checkpoint_path
        for name in os.listdir(args.checkpoint_path):
            full_name = os.path.join(args.checkpoint_path, name)
            # Check if they are dirs the names of which are numbers
            if os.path.isdir(full_name) and name.isdigit():
                timesteps.append(int(name))

        if args.load_step == 0:
            # choose the max timestep
            timestep_to_load = max(timesteps)
        else:
            # choose the timestep closest to load_step
            timestep_to_load = min(timesteps, key=lambda x: abs(x - args.load_step))

        model_path = os.path.join(args.checkpoint_path, str(timestep_to_load))

        logger.console_logger.info("Loading model from {}".format(model_path))
        learner.load_models(model_path)
        runner.t_env = timestep_to_load

        # gire_z generate data
        if args.generate_data:
            n_agent = groups['agents']
            o_dim = scheme['obs']['vshape']
            s_dim = scheme['state']['vshape']
            a_dim = scheme['avail_actions']['vshape'][0]
            data_save_path = os.path.join(model_path, args.env_args['map_name'] + '_dims.csv')
            with open(data_save_path, 'w', encoding='utf8', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([o_dim, s_dim, a_dim, n_agent])

            data_save_path = os.path.join(model_path, args.env_args['map_name'] + '.npy')
            all_data = []
            for k in range(100):
                print('episode {} is generating.'.format(k))
                # Run for a whole episode at a time
                with th.no_grad():
                    episode_batch = runner.run(test_mode=False)
                    episode_obs_input = []
                    episode_length = episode_batch.data.transition_data['obs'].size()[1]
                    for i in range(episode_length):
                        episode_obs_input.append(build_inputs(episode_batch, i, args).unsqueeze(1))
                    episode_obs_input = th.cat(episode_obs_input, dim=1)
                    episode_states = episode_batch.data.transition_data['state'].unsqueeze(2).repeat(1, 1, n_agent, 1)
                    episode_data = th.cat((episode_obs_input, episode_states), dim=-1).view(-1, (
                                o_dim + a_dim + n_agent) + s_dim).cpu().numpy()
                    all_data.append(episode_data)
                    # with open(data_save_path, 'a', encoding='utf8', newline='') as f:
                    #     writer = csv.writer(f)
                    #     # writer.writerow(headers)
                    #     writer.writerows(episode_data)

            np.save(data_save_path, np.array(all_data))

            return

        if args.evaluate or args.save_replay:
            evaluate_sequential(args, runner)
            return

    # start training
    episode = 0
    last_test_T = -args.test_interval - 1
    last_log_T = 0
    model_save_time = 0

    start_time = time.time()
    last_time = start_time

    logger.console_logger.info("Beginning training for {} timesteps".format(args.t_max))

    # plot data
    plot_win_rates = []

    while runner.t_env <= args.t_max:

        # Run for a whole episode at a time

        with th.no_grad():
            episode_batch = runner.run(test_mode=False)
            buffer.insert_episode_batch(episode_batch)

        if buffer.can_sample(args.batch_size):
            next_episode = episode + args.batch_size_run
            if args.accumulated_episodes and next_episode % args.accumulated_episodes != 0:
                continue

            episode_sample = buffer.sample(args.batch_size)

            # Truncate batch to only filled timesteps
            max_ep_t = episode_sample.max_t_filled()
            episode_sample = episode_sample[:, :max_ep_t]

            if episode_sample.device != args.device:
                episode_sample.to(args.device)

            learner.train(episode_sample, runner.t_env, episode)
            del episode_sample

        # Execute test runs once in a while
        n_test_runs = max(1, args.test_nepisode // runner.batch_size)
        if (runner.t_env - last_test_T) / args.test_interval >= 1.0:

            logger.console_logger.info("t_env: {} / {}".format(runner.t_env, args.t_max))
            logger.console_logger.info("Estimated time left: {}. Time passed: {}".format(
                time_left(last_time, last_test_T, runner.t_env, args.t_max), time_str(time.time() - start_time)))
            last_time = time.time()

            last_test_T = runner.t_env

            test_win_rates = []
            for _ in range(n_test_runs):
                win_rate = runner.run(test_mode=True)
                test_win_rates.append(win_rate)

            test_win_rate = sum(test_win_rates) / len(test_win_rates)
            plot_win_rates.append([runner.t_env, test_win_rate])

        if args.save_model and (runner.t_env - model_save_time >= args.save_model_interval or model_save_time == 0):
            model_save_time = runner.t_env
            save_path = os.path.join(args.local_results_path, "models", args.env_args['map_name'], args.unique_token,
                                     "seed_" + str(args.seed), str(runner.t_env))
            #"results/models/{}".format(unique_token)
            os.makedirs(save_path, exist_ok=True)
            logger.console_logger.info("Saving models to {}".format(save_path))

            # learner should handle saving/loading -- delegate actor save/load to mac,
            # use appropriate filenames to do critics, optimizer states
            learner.save_models(save_path)

            # 保存训练胜率 .npy
            temp_plot_win_rates = np.array(plot_win_rates[:])
            plot_path = os.path.join("results", "plot_data", args.env_args['map_name'], args.unique_token)
            os.makedirs(plot_path, exist_ok=True)
            np.save(os.path.join(plot_path, "win_rates_seed_{}_{}.npy".format(args.seed, runner.t_env)), temp_plot_win_rates)
            logger.console_logger.info("Saving plot data to {}".format(plot_path))

        episode += args.batch_size_run

        if (runner.t_env - last_log_T) >= args.log_interval:
            logger.log_stat("episode", episode, runner.t_env)
            logger.print_recent_stats()
            last_log_T = runner.t_env

    runner.close_env()
    logger.console_logger.info("Finished Training")


def args_sanity_check(config, _log):

    # set CUDA flags
    # config["use_cuda"] = True # Use cuda whenever possible!
    if config["use_cuda"] and not th.cuda.is_available():
        config["use_cuda"] = False
        _log.warning("CUDA flag use_cuda was switched OFF automatically because no CUDA devices are available!")

    if config["test_nepisode"] < config["batch_size_run"]:
        config["test_nepisode"] = config["batch_size_run"]
    else:
        config["test_nepisode"] = (config["test_nepisode"]//config["batch_size_run"]) * config["batch_size_run"]

    return config


def build_inputs(batch, t, args):  # add
    # Assumes homogenous agents with flat observations.
    # Other MACs might want to e.g. delegate building inputs to each agent
    bs = batch.batch_size
    inputs = []
    inputs.append(batch["obs"][:, t])  # b1av
    if args.obs_last_action:
        if t == 0:
            inputs.append(th.zeros_like(batch["actions_onehot"][:, t]))
        else:
            inputs.append(batch["actions_onehot"][:, t - 1])
    if args.obs_agent_id:
        inputs.append(th.eye(args.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1))

    inputs = th.cat([x.reshape(bs, args.n_agents, -1) for x in inputs], dim=-1)
    return inputs
