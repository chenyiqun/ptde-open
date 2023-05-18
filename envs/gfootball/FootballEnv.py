import numpy as np
import gfootball.env as football_env
from gfootball.env import observation_preprocessing
from ..multiagentenv import MultiAgentEnv
import gym
import torch as th
import copy


class GoogleFootballEnv(MultiAgentEnv):

    def __init__(
        self,
        dense_reward=False,
        write_full_episode_dumps=False,
        write_goal_dumps=False,
        dump_freq=0,
        render=False,
        num_agents=4,
        time_limit=200,
        time_step=0,
        map_name='academy_counterattack_hard',
        stacked=False,
        representation="raw",
        rewards='scoring,checkpoints',
        logdir='football_dumps',
        write_video=False,
        number_of_right_players_agent_controls=0,
        seed=0,
    ):
        self.dense_reward = dense_reward
        self.write_full_episode_dumps = write_full_episode_dumps
        self.write_goal_dumps = write_goal_dumps
        self.dump_freq = dump_freq
        self.render = render
        self.n_agents = num_agents
        self.episode_limit = time_limit
        self.time_step = time_step
        self.env_name = map_name
        self.stacked = stacked
        self.representation = representation
        self.rewards = rewards
        self.logdir = logdir
        self.write_video = write_video
        self.number_of_right_players_agent_controls = number_of_right_players_agent_controls
        self.seed = seed

        self.env = football_env.create_environment(
            write_full_episode_dumps=self.write_full_episode_dumps,
            write_goal_dumps=self.write_goal_dumps,
            env_name=self.env_name,
            stacked=self.stacked,
            representation=self.representation,
            rewards=self.rewards,
            logdir=self.logdir,
            render=self.render,
            write_video=self.write_video,
            dump_frequency=self.dump_freq,
            number_of_left_players_agent_controls=self.n_agents,
            number_of_right_players_agent_controls=self.number_of_right_players_agent_controls,
            channel_dimensions=(observation_preprocessing.SMM_WIDTH, observation_preprocessing.SMM_HEIGHT))
        self.env.seed(self.seed)

        # obs_space_low = self.env.observation_space.low[0]
        # obs_space_high = self.env.observation_space.high[0]

        self.action_space = [gym.spaces.Discrete(
            self.env.action_space.nvec[1]) for _ in range(self.n_agents)]
        # self.observation_space = [
        #     gym.spaces.Box(low=obs_space_low, high=obs_space_high, dtype=self.env.observation_space.dtype) for _ in range(self.n_agents)
        # ]

        self.n_actions = self.action_space[0].n
        # self.obs = None

        self.global_obs_dict_last = {}
        self.global_obs_dict = {}
        self.partial_obs_list_last = []
        self.partial_obs_list = []
        self.position_rank = False  # False为agent i的信息在向量中位置固定；True为根据距离排序过的，不固定。

        self.state = None
        self.obs = None

    def step(self, _actions):
        """Returns reward, terminated, info."""
        if th.is_tensor(_actions):
            actions = _actions.cpu().numpy()
        else:
            actions = _actions
        self.time_step += 1
        obs, rewards, done, infos = self.env.step(actions.tolist())

        self.get_obs_dict(obs)
        self.get_obs()
        self.get_global_state()

        if self.time_step >= self.episode_limit:
            done = True

        return sum(rewards), done, infos

    def get_obs(self):
        """Returns all agent observations in a list."""
        obs_total = []
        for i in range(self.n_agents):
            obs = []

            obs.extend(self.partial_obs_list[i]['position'])  # 绝对位置
            obs.extend(self.partial_obs_list[i]['velocity'])  # 自身绝对速度。
            obs.extend(self.partial_obs_list[i]['mate_relative_position_and_velocity'])  # 相对队友的位置和速度（方向）
            obs.extend(self.partial_obs_list[i]['enemy_relative_position_and_velocity'])  # 相对对手的位置和速度（方向）
            obs.extend(self.global_obs_dict['ball_position'])  # 球全局位置
            if self.global_obs_dict['ball_owned_team'] == 0:  # 0左队持球 -1无人持球 1右队持球
                obs.extend([1, 0, 0])
            elif self.global_obs_dict['ball_owned_team'] == -1:
                obs.extend([0, 1, 0])
            else:
                obs.extend([0, 0, 1])

            if self.env_name == 'academy_3_vs_1_with_keeper' or \
                    self.env_name == 'academy_3_vs_2_with_keeper' or \
                    self.env_name == 'academy_3_vs_3_with_keeper' or \
                    self.env_name == 'academy_run_pass_and_shoot_with_keeper':
                obs.extend(np.eye(self.n_agents + 1)[self.partial_obs_list[i]['id']])
            else:
                obs.extend(np.eye(11)[self.partial_obs_list[i]['id']])

            obs_total.append(obs)

        self.obs = np.array(obs_total)


            # obs.extend(self.partial_obs_list[i]['position'])  # 绝对位置
            # obs.extend(self.partial_obs_list[i]['velocity'])  # 自身自身绝绝对速度，此处只包括自己的绝对速度。
            # obs.append(self.partial_obs_list[i]['tired'])  # 自身疲劳值，only own。
            # obs.extend(self.partial_obs_list[i]['role'])  # 自身角色
            # obs.extend(self.partial_obs_list[i]['sticky_actions'])  # 自身粘滞动作
            # # obs.append(self.partial_obs_list[i]['yellow_card'])  # 黄牌（小场景无黄牌）
            # obs.extend(self.partial_obs_list[i]['ball_relative_position'])  # 球的相对位置
            # obs.append(self.partial_obs_list[i]['own_ball'])  # 自身是否持球？
            # obs.extend(self.partial_obs_list[i]['mate_relative_position_and_velocity'])  # 队友的相对位置和速度（后续可改进为范围内的队友和对手，而非全部）
            # obs.extend(self.partial_obs_list[i]['enemy_relative_position_and_velocity'])  # 对手的相对位置和速度
            # obs.extend(self.global_obs_dict['ball_position'])  # 球全局位置
            # obs.extend(self.global_obs_dict['ball_velocity'])  # 球全局速度
            # obs.extend(self.global_obs_dict['ball_rotation'])  # 球全局转速
            # if self.global_obs_dict['ball_owned_team'] == 0:  # 0左队持球 -1无人持球 1右队持球
            #     obs.extend([1, 0, 0])
            # elif self.global_obs_dict['ball_owned_team'] == -1:
            #     obs.extend([0, 1, 0])
            # else:
            #     obs.extend([0, 0, 1])

            # obs.extend(self.global_obs_dict['score'])  # 进球，不需要。
            # obs.extend(self.global_obs_dict['game_mode'])  # 小场景下固定。在全场中有脚球、自由、黄牌等阶段/模式。
            # obs.append(self.global_obs_dict['left_step'])

        #     obs_total.append(obs)
        #
        # self.obs = np.array(obs_total)  # 长度为智能体的数量，obs_total中每个元素是一个列表。

    # def get_obs_agent(self, agent_id):
    #     """Returns observation for agent_id."""
    #     return self.obs[agent_id].reshape(-1)

    def get_obs_size(self):
        """Returns the size of the observation."""
        return self.obs.shape[1]

    def get_global_state(self):

        global_state = []

        global_state.extend(self.global_obs_dict['left_team_position'].flatten())  # 我方绝对位置
        global_state.extend(self.global_obs_dict['left_team_velocity'].flatten())  # 我方绝对速度
        global_state.extend(self.global_obs_dict['left_team_tired'])  # 我方绝对疲劳
        for role in self.global_obs_dict['left_team_role']:  # 我方球员类型
            temp_role = np.array([0 for _ in range(10)])
            temp_role[role] = 1
            global_state.extend(temp_role)

        global_state.extend(self.global_obs_dict['right_team_position'].flatten())  # 敌方绝对位置
        global_state.extend(self.global_obs_dict['right_team_velocity'].flatten())  # 敌方绝对速度
        global_state.extend(self.global_obs_dict['right_team_tired'])  # 敌方绝对疲劳
        for role in self.global_obs_dict['right_team_role']:  # 敌方球员类型
            temp_role = np.array([0 for _ in range(10)])
            temp_role[role] = 1
            global_state.extend(temp_role)

        global_state.extend(self.global_obs_dict['ball_position'])  # 球的绝对位置
        global_state.extend(self.global_obs_dict['ball_velocity'])  # 球的绝对速度
        global_state.extend(self.global_obs_dict['ball_rotation'])  # 球的绝对转速
        # 持球队伍
        if self.global_obs_dict['ball_owned_team'] == 0:  # 0左队持球 -1无人持球 1右队持球
            global_state.extend([1, 0, 0])
        elif self.global_obs_dict['ball_owned_team'] == -1:
            global_state.extend([0, 1, 0])
        else:
            global_state.extend([0, 0, 1])
        # 持球人id
        temp_id = np.array([0 for _ in range(12)])
        if self.global_obs_dict['ball_owned_player'] == -1:
            temp_id[-1] = 1
        else:
            temp_id[self.global_obs_dict['ball_owned_player']] = 1
        global_state.extend(temp_id)

        self.state = global_state

        # global_state = []
        # for value in self.global_obs_dict.values():
        #     if len(value.shape) == 1:
        #         global_state.extend(value)
        #     elif len(value.shape) == 0:
        #         global_state.append(value)
        #     elif len(value.shape) > 1:
        #         global_state.extend(value.flatten())
        #
        # self.state = global_state

    # def get_state(self):
    #     """Returns the global state."""
    #     return self.get_global_state()

    def get_state_size(self):
        """Returns the size of the global state."""
        return len(self.state)

    def get_avail_actions(self):
        """Returns the available actions of all agents in a list."""
        return [[1 for _ in range(self.n_actions)] for agent_id in range(self.n_agents)]

    def get_avail_agent_actions(self, agent_id):
        """Returns the available actions for agent_id."""
        return self.get_avail_actions()[agent_id]

    def get_total_actions(self):
        """Returns the total number of actions an agent could ever take."""
        return self.action_space[0].n

    def reset(self):
        """Returns initial observations and states."""
        self.time_step = 0

        obs = self.env.reset()

        self.get_obs_dict(obs)
        self.get_obs()
        self.get_global_state()

        return self.obs, self.state

    def render(self):
        pass

    def close(self):
        self.env.close()

    def seed(self):
        pass

    def save_replay(self):
        """Save a replay."""
        pass

    def get_stats(self):
        return  {}

    def get_obs_dict(self, state):
        # 游戏全局状态
        self.global_obs_dict_last = copy.deepcopy(self.global_obs_dict)
        self.partial_obs_list_last = copy.deepcopy(self.partial_obs_list)
        self.partial_obs_list = []
        self.global_obs_dict['ball_position'] = np.array(state[0]['ball'])  # 球的位置
        self.global_obs_dict['ball_velocity'] = np.array(state[0]['ball_direction'])  # 球速度
        self.global_obs_dict['ball_rotation'] = np.array(state[0]['ball_rotation'])  # 球转速
        if 'ball_owned_team' not in self.global_obs_dict:
            self.global_obs_dict['ball_possession'] = np.array(state[0]['ball_owned_team'])
        else:
            if np.array(state[0]['ball_owned_team']) != -1 and np.array(state[0]['ball_owned_team']) != \
                    self.global_obs_dict['ball_possession']:
                self.global_obs_dict['ball_possession'] = np.array(state[0]['ball_owned_team'])
        self.global_obs_dict['ball_owned_team'] = np.array(state[0]['ball_owned_team'])  # 0左队持球 -1无人持球 1右队持球
        # self.global_obs_dict['ball_owned_player'] = np.eye(self.n_agents)[np.array(state[0]['ball_owned_player'])]  # 持球人id onehot
        self.global_obs_dict['ball_owned_player'] = np.array(state[0]['ball_owned_player'])  # 持球人id
        # self.global_obs_dict['score'] = np.array(state[0]['score'])  # 比分
        # self.global_obs_dict['left_step'] = np.array(state[0]['steps_left'])  # 剩余时间 3000-0
        # self.global_obs_dict['game_mode'] = np.array([0 for _ in range(7)])
        # self.global_obs_dict['game_mode'][int(state[0]['game_mode'])] = 1
        self.global_obs_dict['left_team_position'] = state[0]['left_team']
        self.global_obs_dict['left_team_velocity'] = state[0]['left_team_direction']
        self.global_obs_dict['left_team_tired'] = state[0]['left_team_tired_factor']
        self.global_obs_dict['left_team_role'] = state[0]['left_team_roles']
        # self.global_obs_dict['left_team_yellow_card'] = state[0]['left_team_yellow_card']
        self.global_obs_dict['right_team_position'] = state[0]['right_team']
        self.global_obs_dict['right_team_velocity'] = state[0]['right_team_direction']
        self.global_obs_dict['right_team_tired'] = state[0]['right_team_tired_factor']
        self.global_obs_dict['right_team_role'] = state[0]['right_team_roles']
        # self.global_obs_dict['right_team_yellow_card'] = state[0]['right_team_yellow_card']
        # self.global_obs_dict['agent_id'] = []
        # for i in range(len(state)):
        #     self.global_obs_dict['agent_id'].append(state[i]['active'])
        # self.global_obs_dict['agent_id'] = np.array(self.global_obs_dict['agent_id'])

        # 个人视角
        for i in range(self.n_agents):
            dic = {}
            dic['id'] = state[i]['active']  # one hot id: np.eye(self.n_agents+1)[state[i]['active']]
            dic['position'] = np.array(self.global_obs_dict['left_team_position'][dic['id']])
            dic['velocity'] = np.array(self.global_obs_dict['left_team_velocity'][dic['id']])
            dic['tired'] = self.global_obs_dict['left_team_tired'][dic['id']]
            dic['role'] = np.array([0 for _ in range(10)])
            dic['role'][self.global_obs_dict['left_team_role'][dic['id']]] = 1
            # dic['yellow_card'] = 1 if self.global_obs_dict['left_team_yellow_card'][dic['id']] else 0
            dic['sticky_actions'] = state[i]['sticky_actions']
            dic['ball_relative_position'] = self.global_obs_dict['ball_position'][:2] - dic['position']
            if self.global_obs_dict['ball_owned_team'] == 0 and self.global_obs_dict['ball_owned_player'] == dic['id']:
                dic['own_ball'] = 1
            else:
                dic['own_ball'] = 0
            dic['mate_relative_position_and_velocity'] = self.get_mate_relative_position_and_velocity(
                self.global_obs_dict['left_team_position'], self.global_obs_dict['left_team_velocity'], dic['id'])
            dic['enemy_relative_position_and_velocity'] = self.get_enemy_relative_position_and_velocity(
                self.global_obs_dict['right_team_position'], self.global_obs_dict['right_team_velocity'],
                dic['position'])
            self.partial_obs_list.append(dic)
        return

    def get_mate_relative_position_and_velocity(self, team_position, team_velocity, id, ):
        if self.position_rank:
            team_relative_pos_and_velocity = []
            team_pos = team_position
            team_pos = np.array(team_pos) - np.array(team_position[id])
            team_dis = {str(i): np.linalg.norm(team_pos[i]) for i in range(len(team_pos)) if i != id}
            team_dis_list = [value for value in team_dis.values()]
            team_dis_list.sort()
            for i in range(len(team_dis_list)):
                index = int([ind for ind in team_dis.keys() if team_dis[ind] == team_dis_list[i]][0])
                team_relative_pos_and_velocity.extend(team_pos[index])
                team_relative_pos_and_velocity.extend(team_velocity[index])
        else:
            team_relative_pos_and_velocity = []
            team_pos = np.array(team_position) - np.array(team_position[id])
            for i in range(len(team_pos)):
                team_relative_pos_and_velocity.extend(team_pos[i])
                team_relative_pos_and_velocity.extend(team_velocity[i])
        return team_relative_pos_and_velocity

    def get_enemy_relative_position_and_velocity(self, team_position, team_velocity, position):
        if self.position_rank:
            team_relative_pos_and_velocity = []
            team_pos = team_position
            team_pos = np.array(team_pos) - np.array(position)
            team_dis = {str(i): np.linalg.norm(team_pos[i]) for i in range(len(team_pos))}
            team_dis_list = [value for value in team_dis.values()]
            team_dis_list.sort()
            for i in range(len(team_dis_list)):
                index = int([ind for ind in team_dis.keys() if team_dis[ind] == team_dis_list[i]][0])
                team_relative_pos_and_velocity.extend(team_pos[index])
                team_relative_pos_and_velocity.extend(team_velocity[index])
        else:
            team_relative_pos_and_velocity = []
            team_pos = np.array(team_position) - np.array(position)
            for i in range(len(team_pos)):
                team_relative_pos_and_velocity.extend(team_pos[i])
                team_relative_pos_and_velocity.extend(team_velocity[i])
        return team_relative_pos_and_velocity


# class GoogleFootballEnv(MultiAgentEnv):
#
#     def __init__(
#         self,
#         dense_reward=False,
#         write_full_episode_dumps=False,
#         write_goal_dumps=False,
#         dump_freq=0,
#         render=False,
#         num_agents=4,
#         time_limit=200,
#         time_step=0,
#         map_name='academy_counterattack_hard',
#         stacked=False,
#         representation="simple115",
#         rewards='scoring,checkpoints',
#         logdir='football_dumps',
#         write_video=False,
#         number_of_right_players_agent_controls=0,
#         seed=0,
#     ):
#         self.dense_reward = dense_reward
#         self.write_full_episode_dumps = write_full_episode_dumps
#         self.write_goal_dumps = write_goal_dumps
#         self.dump_freq = dump_freq
#         self.render = render
#         self.n_agents = num_agents
#         self.episode_limit = time_limit
#         self.time_step = time_step
#         self.env_name = map_name
#         self.stacked = stacked
#         self.representation = representation
#         self.rewards = rewards
#         self.logdir = logdir
#         self.write_video = write_video
#         self.number_of_right_players_agent_controls = number_of_right_players_agent_controls
#         self.seed = seed
#
#         self.env = football_env.create_environment(
#             write_full_episode_dumps=self.write_full_episode_dumps,
#             write_goal_dumps=self.write_goal_dumps,
#             env_name=self.env_name,
#             stacked=self.stacked,
#             representation=self.representation,
#             rewards=self.rewards,
#             logdir=self.logdir,
#             render=self.render,
#             write_video=self.write_video,
#             dump_frequency=self.dump_freq,
#             number_of_left_players_agent_controls=self.n_agents,
#             number_of_right_players_agent_controls=self.number_of_right_players_agent_controls,
#             channel_dimensions=(observation_preprocessing.SMM_WIDTH, observation_preprocessing.SMM_HEIGHT))
#         self.env.seed(self.seed)
#
#         obs_space_low = self.env.observation_space.low[0]
#         obs_space_high = self.env.observation_space.high[0]
#
#         self.action_space = [gym.spaces.Discrete(
#             self.env.action_space.nvec[1]) for _ in range(self.n_agents)]
#         self.observation_space = [
#             gym.spaces.Box(low=obs_space_low, high=obs_space_high, dtype=self.env.observation_space.dtype) for _ in range(self.n_agents)
#         ]
#
#         self.n_actions = self.action_space[0].n
#         self.obs = None
#
#     def step(self, _actions):
#         """Returns reward, terminated, info."""
#         if th.is_tensor(_actions):
#             actions = _actions.cpu().numpy()
#         else:
#             actions = _actions
#         self.time_step += 1
#         obs, rewards, done, infos = self.env.step(actions.tolist())
#
#         self.obs = obs
#
#         if self.time_step >= self.episode_limit:
#             done = True
#
#         return sum(rewards), done, infos
#
#     def get_obs(self):
#         """Returns all agent observations in a list."""
#         return self.obs.reshape(self.n_agents, -1)
#
#     def get_obs_agent(self, agent_id):
#         """Returns observation for agent_id."""
#         return self.obs[agent_id].reshape(-1)
#
#     def get_obs_size(self):
#         """Returns the size of the observation."""
#         obs_size = np.array(self.env.observation_space.shape[1:])
#         return int(obs_size.prod())
#
#     def get_global_state(self):
#         return self.obs.flatten()
#
#     def get_state(self):
#         """Returns the global state."""
#         return self.get_global_state()
#
#     def get_state_size(self):
#         """Returns the size of the global state."""
#         return self.get_obs_size() * self.n_agents
#
#     def get_avail_actions(self):
#         """Returns the available actions of all agents in a list."""
#         return [[1 for _ in range(self.n_actions)] for agent_id in range(self.n_agents)]
#
#     def get_avail_agent_actions(self, agent_id):
#         """Returns the available actions for agent_id."""
#         return self.get_avail_actions()[agent_id]
#
#     def get_total_actions(self):
#         """Returns the total number of actions an agent could ever take."""
#         return self.action_space[0].n
#
#     def reset(self):
#         """Returns initial observations and states."""
#         self.time_step = 0
#         self.obs = self.env.reset()
#
#         return self.get_obs(), self.get_global_state()
#
#     def render(self):
#         pass
#
#     def close(self):
#         self.env.close()
#
#     def seed(self):
#         pass
#
#     def save_replay(self):
#         """Save a replay."""
#         pass
#
#     def get_stats(self):
#         return  {}
