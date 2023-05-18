import numpy as np
import copy
import random
import torch
import matplotlib.pyplot as plt


def seed(num):
    torch.manual_seed(num)
    torch.cuda.manual_seed(num)
    torch.cuda.manual_seed_all(num)
    np.random.seed(num)
    random.seed(num)


def clip(x, ma, mi):
    if x > ma:
        x = ma
    if x < mi:
        x = mi
    return x


def layer(i):
    if i == 1 or i ==2:
        return 132
    if i == 3:
        return 64
    if i == 4:
        return 32


class Scenario(object):
    def __init__(self, all_arg):
        self.env_name = 'academy_3_vs_1_with_keeper'
        self.representation = 'raw'
        self.number_of_left_players_agent_controls = 3
        self.number_of_right_players_agent_controls = 0
        self.global_obs_dict_last = {}
        self.global_obs_dict = {}
        self.partial_obs_list_last = []
        self.partial_obs_list = []
        self.eg_list = np.load('/home/eg_list.npy', allow_pickle=True)
        self.eg = 0
        self.position_rank = True
        self.use_eg = all_arg.use_eg
        self.eg_layer = all_arg.eg_layer
        self.gamma = all_arg.gamma

    def seed(self, seed_num):
        seed(seed_num)

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
        self.global_obs_dict['ball_owned_player'] = np.array(state[0]['ball_owned_player'])  # 持球人id
        self.global_obs_dict['score'] = np.array(state[0]['score'])  # 比分
        self.global_obs_dict['left_step'] = np.array(state[0]['steps_left'])  # 剩余时间 3000-0
        self.global_obs_dict['game_mode'] = np.array([0 for _ in range(7)])
        self.global_obs_dict['game_mode'][int(state[0]['game_mode'])] = 1
        self.global_obs_dict['left_team_position'] = state[0]['left_team']
        self.global_obs_dict['left_team_velocity'] = state[0]['left_team_direction']
        self.global_obs_dict['left_team_tired'] = state[0]['left_team_tired_factor']
        self.global_obs_dict['left_team_role'] = state[0]['left_team_roles']
        self.global_obs_dict['left_team_yellow_card'] = state[0]['left_team_yellow_card']
        self.global_obs_dict['right_team_position'] = state[0]['right_team']
        self.global_obs_dict['right_team_velocity'] = state[0]['right_team_direction']
        self.global_obs_dict['right_team_tired'] = state[0]['right_team_tired_factor']
        self.global_obs_dict['right_team_role'] = state[0]['right_team_roles']
        self.global_obs_dict['right_team_yellow_card'] = state[0]['right_team_yellow_card']
        self.global_obs_dict['agent_id'] = []
        for i in range(len(state)):
            self.global_obs_dict['agent_id'].append(state[i]['active'])

        # 个人视角
        for i in range(self.number_of_left_players_agent_controls):
            dic = {}
            dic['id'] = state[i]['active']
            dic['position'] = np.array(self.global_obs_dict['left_team_position'][dic['id']])
            dic['velocity'] = np.array(self.global_obs_dict['left_team_velocity'][dic['id']])
            dic['tired'] = self.global_obs_dict['left_team_tired'][dic['id']]
            dic['role'] = np.array([0 for _ in range(10)])
            dic['role'][self.global_obs_dict['left_team_role'][dic['id']]] = 1
            dic['yellow_card'] = 1 if self.global_obs_dict['left_team_yellow_card'][dic['id']] else 0
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

    def plot(self):
        for p in self.global_obs_dict['left_team_position']:
            plt.scatter(p[0], p[1], color='blue')
        for p in self.global_obs_dict['right_team_position']:
            plt.scatter(p[0], p[1], color='red')
        plt.scatter(self.global_obs_dict['ball_position'][0], self.global_obs_dict['ball_position'][1], color='black')
        if self.global_obs_dict['ball_owned_team'] == 0:
            plt.ylabel('home')
        if self.global_obs_dict['ball_owned_team'] == 1:
            plt.ylabel('away')
        if self.global_obs_dict['ball_owned_team'] == -1:
            plt.ylabel('none')
        plt.show()

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

    def feature(self):  # obs
        obs_total = []
        for i in range(self.number_of_left_players_agent_controls):
            obs = []
            obs.extend(self.partial_obs_list[i]['position'])
            obs.extend(self.partial_obs_list[i]['velocity'])
            obs.append(self.partial_obs_list[i]['tired'])
            # obs.extend(self.partial_obs_list[i]['role'])
            obs.extend(self.partial_obs_list[i]['sticky_actions'])
            obs.append(self.partial_obs_list[i]['yellow_card'])
            obs.extend(self.partial_obs_list[i]['ball_relative_position'])
            obs.append(self.partial_obs_list[i]['own_ball'])
            obs.extend(self.partial_obs_list[i]['mate_relative_position_and_velocity'])
            obs.extend(self.partial_obs_list[i]['enemy_relative_position_and_velocity'])
            obs.extend(self.global_obs_dict['ball_position'])
            obs.extend(self.global_obs_dict['ball_velocity'])
            obs.extend(self.global_obs_dict['ball_rotation'])
            if self.global_obs_dict['ball_owned_team'] == 0:
                obs.extend([1, 0, 0])
            elif self.global_obs_dict['ball_owned_team'] == -1:
                obs.extend([0, 1, 0])
            else:
                obs.extend([0, 0, 1])
            obs.extend(self.global_obs_dict['score'])
            obs.extend(self.global_obs_dict['game_mode'])
            obs.append(self.global_obs_dict['left_step'])
            obs_total.append(obs)
        return obs_total

    def reward(self, rew, reward, factor):
        gamma = self.gamma
        m = len(self.eg_list)
        n = len(self.eg_list[0])
        if reward == 'score' or reward == 'checkpoint':
            return [[rew[i]] for i in range(self.number_of_left_players_agent_controls)]
        elif reward == 'eg':
            r = [[rew[i]] for i in range(self.number_of_left_players_agent_controls)]

            if self.global_obs_dict['ball_possession'] == 0 and self.global_obs_dict_last['ball_possession'] == 1:  # 奖励
                for i in range(len(r)):
                    r[i][0] += gamma * 0.2 * factor
            if self.global_obs_dict['ball_possession'] == 1 and self.global_obs_dict_last['ball_possession'] == 0:  # 惩罚
                for i in range(len(r)):
                    r[i][0] -= gamma * 0.2 * factor

            x = int((self.global_obs_dict['ball_position'][0] + 1) * m / 2)
            y = int(abs(self.global_obs_dict['ball_position'][1] * n / 0.42))
            x = clip(x, m - 1, 0)
            y = clip(y, n - 1, 0)
            for i in range(len(r)):
                r[i][0] += (gamma * self.eg_list[x][y][0] - self.eg) * factor
            self.eg = self.eg_list[x][y][0]
        elif reward == 'x':
            r = [[rew[i]] for i in range(self.number_of_left_players_agent_controls)]
            for i in range(len(r)):
                r[i][0] += (gamma * self.global_obs_dict['ball_position'][0] -
                            self.global_obs_dict_last['ball_position'][0]) * factor
        return r

    def get_available_action(self):
        available_actions = []
        for i in range(self.number_of_left_players_agent_controls):
            # sticky_actions：10个粘滞动作的状态0left, 1top_left, 2top, 3top_right, 4right, 5bottom_right, 6bottom, 7bottom_left, 8sprint, 9dibble(运球)
            # action_idle = 0, a no-op action, sticky actions are not affected (player maintains his directional movement etc.).
            # action_left = 1, run to the left, sticky action.
            # action_top_left = 2, run to the top-left, sticky action.
            # action_top = 3, run to the top, sticky action.
            # action_top_right = 4, run to the top-right, sticky action.
            # action_right = 5, run to the right, sticky action.
            # action_bottom_right = 6, run to the bottom-right, sticky action.
            # action_bottom = 7, run to the bottom, sticky action.
            # action_bottom_left = 8, run to the bottom-left, sticky action.
            # action_long_pass = 9, perform a long pass to the player on your team. Player to pass the ball to is auto-determined based on the movement direction.
            # action_high_pass = 10, perform a high pass, similar to action_long_pass.
            # action_short_pass = 11, perform a short pass, similar to action_long_pass.
            # action_shot = 12, perform a shot, always in the direction of the opponent's goal.
            # action_sprint = 13, start sprinting, sticky action. Player moves faster, but has worse ball handling.
            # action_release_direction = 14, reset current movement direction.
            # action_release_sprint = 15, stop sprinting.
            # action_sliding = 16, perform a slide (effective when not having a ball).
            # action_dribble = 17, start dribbling (effective when having a ball), sticky action. Player moves slower, but it is harder to take over the ball from him.
            # action_release_dribble = 18, stop dribbling.
            available_action = np.array([0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])  # 去掉不作为
            idle, left, top_left, top, top_right, right, bottom_right, bottom, bottom_left, long_pass, high_pass = 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
            short_pass, shot, sprint, release_direction, release_sprint, sliding, dribble, release_dribble = 11, 12, 13, 14, 15, 16, 17, 18
            # 粘滞动作屏蔽
            state = self.partial_obs_list[i]

            sticky_actions = state['sticky_actions']
            if sticky_actions[8] == 0:  # 没有在冲刺哦
                available_action[release_sprint] = 0
            if sticky_actions[9] == 0:  # 没有运球
                available_action[release_dribble] = 0
            if np.sum(sticky_actions[: 8]) == 0:  # 没有在跑
                available_action[release_direction] = 0

            # 持球屏蔽
            if self.global_obs_dict['ball_owned_team'] == 0:  # 我方持球
                available_action[sliding] = 0
            # 远球
            if np.linalg.norm(state['ball_relative_position']) > 0.05:  # 我离球很远
                available_action[long_pass] = 0
                available_action[high_pass] = 0
                available_action[short_pass] = 0
                available_action[shot] = 0
                available_action[sliding] = 0
                available_action[dribble] = 0
            # if state['own_ball'] == 0:  # 我没有持球
                # available_action[dribble] = 0
            # 远距离射门屏蔽
            if state['position'][0] < 0.7:
                available_action[shot] = 0
            if state['position'][0] < - 0.9:
                available_action[left] = 0
                available_action[bottom_left] = 0
                available_action[top_left] = 0
            if state['position'][0] > 0.9:
                available_action[right] = 0
                available_action[bottom_right] = 0
                available_action[top_right] = 0
            if state['position'][1] < -0.38:
                available_action[top_left] = 0
                available_action[top_right] = 0
                available_action[top] = 0
            if state['position'][1] > 0.38:
                available_action[bottom_left] = 0
                available_action[bottom_right] = 0
                available_action[bottom] = 0
            if np.linalg.norm(state['velocity']) < 0.01:
                available_action[sprint] = 0
            available_actions.append(available_action)
        return available_actions

    def global_process(self):
        attribute = []
        attribute.extend(self.global_obs_dict['left_team_position'].flatten())  # position of left team, x:[-1, 1], y:[-0.42, 0.42]
        attribute.extend(self.global_obs_dict['left_team_velocity'].flatten())  # speed of left team
        attribute.extend(self.global_obs_dict['left_team_tired'])  # tired factor of left team [0, 1]
        attribute.extend(self.global_obs_dict['right_team_position'].flatten())  # position of right team
        attribute.extend(self.global_obs_dict['right_team_velocity'].flatten())  # speed of right team
        attribute.extend(self.global_obs_dict['right_team_tired'])  # tired factor of right team
        attribute.extend(self.global_obs_dict['ball_position'])  # ball position
        attribute.extend(self.global_obs_dict['ball_velocity'])  # ball speed
        attribute.extend(self.global_obs_dict['ball_rotation'])  # ball rotation
        if self.global_obs_dict['ball_owned_team'] == 0:  # ball_owned_team to one-hot
            attribute.extend([1, 0, 0])  # left
        if self.global_obs_dict['ball_owned_team'] == -1:  #
            attribute.extend([0, 1, 0])  # none
        if self.global_obs_dict['ball_owned_team'] == 1:  #
            attribute.extend([0, 0, 1])  # right
        attribute.extend(self.global_obs_dict['score'])  # score
        attribute.extend([self.global_obs_dict['left_step']])  # [3000, 0]
        attribute.extend(self.global_obs_dict['game_mode'])
        return attribute



class Scenario3v1(Scenario):
    def __init__(self, all_arg):
        super(Scenario3v1, self).__init__(all_arg)
        self.env_name = 'academy_3_vs_1_with_keeper'
        self.representation = 'raw'
        self.number_of_left_players_agent_controls = 3
        self.number_of_right_players_agent_controls = 0
        self.observation_space = 61
        self.share_observation_space = 61

class ScenarioPass(Scenario):
    def __init__(self, all_arg):
        super(ScenarioPass, self).__init__(all_arg)
        self.env_name = 'academy_pass_and_shoot_with_keeper'
        self.representation = 'raw'
        self.number_of_left_players_agent_controls = 2
        self.number_of_right_players_agent_controls = 0
        self.observation_space = 57
        self.share_observation_space = 57

class ScenarioRunPass(Scenario):
    def __init__(self, all_arg):
        super(ScenarioRunPass, self).__init__(all_arg)
        self.env_name = 'academy_run_pass_and_shoot_with_keeper'
        self.representation = 'raw'
        self.number_of_left_players_agent_controls = 2
        self.number_of_right_players_agent_controls = 0
        self.observation_space = 57
        self.share_observation_space = 57



class Scenario5v5(Scenario):
    def __init__(self):
        super(Scenario5v5, self).__init__()
        self.env_name = '5_vs_5'
        self.representation = 'raw'
        self.number_of_left_players_agent_controls = 4
        self.number_of_right_players_agent_controls = 0
        self.observation_space = 69
        self.share_observation_space = 69
        self.position_rank = False

    def feature(self):
        obs_total = []
        for i in range(self.number_of_left_players_agent_controls):
            obs = []
            obs.extend(self.partial_obs_list[i]['position'])
            # obs.extend(self.partial_obs_list[i]['velocity'])
            obs.append(self.partial_obs_list[i]['tired'])
            # obs.extend(self.partial_obs_list[i]['role'])
            obs.extend(self.partial_obs_list[i]['sticky_actions'])
            # obs.append(self.partial_obs_list[i]['yellow_card'])
            obs.extend(self.partial_obs_list[i]['ball_relative_position'])
            obs.append(self.partial_obs_list[i]['own_ball'])
            obs.extend(self.partial_obs_list[i]['mate_relative_position_and_velocity'])
            obs.extend(self.partial_obs_list[i]['enemy_relative_position_and_velocity'])
            obs.extend(self.global_obs_dict['ball_position'])
            obs.extend(self.global_obs_dict['ball_velocity'])
            obs.extend(self.global_obs_dict['ball_rotation'])
            if self.global_obs_dict['ball_owned_team'] == 0:
                obs.extend([1, 0, 0])
            elif self.global_obs_dict['ball_owned_team'] == -1:
                obs.extend([0, 1, 0])
            else:
                obs.extend([0, 0, 1])
            # obs.extend(self.global_obs_dict['score'])
            # obs.extend(self.global_obs_dict['game_mode'])
            obs.append(self.global_obs_dict['left_step'])
            obs_total.append(obs)
        return obs_total

    def reward(self, rew, reward, factor):
        gamma = 0.99
        m = len(self.eg_list)
        n = len(self.eg_list[0])
        if rew[0] == 1:
            r = [[1], [1], [0.2], [0.2]]
        elif rew[0] == -1:
            r = [[-0.2], [-0.2], [-1], [-1]]
        else:
            r = [[rew[i]] for i in range(self.number_of_left_players_agent_controls)]
        if reward == 'score' or reward == 'checkpoint':
            return r
        elif reward == 'eg':
            if self.global_obs_dict['ball_possession'] == 0 and self.global_obs_dict_last['ball_possession'] == 1:  # 奖励
                for i in range(len(r)):
                    r[i][0] += gamma * 0.2 * factor
            if self.global_obs_dict['ball_possession'] == 1 and self.global_obs_dict_last['ball_possession'] == 0:  # 惩罚
                for i in range(len(r)):
                    r[i][0] -= gamma * 0.2 * factor

            x = int((self.global_obs_dict['ball_position'][0] + 1) * m / 2)
            y = int(abs(self.global_obs_dict['ball_position'][1] * n / 0.42))
            x = clip(x, m - 1, 0)
            y = clip(y, n - 1, 0)
            for i in range(len(r)):
                r[i][0] += (gamma * self.eg_list[x][y][0] - self.eg) * factor
            self.eg = self.eg_list[x][y][0]
        elif reward == 'x':
            r = [[rew[i]] for i in range(self.number_of_left_players_agent_controls)]
            for i in range(len(r)):
                r[i][0] += (gamma * self.global_obs_dict['ball_position'][0] -
                            self.global_obs_dict_last['ball_position'][0]) * factor
        return r


class ScenarioCounterEasy(Scenario):
    def __init__(self):
        super(ScenarioCounterEasy, self).__init__()
        self.env_name = 'academy_counterattack_easy'
        self.representation = 'raw'
        self.number_of_left_players_agent_controls = 4
        self.number_of_right_players_agent_controls = 0
        self.observation_space = 115
        self.share_observation_space = 115

    def feature(self):
        obs_total = []
        for i in range(self.number_of_left_players_agent_controls):
            obs = []
            obs.extend(self.partial_obs_list[i]['position'])
            obs.extend(self.partial_obs_list[i]['velocity'])
            obs.append(self.partial_obs_list[i]['tired'])
            # obs.extend(self.partial_obs_list[i]['role'])
            obs.extend(self.partial_obs_list[i]['sticky_actions'])
            # obs.append(self.partial_obs_list[i]['yellow_card'])
            obs.extend(self.partial_obs_list[i]['ball_relative_position'])
            obs.append(self.partial_obs_list[i]['own_ball'])
            obs.extend(self.partial_obs_list[i]['mate_relative_position_and_velocity'])
            obs.extend(self.partial_obs_list[i]['enemy_relative_position_and_velocity'])
            obs.extend(self.global_obs_dict['ball_position'])
            obs.extend(self.global_obs_dict['ball_velocity'])
            obs.extend(self.global_obs_dict['ball_rotation'])
            if self.global_obs_dict['ball_owned_team'] == 0:
                obs.extend([1, 0, 0])
            elif self.global_obs_dict['ball_owned_team'] == -1:
                obs.extend([0, 1, 0])
            else:
                obs.extend([0, 0, 1])
            # obs.extend(self.global_obs_dict['score'])
            # obs.extend(self.global_obs_dict['game_mode'])
            obs.append(self.global_obs_dict['left_step'])
            obs_total.append(obs)
        return obs_total

class ScenarioCounterHard(Scenario):
    def __init__(self, all_arg):
        super(ScenarioCounterHard, self).__init__(all_arg)
        self.env_name = 'academy_counterattack_hard'
        self.representation = 'raw'
        self.number_of_left_players_agent_controls = 4
        self.number_of_right_players_agent_controls = 0
        self.use_eg = all_arg.use_eg
        self.eg_layer = all_arg.eg_layer
        if self.use_eg:
            self.observation_space = 22 + layer(self.eg_layer)
            self.share_observation_space = layer(self.eg_layer)
        else:
            self.observation_space = 22 + 132
            self.share_observation_space = 22 + 132

    def feature(self):
        obs_total = []
        if self.use_eg:
            for i in range(self.number_of_left_players_agent_controls):
                # obs = list(self.net.activation3_out(torch.tensor(self.global_process()).to(torch.float32)))
                # obs = self.global_process()
                # for j in range(len(obs)):
                # obs[j] = obs[j].item()
                obs = []
                obs.extend(self.partial_obs_list[i]['position'])
                obs.extend(self.partial_obs_list[i]['velocity'])
                obs.append(self.partial_obs_list[i]['tired'])
                obs.extend(self.partial_obs_list[i]['ball_relative_position'])
                obs.append(self.partial_obs_list[i]['own_ball'])
                # obs.extend(self.partial_obs_list[i]['role'])
                obs.extend(self.partial_obs_list[i]['sticky_actions'])
                obs.append(self.global_obs_dict['ball_position'][2])
                if self.global_obs_dict['ball_owned_team'] == 0:
                    obs.extend([1, 0, 0])
                elif self.global_obs_dict['ball_owned_team'] == -1:
                    obs.extend([0, 1, 0])
                else:
                    obs.extend([0, 0, 1])
                obs_total.append(obs)
        else:
            for i in range(self.number_of_left_players_agent_controls):
                # obs = list(self.net.activation3_out(torch.tensor(self.global_process()).to(torch.float32)))
                # obs = self.global_process()
                # for j in range(len(obs)):
                # obs[j] = obs[j].item()
                obs = self.global_process()
                obs.extend(self.partial_obs_list[i]['position'])
                obs.extend(self.partial_obs_list[i]['velocity'])
                obs.append(self.partial_obs_list[i]['tired'])
                obs.extend(self.partial_obs_list[i]['ball_relative_position'])
                obs.append(self.partial_obs_list[i]['own_ball'])
                # obs.extend(self.partial_obs_list[i]['role'])
                obs.extend(self.partial_obs_list[i]['sticky_actions'])
                obs.append(self.global_obs_dict['ball_position'][2])
                if self.global_obs_dict['ball_owned_team'] == 0:
                    obs.extend([1, 0, 0])
                elif self.global_obs_dict['ball_owned_team'] == -1:
                    obs.extend([0, 1, 0])
                else:
                    obs.extend([0, 0, 1])
                obs_total.append(obs)
        return obs_total

class ScenarioCorner(Scenario):
    def __init__(self, all_arg):
        super(ScenarioCorner, self).__init__(all_arg)
        self.env_name = 'academy_corner'
        self.representation = 'raw'
        self.number_of_left_players_agent_controls = 11
        self.number_of_right_players_agent_controls = 0
        self.use_eg = all_arg.use_eg
        self.eg_layer = all_arg.eg_layer
        if self.use_eg:
            self.observation_space = 22 + layer(self.eg_layer)
            self.share_observation_space = layer(self.eg_layer)
        else:
            self.observation_space = 22 + 132
            self.share_observation_space = 22 + 132

    def feature(self):
        obs_total = []
        if self.use_eg:
            for i in range(self.number_of_left_players_agent_controls):
                # obs = list(self.net.activation3_out(torch.tensor(self.global_process()).to(torch.float32)))
                # obs = self.global_process()
                # for j in range(len(obs)):
                # obs[j] = obs[j].item()
                obs = []
                obs.extend(self.partial_obs_list[i]['position'])
                obs.extend(self.partial_obs_list[i]['velocity'])
                obs.append(self.partial_obs_list[i]['tired'])
                obs.extend(self.partial_obs_list[i]['ball_relative_position'])
                obs.append(self.partial_obs_list[i]['own_ball'])
                # obs.extend(self.partial_obs_list[i]['role'])
                obs.extend(self.partial_obs_list[i]['sticky_actions'])
                obs.append(self.global_obs_dict['ball_position'][2])
                if self.global_obs_dict['ball_owned_team'] == 0:
                    obs.extend([1, 0, 0])
                elif self.global_obs_dict['ball_owned_team'] == -1:
                    obs.extend([0, 1, 0])
                else:
                    obs.extend([0, 0, 1])
                obs_total.append(obs)
        else:
            for i in range(self.number_of_left_players_agent_controls):
                # obs = list(self.net.activation3_out(torch.tensor(self.global_process()).to(torch.float32)))
                # obs = self.global_process()
                # for j in range(len(obs)):
                # obs[j] = obs[j].item()
                obs = self.global_process()
                obs.extend(self.partial_obs_list[i]['position'])
                obs.extend(self.partial_obs_list[i]['velocity'])
                obs.append(self.partial_obs_list[i]['tired'])
                obs.extend(self.partial_obs_list[i]['ball_relative_position'])
                obs.append(self.partial_obs_list[i]['own_ball'])
                # obs.extend(self.partial_obs_list[i]['role'])
                obs.extend(self.partial_obs_list[i]['sticky_actions'])
                obs.append(self.global_obs_dict['ball_position'][2])
                if self.global_obs_dict['ball_owned_team'] == 0:
                    obs.extend([1, 0, 0])
                elif self.global_obs_dict['ball_owned_team'] == -1:
                    obs.extend([0, 1, 0])
                else:
                    obs.extend([0, 0, 1])
                obs_total.append(obs)
        return obs_total


class Scenario11v11(Scenario):
    '''
            0守门员
            1中后卫  2左后卫  3右后卫 4防守中锋
            5中心中锋 6左中锋 7右中锋
            8进攻中场 9前锋
            left_team_roles : 0,7,9,2,1,1,3,5,5,5,6。守，攻，攻，防，防，防，防，中，中，中，攻
    '''
    def __init__(self, all_arg):
        super(Scenario11v11, self).__init__(all_arg)
        self.env_name = '11_vs_11_easy_stochastic'
        self.representation = 'raw'
        self.number_of_left_players_agent_controls = all_arg.num_agents
        self.number_of_right_players_agent_controls = 0
        self.use_eg = all_arg.use_eg
        self.eg_layer = all_arg.eg_layer
        self.position_rank = False
        if self.use_eg:
            self.observation_space = 32 + layer(self.eg_layer)
            self.share_observation_space = layer(self.eg_layer)
        else:
            self.observation_space = 32 + 132
            self.share_observation_space = 32 + 132

    def feature(self):
        obs_total = []
        if self.use_eg:
            for i in range(self.number_of_left_players_agent_controls):
                obs = []
                obs.extend(self.partial_obs_list[i]['position'])
                obs.extend(self.partial_obs_list[i]['velocity'])
                obs.append(self.partial_obs_list[i]['tired'])
                obs.extend(self.partial_obs_list[i]['ball_relative_position'])
                obs.append(self.partial_obs_list[i]['own_ball'])
                obs.extend(self.partial_obs_list[i]['role'])
                obs.extend(self.partial_obs_list[i]['sticky_actions'])
                obs.append(self.global_obs_dict['ball_position'][2])
                if self.global_obs_dict['ball_owned_team'] == 0:
                    obs.extend([1, 0, 0])
                elif self.global_obs_dict['ball_owned_team'] == -1:
                    obs.extend([0, 1, 0])
                else:
                    obs.extend([0, 0, 1])
                obs_total.append(obs)
        else:
            for i in range(self.number_of_left_players_agent_controls):
                # obs = list(self.net.activation3_out(torch.tensor(self.global_process()).to(torch.float32)))
                # obs = self.global_process()
                # for j in range(len(obs)):
                # obs[j] = obs[j].item()
                obs = self.global_process()
                obs.extend(self.partial_obs_list[i]['position'])
                obs.extend(self.partial_obs_list[i]['velocity'])
                obs.append(self.partial_obs_list[i]['tired'])
                obs.extend(self.partial_obs_list[i]['ball_relative_position'])
                obs.append(self.partial_obs_list[i]['own_ball'])
                obs.extend(self.partial_obs_list[i]['role'])
                obs.extend(self.partial_obs_list[i]['sticky_actions'])
                obs.append(self.global_obs_dict['ball_position'][2])
                if self.global_obs_dict['ball_owned_team'] == 0:
                    obs.extend([1, 0, 0])
                elif self.global_obs_dict['ball_owned_team'] == -1:
                    obs.extend([0, 1, 0])
                else:
                    obs.extend([0, 0, 1])
                obs_total.append(obs)
        return obs_total
    '''
    def reward(self, rew, reward, factor):
        gamma = 0.99
        m = len(self.eg_list)
        n = len(self.eg_list[0])
        r = [[0] for _ in range(11)]
        # 攻防分配
        if rew[0] > 0:
            r[0][0] += 0
            r[1][0] += 1
            r[2][0] += 1
            r[3][0] += 0.2
            r[4][0] += 0.2
            r[5][0] += 0.2
            r[6][0] += 0.2
            r[7][0] += 0.6
            r[8][0] += 0.6
            r[9][0] += 0.6
            r[10][0] += 1
        if rew[0] < 0:
            r[0][0] -= 1.5
            r[1][0] -= 0.2
            r[2][0] -= 0.2
            r[3][0] -= 1
            r[4][0] -= 1
            r[5][0] -= 1
            r[6][0] -= 1
            r[7][0] -= 0.6
            r[8][0] -= 0.6
            r[9][0] -= 0.6
            r[10][0] -= 0.2
        # 守门员边界约束
        if self.partial_obs_list[0]['position'][0] > -0.835 or self.partial_obs_list[0]['position'][1] > 0.2:
            r[0][0] -= 0.001
        # 持球奖惩
        if self.global_obs_dict['ball_possession'] == 0 and self.global_obs_dict_last['ball_possession'] == 1:  # 奖励
            for i in range(len(r)):
                r[i][0] += gamma * 0.2 * factor
        if self.global_obs_dict['ball_possession'] == 1 and self.global_obs_dict_last['ball_possession'] == 0:  # 惩罚
            for i in range(len(r)):
                r[i][0] -= gamma * 0.2 * factor
        # eg奖励
        x = int((self.global_obs_dict['ball_position'][0] + 1) * m / 2)
        y = int(abs(self.global_obs_dict['ball_position'][1] * n / 0.42))
        x = clip(x, m - 1, 0)
        y = clip(y, n - 1, 0)
        for i in range(len(r)):
            r[i][0] += (gamma * self.eg_list[x][y][0] - self.eg) * factor
        self.eg = self.eg_list[x][y][0]
        rr = []
        for id in self.global_obs_dict['agent_id']:
            rr.append(r[id])
        return rr
    '''
