from envs import REGISTRY as env_REGISTRY
from functools import partial
from components.episode_buffer import EpisodeBatch
from utils.risk import get_risk_q, get_risk_q_mode
import numpy as np
import torch as th
import time

class EpisodeRunner:

    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.batch_size = self.args.batch_size_run
        assert self.batch_size == 1

        self.env = env_REGISTRY[self.args.env](**self.args.env_args)
        self.episode_limit = self.env.episode_limit
        self.t = 0

        self.t_env = 0

        self.train_returns = []
        self.test_returns = []
        self.train_fails = [] #measure the failure events of risk-sensitive environments  20230108
        self.test_fails = [] #measure the failure events of risk-sensitive environments   20230108
        self.train_stats = {}
        self.test_stats = {}
        self.train_win_rates = []
        self.test_win_rates = []
        # self.train_battle_wons = []
        # self.test_battle_wons = []

        # Log the first run
        self.log_train_stats_t = -1000000

        self.inference_time = 0

    def setup(self, scheme, groups, preprocess, mac):
        self.new_batch = partial(EpisodeBatch, scheme, groups, self.batch_size, self.episode_limit + 1,
                                 preprocess=preprocess, device=self.args.device)
        self.mac = mac

    def get_env_info(self):
        return self.env.get_env_info()

    def save_replay(self):
        self.env.save_replay()

    def close_env(self):
        self.env.close()

    def reset(self):
        self.batch = self.new_batch()
        self.env.reset()
        self.t = 0

    def run(self, test_mode=False):
        self.reset()

        terminated = False
        episode_return = 0
        self.mac.init_hidden(batch_size=self.batch_size)
        
        map_episode_limit = self.episode_limit + 1 # 防止越界
        # 保存轨迹
        trajectory = {
            'actions': np.zeros((map_episode_limit,self.env.n_agents, 1)),
            'actions_onehot': np.zeros((map_episode_limit, self.env.n_agents, self.env.n_actions)),
            'avail_actions': np.zeros((map_episode_limit, self.env.n_agents, self.env.n_actions)),
            'filled': np.zeros((map_episode_limit, 1)),
            'obs': np.zeros((map_episode_limit, self.env.n_agents, self.env.get_obs_size())),
            'reward': np.zeros((map_episode_limit, 1)),
            'state': np.zeros((map_episode_limit, self.env.get_state_size())),
            'terminated': np.zeros((map_episode_limit, 1))
        }

        while not terminated:

            pre_transition_data = {
                "state": [self.env.get_state()],
                "avail_actions": [self.env.get_avail_actions()],
                "obs": [self.env.get_cut_down_obs()]
            }

            self.batch.update(pre_transition_data, ts=self.t)

            # Pass the entire batch of experiences up till now to the agents
            # Receive the actions for each agent at this timestep in a batch of size 1

            start_infer1 = time.time()

            actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)

            end_infer1 = time.time()
            self.inference_time += (end_infer1 - start_infer1)



            cpu_actions = actions.to("cpu").numpy()  #通过这种方式，将numpy数据放到pos_transition_data，在MMM地图上能够节约1G左右的GPU显存
            # 与环境进行交互，并保存奖励
            reward, terminated, env_info = self.env.step(cpu_actions[0])
            episode_return += reward

            post_transition_data = {
                "actions": cpu_actions,
                "reward": [(reward,)],
                "terminated": [(terminated != env_info.get("episode_limit", False),)],
            }

            self.batch.update(post_transition_data, ts=self.t)

            
            # 保存到轨迹中
            trajectory['actions'][self.t, :] = cpu_actions.T # 转置
            trajectory['actions_onehot'][self.t, np.arange(cpu_actions.shape[1]), cpu_actions[0]] = 1
            trajectory['avail_actions'][self.t] = pre_transition_data['avail_actions'][0]
            trajectory['filled'][self.t] = 1
            trajectory['obs'][self.t] = pre_transition_data['obs'][0]
            trajectory['reward'][self.t] = post_transition_data['reward'][0]
            trajectory['state'][self.t] = pre_transition_data['state'][0]
            trajectory['terminated'][self.t] = post_transition_data['terminated'][0]

            self.t += 1



        # 最后一步的数据
        last_data = {
            "state": [self.env.get_state()],
            "avail_actions": [self.env.get_avail_actions()],
            "obs": [self.env.get_cut_down_obs()]
        }

        self.batch.update(last_data, ts=self.t)

        # Select actions in the last stored state

        start_infer1 = time.time()
        actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
        end_infer1 = time.time()
        self.inference_time += (end_infer1 - start_infer1)


        cpu_actions = actions.to("cpu").numpy()
        self.batch.update({"actions": cpu_actions}, ts=self.t)
        # self.batch.update({"actions": actions}, ts=self.t)

        # 保存最后一步的数据
        trajectory['actions'][self.t, :] = cpu_actions.T
        trajectory['actions_onehot'][self.t, np.arange(cpu_actions.shape[1]), cpu_actions[0]] = 1
        trajectory['avail_actions'][self.t] = pre_transition_data['avail_actions'][0]
        trajectory['filled'][self.t] = 1
        trajectory['obs'][self.t] = last_data['obs'][0]
        trajectory['reward'][self.t] = np.array(0.0)
        trajectory['state'][self.t] = last_data['state'][0]
        trajectory['terminated'][self.t] = np.array(0)

        cur_stats = self.test_stats if test_mode else self.train_stats
        cur_returns = self.test_returns if test_mode else self.train_returns
        cur_fails = self.test_fails if test_mode else self.train_fails
        log_prefix = "test_" if test_mode else ""
        for k in set(cur_stats) | set(env_info) :
            if k == "num_fails":
                cur_fails.append(env_info.get(k,0))
            else:
                cur_stats.update({k: cur_stats.get(k, 0) + env_info.get(k, 0)})
        # cur_stats.update({k: cur_stats.get(k, 0) + env_info.get(k, 0) for k in set(cur_stats) | set(env_info)})
        cur_stats["n_episodes"] = 1 + cur_stats.get("n_episodes", 0)
        cur_stats["ep_length"] = self.t + cur_stats.get("ep_length", 0)

        if not test_mode:
            self.t_env += self.t

        cur_returns.append(episode_return)

        if test_mode and (len(self.test_returns) == self.args.test_nepisode):
            self._log(cur_returns, cur_fails, cur_stats, log_prefix)
        elif self.t_env - self.log_train_stats_t >= self.args.runner_log_interval:
            self._log(cur_returns, cur_fails, cur_stats, log_prefix)

            minute_inference, second_inference = divmod(self.inference_time, 60)

            minute_inference += (second_inference / 60)

            self.logger.log_stat("minute_inference", minute_inference, self.t_env)

            if hasattr(self.mac.action_selector, "epsilon"):
                self.logger.log_stat("epsilon", self.mac.action_selector.epsilon, self.t_env)
            self.log_train_stats_t = self.t_env

        # 如果需要收集数据，且指定了最大轨迹数，则返回trajectory和self.batch
        if self.args.if_collect_data and hasattr(self.args, 'max_trajectories'):
            return trajectory, self.batch
        else:
            return self.batch

    def _log(self, returns, fails, stats, prefix):
        if len(fails)>0:
            self.logger.log_stat(prefix + "fail_mean", np.mean(fails), self.t_env)
            self.logger.log_stat(prefix + "fail_std", np.std(fails), self.t_env)
            self.logger.log_stat(prefix+"fail_cvar01", compute_cvar(fails, 0.1), self.t_env)
            self.logger.log_stat(prefix+"fail_cvar02", compute_cvar(fails, 0.2), self.t_env)
            self.logger.log_stat(prefix+"fail_cvar03", compute_cvar(fails, 0.3), self.t_env)
            self.logger.log_stat(prefix+"fail_cvar04", compute_cvar(fails, 0.4), self.t_env)
            self.logger.log_stat(prefix+"fail_cvar05", compute_cvar(fails, 0.5), self.t_env)
        self.logger.log_stat(prefix + "return_mean", np.mean(returns), self.t_env)
        self.logger.log_stat(prefix + "return_std", np.std(returns), self.t_env)
        returns.clear()
        fails.clear()

        for k, v in stats.items():
            if k != "n_episodes":
                self.logger.log_stat(prefix + k + "_mean" , v/stats["n_episodes"], self.t_env)
        stats.clear()

def compute_cvar(data, alpha):
    if not isinstance(data, np.ndarray):
        data = np.array(data)
    N = data.shape[0]
    sorted_data = np.sort(data)
    
    # 计算alpha对应的VaR值在数组中的位置
    var_index = int(np.ceil(alpha * len(sorted_data))) - 1
    
    # 计算CVaR值
    cvar = np.mean(sorted_data[:var_index + 1])
    
    if np.isnan(cvar):
        return data[0]
    else:
        return cvar
