import time

from envs import REGISTRY as env_REGISTRY
from functools import partial
from components.episode_buffer import EpisodeBatch
import numpy as np


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
        self.train_stats = {}
        self.test_stats = {}

        # Log the first run
        self.log_train_stats_t = -1000000


        self.total_env_time = 0

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

        map_episode_limit = self.episode_limit + 1  # 防止越界
        # 保存轨迹
        trajectory = {
            'actions': np.zeros((map_episode_limit, self.env.n_agents, 1)),
            'actions_onehot': np.zeros((map_episode_limit, self.env.n_agents, self.env.n_actions)),
            'avail_actions': np.zeros((map_episode_limit, self.env.n_agents, self.env.n_actions)),
            'filled': np.zeros((map_episode_limit, 1)),
            'obs': np.zeros((map_episode_limit, self.env.n_agents, self.env.get_obs_size())),
            'reward': np.zeros((map_episode_limit, 1)),
            'state': np.zeros((map_episode_limit, self.env.get_state_size())),
            'terminated': np.zeros((map_episode_limit, 1))
        }



        while not terminated:

            env_start_time = time.time()
            pre_transition_data = {
                "state": [self.env.get_state()],
                "avail_actions": [self.env.get_avail_actions()],
                "obs": [self.env.get_obs()]
            }
            env_end_time = time.time()
            self.total_env_time += (env_end_time - env_start_time)


            self.batch.update(pre_transition_data, ts=self.t)

            # Pass the entire batch of experiences up till now to the agents
            # Receive the actions for each agent at this timestep in a batch of size 1







            start_infer1 = time.time()
            actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
            end_infer1 = time.time()
            self.inference_time += (end_infer1 - start_infer1)
            # Fix memory leak
            cpu_actions = actions.to("cpu").numpy()

            env_start_time_2 = time.time()
            reward, terminated, env_info = self.env.step(actions[0])
            env_end_time_2 = time.time()
            self.total_env_time += (env_end_time_2 - env_start_time_2)


            episode_return += reward


            post_transition_data = {
                "actions": cpu_actions,
                "reward": [(reward,)],
                "terminated": [(terminated != env_info.get("episode_limit", False),)],
            }



            self.batch.update(post_transition_data, ts=self.t)

            # 保存到轨迹中
            trajectory['actions'][self.t, :] = cpu_actions.T  # 转置
            trajectory['actions_onehot'][self.t, np.arange(cpu_actions.shape[1]), cpu_actions[0]] = 1
            trajectory['avail_actions'][self.t] = pre_transition_data['avail_actions'][0]
            trajectory['filled'][self.t] = 1
            trajectory['obs'][self.t] = pre_transition_data['obs'][0]
            trajectory['reward'][self.t] = post_transition_data['reward'][0]
            trajectory['state'][self.t] = pre_transition_data['state'][0]
            trajectory['terminated'][self.t] = post_transition_data['terminated'][0]


            self.t += 1

        env_start_time_3 = time.time()
        last_data = {
            "state": [self.env.get_state()],
            "avail_actions": [self.env.get_avail_actions()],
            "obs": [self.env.get_obs()]
        }
        env_end_time_3 = time.time()
        self.total_env_time += (env_end_time_3 - env_start_time_3)

        self.batch.update(last_data, ts=self.t)

        # Select actions in the last stored state
        start_infer2 = time.time()
        actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
        end_infer2 = time.time()
        self.inference_time += (end_infer2 - start_infer2)

        # Fix memory leak
        cpu_actions = actions.to("cpu").numpy()
        self.batch.update({"actions": cpu_actions}, ts=self.t)

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
        log_prefix = "test_" if test_mode else ""
        cur_stats.update({k: cur_stats.get(k, 0) + env_info.get(k, 0) for k in set(cur_stats) | set(env_info)})
        cur_stats["n_episodes"] = 1 + cur_stats.get("n_episodes", 0)
        cur_stats["ep_length"] = self.t + cur_stats.get("ep_length", 0)

        if not test_mode:
            self.t_env += self.t

        cur_returns.append(episode_return)

        if test_mode and (len(self.test_returns) == self.args.test_nepisode):
            self._log(cur_returns, cur_stats, log_prefix)
        elif self.t_env - self.log_train_stats_t >= self.args.runner_log_interval:
            self._log(cur_returns, cur_stats, log_prefix)

            minute_env, second_env = divmod(self.total_env_time, 60)

            minute_env += (second_env / 60)

            self.logger.log_stat("minute_env", minute_env, self.t_env)

            minute_infer, second_infer = divmod(self.inference_time, 60)

            minute_infer += (second_infer / 60)

            self.logger.log_stat("minute_infer", minute_infer, self.t_env)

            if hasattr(self.mac.action_selector, "epsilon"):
                self.logger.log_stat("epsilon", self.mac.action_selector.epsilon, self.t_env)
            self.log_train_stats_t = self.t_env
            # 如果需要收集数据，且指定了最大轨迹数，则返回trajectory和self.batch

        if self.args.if_collect_data and hasattr(self.args, 'max_trajectories'):
            return trajectory, self.batch
        else:
            return self.batch

    def _log(self, returns, stats, prefix):
        self.logger.log_stat(prefix + "return_mean", np.mean(returns), self.t_env)
        self.logger.log_stat(prefix + "return_std", np.std(returns), self.t_env)
        returns.clear()

        for k, v in stats.items():
            if k != "n_episodes":
                self.logger.log_stat(prefix + k + "_mean" , v/stats["n_episodes"], self.t_env)
        stats.clear()
