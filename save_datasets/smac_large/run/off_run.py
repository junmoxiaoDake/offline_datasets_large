import datetime
import os
import pprint
import time
import threading
import torch as th
import h5py
import sys
from types import SimpleNamespace as SN
from utils.logging import Logger
from utils.timehelper import time_left, time_str
from os.path import dirname, abspath

from learners import REGISTRY as le_REGISTRY
from runners import REGISTRY as r_REGISTRY
from controllers import REGISTRY as mac_REGISTRY
from components.episode_buffer import ReplayBuffer, OfflineBuffer
from components.transforms import OneHot
from components.epidode_sequence import SequenceDataset
from smac.env import StarCraft2Env
import numpy as np
# from config.locomotion_config import Config
# from modules.agents.temporal import TemporalUnet, MulTemporalUnet
# from diffusion.diffusion import GaussianInvDynDiffusion
# from utils.training import Trainer
import utils.arrays 

def get_agent_own_state_size(env_args):
    sc_env = StarCraft2Env(**env_args)
    # qatten parameter setting (only use in qatten)
    return  4 + sc_env.shield_bits_ally + sc_env.unit_type_bits

def run(_run, _config, _log):

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
    args.unique_token = unique_token
    if args.use_tensorboard:
        tb_logs_direc = os.path.join(dirname(dirname(abspath(__file__))), "results", "tb_logs")
        tb_exp_direc = os.path.join(tb_logs_direc, "{}").format(unique_token)
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


def evaluate_sequential(args, runner):

    for _ in range(args.test_nepisode):
        runner.run(test_mode=True)

    if args.save_replay:
        runner.save_replay()

    runner.close_env()

def run_sequential(args, logger):
 
    runner = r_REGISTRY[args.runner](args=args, logger=logger)

    # Set up schemes and groups here
    env_info = runner.get_env_info()
    args.n_agents = env_info["n_agents"]
    args.n_actions = env_info["n_actions"]
    args.state_shape = env_info["state_shape"]


    if getattr(args, 'agent_own_state_size', False):
        args.agent_own_state_size = get_agent_own_state_size(args.env_args)

    # Default/Base scheme
    scheme = {
            "state": {"vshape": env_info["state_shape"]},
            "obs": {"vshape": env_info["obs_shape"], "group": "agents"},
            "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
            "avail_actions": {"vshape": (env_info["n_actions"],), "group": "agents", "dtype": th.int},
            "reward": {"vshape": (1,)},
            "terminated": {"vshape": (1,), "dtype": th.uint8},
        }
    
    groups = {
        "agents" : args.n_agents
    }
    
    preprocess = {
        "actions" : ("actions_onehot", [OneHot(out_dim=args.n_actions)])
    }
    # 将dataset数据转化为数据
    arg_path = '/data/lc/lichao/Offline_MARL/results/dataset/MMM2/expert/part_0.h5' 
    # 从数据集中读取数据并转化为sequenceDataset.
    dataset = SequenceDataset(arg_path, horizon=20, termination_penalty=0, include_returns=True)


    # 定义模型
    observation_dim = dataset.observation_dim
    action_dim = dataset.action_dim


    # 初始化模型
    model = MulTemporalUnet(horizon=20, transition_dim=observation_dim, cond_dim=observation_dim,
                        dim_mults=(1,4,8), returns_condition=True, dim=128, condition_dropout=0.25,
                        calc_energy=False)

    diffusion = GaussianInvDynDiffusion(model, horizon=20, observation_dim=observation_dim, action_dim=action_dim, n_timesteps=200,
                                        loss_type='l2', clip_denoised=True, predict_epsilon=True, hidden_dim=256,
                                        loss_weights=None, loss_discount=1.0, returns_condition=True,
                                        condition_guidance_w=1.2, agent_num=args.n_agents, ar_inv=False)
    path = r'/data/lc/lichao/Offline_MARL/ResQ_src'
    trainer = Trainer(diffusion, dataset, bucket=path, logger=logger)
    
    # test forward & backward pass
    logger.console_logger.info('Testing forward ...')
    batch = utils.arrays.batchify(dataset[0], args.device)
    loss, _ = diffusion.loss(*batch)
    loss.backward()
    logger.console_logger.info('√')

    # train 代码部分
    n_train_steps = 1000000.0
    n_steps_per_epoch = 10000
    n_epochs = int(n_train_steps // n_steps_per_epoch)

    for i in range(n_epochs):
        logger.console_logger.info(f'Epoch {i} / {n_epochs}')
        trainer.train(n_train_steps = n_steps_per_epoch)

        if i % 100 == 0:
            logger.console_logger.info(utils.arrays.report_parameters(model), color='green')

    buffer = ReplayBuffer(scheme, groups, 1, env_info["episode_limit"] + 1,
                           preprocess=preprocess, device="cpu" if args.buffer_cpu_only else args.device)

    mac = mac_REGISTRY[args.mac](buffer.scheme, groups, args)

    # Give runner the scheme
    runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=mac)

    # Learner
    learner = le_REGISTRY[args.learner](mac, buffer.scheme, logger, args)

    if args.use_cuda:
        learner.cuda()

    if args.checkpoint_path != "":

        timesteps = []
        timestep_to_load = 0

        if not os.path.isdir(args.checkpoint_path):
            logger.console_logger.info("Checkpoint directiory {} doesn't exist".format(args.checkpoint_path))
            return

    #     # Go through all files in args.checkpoint_path
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

    #     logger.console_logger.info("??_Loading model from {}".format(model_path))
        learner.load_models(model_path)
        runner.t_env = timestep_to_load

        if args.evaluate or args.save_replay:
            evaluate_sequential(args, runner)
            return

    #     for name in os.listdir(args.checkpoint_path):
    #         full_name = os.path.join(args.checkpoint_path, name)
    #         timesteps = []
    #         if os.path.isdir(full_name) and name.isdigit():
    #             timesteps.append(int(name))


    # # start training
    # episode = 0
    # last_test_T = -args.test_interval - 1
    # last_log_T = 0
    # model_save_time = 0

    # start_time = time.time()
    # last_time = start_time

    # logger.console_logger.info("Beginning training for {} timesteps".format(args.t_max))

    # while runner.t_env <= args.t_max:

    #     # Run for a whole episode at a time

    #     with th.no_grad():
    #         episode_batch = runner.run(test_mode=False)
    #         buffer.insert_episode_batch(episode_batch)

    #     if buffer.can_sample(args.batch_size):
    #         episode_sample = buffer.sample(args.batch_size)

    #         # Truncate batch to only filled timesteps
    #         max_ep_t = episode_sample.max_t_filled()
    #         episode_sample = episode_sample[:, :max_ep_t]

    #         if episode_sample.device != args.device:
    #             episode_sample.to(args.device)

    #         learner.train(episode_sample, runner.t_env, episode)
    #         del episode_sample

    #     # Execute test runs once in a while
    #     n_test_runs = max(1, args.test_nepisode // runner.batch_size)
    #     if (runner.t_env - last_test_T) / args.test_interval >= 1.0:

    #         logger.console_logger.info("t_env: {} / {}".format(runner.t_env, args.t_max))
    #         logger.console_logger.info("Estimated time left: {}. Time passed: {}".format(
    #             time_left(last_time, last_test_T, runner.t_env, args.t_max), time_str(time.time() - start_time)))
    #         last_time = time.time()

    #         last_test_T = runner.t_env
    #         for _ in range(n_test_runs):
    #             runner.run(test_mode=True)

    #     if args.save_model and (runner.t_env - model_save_time >= args.save_model_interval or model_save_time == 0):
    #         model_save_time = runner.t_env
    #         save_path = os.path.join(args.local_results_path, "models", args.env_args['map_name'], args.unique_token, str(runner.t_env))
    #         #"results/models/{}".format(unique_token)
    #         os.makedirs(save_path, exist_ok=True)
    #         logger.console_logger.info("Saving models to {}".format(save_path))

    #         # learner should handle saving/loading -- delegate actor save/load to mac,
    #         # use appropriate filenames to do critics, optimizer states
    #         learner.save_models(save_path)

    #     episode += args.batch_size_run

    #     if (runner.t_env - last_log_T) >= args.log_interval:
    #         logger.log_stat("episode", episode, runner.t_env)
    #         logger.print_recent_stats()
    #         last_log_T = runner.t_env

    # runner.close_env()
    # logger.console_logger.info("Finished Training")


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
