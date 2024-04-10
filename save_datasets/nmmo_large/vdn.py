import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

from forge.trinity import smith, Trinity, Pantheon, God, Sword
from forge.trinity import ANN

import time
import json
import sys

from forge.blade.action.tree import ActionTree
from forge.blade.action.v2 import ActionV2

from forge.ethyr import torch as torchlib

import random
import argparse

import experiments

# 环境参数
num_agents = 512

# num_actions = 3
state_dim = 4

# 模型参数
hidden_units = 64

# 训练参数
batch_size = 32
learning_rate = 0.001
discount_factor = 0.95
num_episodes = 1

# 创建智能体神经网络模型
class QNetwork(nn.Module):
    def __init__(self):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_units)
        # self.fc2 = nn.Linear(hidden_units, num_actions)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x



def parseArgs():
   parser = argparse.ArgumentParser('Projekt Godsword')
   parser.add_argument('--nRealm', type=int, default='1',
         help='Number of environments (1 per core)')
   parser.add_argument('--api', type=str, default='native',
         help='API to use (native/vecenv)')
   parser.add_argument('--ray', type=str, default='default',
         help='Ray mode (local/default/remote)')
   parser.add_argument('--render', action='store_true', default=False,
         help='Render env')
   return parser.parse_args()

if __name__ == '__main__':

    sys.setrecursionlimit(10000)
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

    args = parseArgs()
    # assert args.api in ('native', 'vecenv')
    # config = experiments.exps['testlaw16']

    # config = experiments.exps['testlaw128']

    # config = experiments.exps['testlaw256']

    config = experiments.exps['testlaw512']

    # 创建智能体
    agents = [ANN(config) for _ in range(num_agents)]
    optimizers = [optim.Adam(agent.parameters(), lr=learning_rate) for agent in agents]

    step = 0


    env = smith.VecEnv(config, args, step)
    #The environment is persistent. Reset only to start it.
    envsObs = env.reset()


    train_start_flag = True


    inference_time = 0
    step1 = 0
    step2 = 0
    step3 = 0
    step4 = 0
    step5 = 0
    step6 = 0

    # 训练VDN算法
    for episode in range(num_episodes):
        # # 初始化环境和状态
        # env.reset()
        # states = [torch.tensor(env.get_state(), dtype=torch.float32) for _ in range(num_agents)]
        done = False

        done_flag = 1

        # 初始化经验回放缓冲区
        replay_buffer = deque(maxlen=60000)

        while not done:
            # 每个智能体根据当前状态选择动作
            # actions = [torch.argmax(agent(state)).item() for agent, state in zip(agents, states)]

            actions = []
            index = 0
            for obs in envsObs:  # Environment
                atns = []
                for ob in obs:  # Agent
                    # ent, stim = ob
                    # s = torchlib.Stim(ent, stim, config)
                    # act = ActionTree(stim, ent, ActionV2).actions()
                    # _, move, attk = act
                    # move_leaves = move.args(stim, ent, config)
                    # attk_leaves = attk.args(stim, ent, config)
                    # s.conv = s.conv.to(device)
                    # s.flat = s.flat.to(device)
                    # s.ents = s.ents.to(device)
                    #
                    # random_int = random.randint(0, 2)
                    # action = attk_leaves[random_int]
                    #
                    # targets = action.args(stim, ent, config)
                    # targets = torch.tensor([e.stim for e in targets]).float()
                    # targets = targets.to(device)
                    # targets_leaves = action.args(stim, ent, config)
                    # moveActionIdx, attackActionIdx, atnArgs, val = agents[index](s.conv, s.flat, s.ents, targets)
                    # action = (move, attk)
                    # arguments = (move_leaves[int(moveActionIdx)], [targets_leaves[int(attackActionIdx)]])
                    # atns.append((ent.entID, action, arguments, float(val)))
                    ent, stim = ob
                    start_time = time.time()
                    action, arguments, atnArgs, val= agents[index](ent, stim)
                    end_time = time.time()
                    inference_time += (end_time - start_time)

                    atns.append((ent.entID, action, arguments, float(val)))
                    index += 1
                actions.append(atns)
            next_envsObs, rews, dones, infos,loadTime, stepEntsTime, stepWorldTime, spawnTime, EnvTime, returnTime  = env.step(actions)
            step1 += loadTime[0]
            step2 += stepEntsTime[0]
            step3 += stepWorldTime[0]
            step4 += spawnTime[0]
            step5 += EnvTime[0]
            step6 += returnTime[0]
            # 执行动作，观察奖励和下一个状态
            # rewards, next_states, done = env.step(actions)

            # 存储经验到经验回放缓冲区
            replay_buffer.append((envsObs, actions, rews, next_envsObs, done))

            # 更新状态
            # states = [torch.tensor(next_state, dtype=torch.float32) for next_state in next_states]

            envsObs = next_envsObs

            # 从经验回放缓冲区中随机采样一批数据
            if done_flag % 1000 == 0:
                batch = np.random.choice(len(replay_buffer), batch_size, replace=False)
                for i in batch:
                    states_batch, actions_batch, rewards_batch, next_states_batch, done_batch = replay_buffer[i]

                    # 计算每个智能体的Q值
                    # Q_values = [agent(states_batch[j]) for j, agent in enumerate(agents)]

                    new_index = 0
                    Q_values = []
                    for obs in states_batch:  # Environment
                        atns = []
                        for ob in obs:  # Agent
                            ent, stim = ob
                            start_time = time.time()
                            action, arguments, atnArgs, val= agents[new_index](ent, stim)

                            end_time = time.time()
                            inference_time += (end_time - start_time)


                            # atns.append((ent.entID, action, arguments, float(val)))
                            new_index += 1
                            Q_values.append(torch.cat((atnArgs[0][0], atnArgs[1][0]), 1))
                        # actions.append(atns)

                    # next_envsObs, rews, dones, infos = env.step(actions)


                    # 更新每个智能体的神经网络
                    for j in range(new_index):
                        target = rewards_batch[0][0] + discount_factor * torch.max(Q_values[j]).item()
                        loss = nn.MSELoss()(Q_values[j], torch.tensor([target]))
                        optimizers[j].zero_grad()
                        loss.backward()
                        optimizers[j].step()


            if train_start_flag:
                print("training......")
                train_start_flag = False

            print("episode:", episode, "step:", done_flag)
            done_flag += 1

            if done_flag >= 1000:

                data = {}
                data['inference_time'] = inference_time
                data['step1'] = step1
                data['step2'] = step2
                data['step3'] = step3
                data['step4'] = step4
                data['step5'] = step5
                data['step6'] = step6
                with open("512agents——improve1.json", "w") as f:
                    json.dump(data, f)

                # with open("256agents.json", "w") as f:
                #     json.dump(data, f)

                break





# 在训练完成后，可以使用智能体来做出决策
def select_actions(states):
    actions = [torch.argmax(agent(torch.tensor(state, dtype=torch.float32))).item() for agent, state in zip(agents, states)]
    return actions
