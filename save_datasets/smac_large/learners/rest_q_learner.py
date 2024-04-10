import copy
from components.episode_buffer import EpisodeBatch
from modules.mixers.vdn import VDNMixer
from modules.mixers.qmix import QMixer
from modules.mixers.qmix_central_no_hyper import QMixerCentralFF
from learners.rest_q_learner_central import get_ws
from modules.mixers.qatten import QattenMixer

from utils.rl_utils import build_td_lambda_targets
import torch as th
from torch.optim import RMSprop
from collections import deque
from controllers import REGISTRY as mac_REGISTRY
from utils.th_utils import get_parameters_num
from envs.one_step_matrix_game import print_matrix_status

"""

v2的restq的
Q = Q_tot(s,a) + w_r(s,a) R(s,a) 
其中 w_r = 1 当a是a-hat是，否则w_r=0
并且限制要求R(s,a)大于0


v1的restq
这个是RESTQ，主要的实现是[w(s,a) Q_tot(s,a) + R_tot(s,a) - Q_jt]^2 最小
w(s,a) 当a是最大值的时候为1，其他时候为0
并且Q_tot在a不是最大值的时候， Q_tot为0或者为一个不大的值

"""

class RestQLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.logger = logger

        self.mac_params = list(mac.parameters())
        self.params = list(self.mac.parameters())

        self.last_target_update_episode = 0
        self.central_mixer = None
        self.mixer = None
        assert args.mixer is not None
        if args.mixer is not None:
            if args.mixer == "vdn":
                self.mixer = VDNMixer()
            elif args.mixer == "qmix":
                self.mixer = QMixer(args)
            elif args.mixer == "qatten":
                self.mixer = QattenMixer(args)
            else:
                raise ValueError("Mixer {} not recognised.".format(args.mixer))
            self.mixer_params = list(self.mixer.parameters())
            self.params += list(self.mixer.parameters())
            self.target_mixer = copy.deepcopy(self.mixer)

        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        self.target_mac = copy.deepcopy(mac)

        # REST Q
        if "rest_mixer" in vars(args):
            if args.rest_mixer == "vdn":
                self.rest_mixer = VDNMixer()
            elif args.rest_mixer == "qmix":
                self.rest_mixer = QMixer(args)
            elif args.rest_mixer == "qatten":
                self.rest_mixer = QattenMixer(args)
            else:
                self.rest_mixer = QMixerCentralFF(args)
        else:
            self.rest_mixer = QMixerCentralFF(args)
        self.rest_target_mixer = copy.deepcopy(self.rest_mixer)
        self.rest_mac = copy.deepcopy(self.mac)  # added for RESTQ
        self.rest_target_mac = copy.deepcopy(self.target_mac)  # added for RESTQ

        self.params += list(self.rest_mixer.parameters())
        self.params += list(self.rest_target_mixer.parameters())
        self.params += list(self.rest_mac.parameters())  # added for RESTQ
        self.params += list(self.rest_target_mac.parameters())  # added for RESTQ

        print('Mixer Size: ')
        print(get_parameters_num(list(self.mixer.parameters()) + list(self.rest_mixer.parameters())))

        self.optimiser = RMSprop(params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)

        self.log_stats_t = -self.args.learner_log_interval - 1

        self.grad_norm = 1
        self.mixer_norm = 1
        self.mixer_norms = deque([1], maxlen=100)

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # Get the relevant quantities
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]

        # Calculate estimated Q-Values
        mac_out = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agent_outs = self.mac.forward(batch, t=t)
            mac_out.append(agent_outs)
        mac_out = th.stack(mac_out, dim=1)  # Concat over time
        # Pick the Q-Values for the actions taken by each agent
        chosen_action_qvals_agents = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the last dim
        chosen_action_qvals = chosen_action_qvals_agents

        rest_mac_out = []
        self.rest_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            rest_agent_outs = self.rest_mac.forward(batch, t=t)
            rest_mac_out.append(rest_agent_outs)
        rest_mac_out = th.stack(rest_mac_out, dim=1)  # Concat over time
        rest_chosen_action_qvals = th.gather(rest_mac_out[:, :-1], dim=3, index=actions).squeeze(3)

        # Calculate the Q-Values necessary for the target
        target_mac_out = []
        self.target_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            target_agent_outs = self.target_mac.forward(batch, t=t)
            target_mac_out.append(target_agent_outs)
        # We don't need the first timesteps Q-Value estimate for calculating targets
        target_mac_out = th.stack(target_mac_out[1:], dim=1)  # We don't need the first time step of mac_out
        # Mask out unavailable actions
        target_mac_out[avail_actions[:, 1:] == 0] = -9999999  # From OG deepmarl

        # Calculate the Q-Values necessary for the target
        rest_target_mac_out = []
        self.rest_target_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            rest_target_agent_outs = self.rest_target_mac.forward(batch, t=t)
            rest_target_mac_out.append(rest_target_agent_outs)
        # We don't need the first timesteps Q-Value estimate for calculating targets
        rest_target_mac_out = th.stack(rest_target_mac_out[1:], dim=1)  # We don't need the first time step of mac_out
        # Mask out unavailable actions
        rest_target_mac_out[avail_actions[:, 1:] == 0] = -9999999  # From OG deepmarl


        # Max over target Q-Values
        if self.args.double_q:  # Get actions that maximise live Q (for double q-learning)
            mac_out_detach = mac_out.clone().detach() #mac_out batch_size, seq_length, n_agents, n_commands
            mac_out_detach[avail_actions == 0] = -9999999
            cur_max_action_targets, cur_max_actions = mac_out_detach[:, :].max(dim=3, keepdim=True)             #(max, max_indices) = torch.max(input, dim, keepdim=False)
            target_max_agent_qvals = th.gather(target_mac_out[:,:], 3, cur_max_actions[:,1:]).squeeze(3)  #注意这里的target是target network的target是为了让网络更加稳定，而不是TD的target

            rest_mac_out_detach = rest_mac_out.clone().detach()
            rest_mac_out_detach[avail_actions == 0] = -9999999
            rest_cur_max_action_targets, rest_cur_max_actions = cur_max_action_targets, cur_max_actions #这一点要注意，RestQ的argmax必须和Q_tot一样的
            # rest_cur_max_action_targets, rest_cur_max_actions = rest_mac_out_detach[:, :].max(dim=3, keepdim=True)             #(max, max_indices) = torch.max(input, dim, keepdim=False)
            rest_target_max_agent_qvals = th.gather(rest_target_mac_out[:,:], 3, rest_cur_max_actions[:,1:]).squeeze(3)
        else:
            raise Exception("Use double q")


        # Mix


        # We use the calculation function of sarsa lambda to approximate q star lambda
        #这个就是weighted qmix论文里面的y_i, y_i = r + \gamma Q^*(s', \tau, argmax_u'Q_tot)
        # targets = build_td_lambda_targets(rewards, terminated, mask, target_max_qvals,
        #                             self.args.n_agents, self.args.gamma, self.args.td_lambda)



        # QMIX loss with weighting
        # ws = th.ones_like(td_error) * self.args.w
        resq_version = self.args.resq_version

        #令  t= (actions == cur_max_actions[:, :-1]),这一步得到t是一个[n_batch, n_time-1, n_agent, 1]的一个False， True的矩阵
        #之后t.min(dim=2)，目的是判断dim=2的agent维度上，是不是所有n_agent都找到最大值了。在min的时候，False相当于0，True相当于1。min的结果是1(True)的话，就意味着所有的agent都取了argmax (True)的值。
        #min(dim=2)[0]这个公式最后面的[0]的意思是取min之后的值，而不是取min后面的indices
        # w_r = w_r_target = is_max_action
        Q_tot = self.mixer(chosen_action_qvals, batch["state"][:, :-1])
        Q_r = self.rest_mixer(rest_chosen_action_qvals, batch["state"][:, :-1])
        target_Q_tot = self.target_mixer(target_max_agent_qvals, batch["state"][:, 1:])
        target_Q_r = self.rest_target_mixer(rest_target_max_agent_qvals, batch["state"][:, 1:])

        is_max_action = (actions == cur_max_actions[:, :-1]).min(dim=2)[0]
        w_r = get_ws(resq_version, is_max_action, Q_tot)
        # is_tot_larger = target_Q_tot > Q_tot
        # w_r = w_r_target = w_r | is_tot_larger
        if resq_version in ["v2"]: #这个是为了匹配老版本
            w_r_target = w_r
        else:
            is_max_action_target = (batch["actions"][:, 1:] == cur_max_actions[:, 1:]).min(dim=2)[0]  #actions 是 batch["actions"][:, :-1]
            w_r_target = get_ws(resq_version, is_max_action_target, Q_tot)
        Q_current = Q_tot + w_r * Q_r
        # target_max_qvals = target_Q_tot + target_Q_r  #20211120，这个改动比较大不知道靠不靠谱，先试试看吧
        target_max_qvals = target_Q_tot + w_r_target * target_Q_r
        targets = rewards + self.args.gamma * (1 - terminated) * target_max_qvals
        td_error = (Q_current - targets.detach())
        mask = mask.expand_as(td_error)
        # 0-out the targets that came from padded data
        masked_td_error = td_error * mask

        # qtot_larger = targets > target_max_qvals[:,1:] #这个是CW qmix的优化版
        # w_r = th.where(is_max_action | qtot_larger, th.ones_like(chosen_action_qvals)*1, th.zeros_like(chosen_action_qvals)) # Target is greater than current max
        # w_to_use = w_r.mean().item() # Average of ws for logging
        # w_r = th.where(td_error < 0, th.ones_like(td_error)*1, th.zeros_like(td_error)) # Target is greater than current max
        # w_to_use = w_r.mean().item() # For logging
        # CW_qmix 当yi > Q^(s,)的最大值的时候，或者当u等于u*的时候，就认为应该赋值为1
        # 要构建w_r，当u = u_hat的时候为1，只能取approximation，当w_r(s)的最大值
        # else: # CW-QMIX
        #     #我这里复制不了CW QMIX的东西，因为我缺乏一个类似Central Qmix的东西，来做approximation
        #
        #     max_action_qtot = self.target_central_mixer(central_target_max_agent_qvals[:, :-1], batch["state"][:, :-1])
        #     qtot_larger = targets > max_action_qtot
        #     # ws = th.where(is_max_action | qtot_larger, th.ones_like(td_error)*1, ws) # Target is greater than current max
        #     ws = th.where(is_max_action | qtot_larger, ws, th.ones_like(td_error)*1) # Target is greater than current max
        #     w_to_use = ws.mean().item() # Average of ws for logging

        # print(chosen_action_qvals.shape, chosen_action_qvals.shape, ws.shape)
        # td_error = (chosen_action_qvals + w_r.detach() * rest_chosen_action_qvals - (targets.detach()))
        # print("td_error.shape", td_error.shape)
        qmix_loss = 1/2 * (masked_td_error ** 2).sum() / mask.sum()
        # print("loss", qmix_loss.item())

        """约束下q_tot(\hat(u)) > R_tot"""
        # t = (1/w_to_use) * ws.detach() * th.clamp(rest_chosen_action_qvals.detach() - chosen_action_qvals, 0) * mask
        # condition_loss = (t**2).sum()/mask.sum()

        """约束非最优的Q_r为0的部分"""
        noopt_loss = None
        noopt_loss1 = None
        noopt_loss2 = None
        w_r2 = 1 - w_r
        if self.args.resq_version in ["v2_wrong", "v2"]:
            w_r2 = 1 - w_r ##之前显示的RestQ_v2，其实应该是v2_wrong的结果。。。为了保证之前实验版本的兼容性，故意将rest_q_learner的ws2的权重的设置和resq_q_learner_central设置的不一样
        if self.args.resq_version in ["v4", "v5", "v6"]:#rest_learner_central v2对应的restq_learner版本为v5
            w_r2 = w_r
        if self.args.resq_version == "v3":
            Q_r_ = th.max(Q_r, 0)[0]  # 这样如果Q_r大于0的话，会被挑出来
            noopt_loss1 = (((Q_r_ * mask) ** 2).sum()) / mask.sum()  # 要求Q_r< 0
            noopt_loss = noopt_loss1
        elif self.args.resq_version in ["v5", "v4"]:
            gap = self.args.max_second_gap
            if self.args.condition_loss == "mse":
                Q_r_ = th.min(Q_r, gap)[0]  # 这样如果Q_r小于gap的话(v2里面gap是0，意味着挑小于0的)，会被挑出来
                t0 = w_r2.detach() * Q_r_ * mask  #w_r2这个时候是将max action 对应的Q挑出来
                noopt_loss1 = ((t0 ** 2).sum()) /mask.sum() #要求Q_r(max) > 0, 也就是说v5, v4 已经是对应了actual_v2+的版本
            elif self.args.condition_loss == "delta":
                t0 = th.where(Q_r<gap, th.ones_like(Q_r), th.zeros_like(Q_r))
                noopt_loss1 = (w_r2.detach() * t0 * mask).sum() * self.args.condition_loss_delta /mask.sum() #要求Q_r(max) > 0
        elif self.args.resq_version in ["v6"]:
            if self.args.condition_loss == "mse":
                t = th.min(chosen_action_qvals - Q_r, 0)[0]  # 额外引入约束，约束Q_tot > Q_r, 这个是我在v6上额外增加的要求，应该不影响性质
                t1 = (w_r2.detach() * t)
                noopt_loss2 = (t1**2).sum()/mask.sum() #要求Q_tot(max) > Q_r(max)
            elif self.args.condition_loss == "delta":
                t0 = th.where( (chosen_action_qvals - Q_r)<0, th.ones_like(Q_r), th.zeros_like(Q_r))
                noopt_loss2 = (w_r2.detach() * t0 * mask).sum() * self.args.condition_loss_delta /mask.sum() #要求Q_r(max) > Q_r(max)但是是以delta的形式来约束的。
        elif self.args.resq_version == "v2": #为了保证原来代码v2的兼容性
            Q_r_ = th.min(Q_r, 0)[0] #这样如果Q_r小于0的话，会被挑出来
            noopt_loss = (((w_r2.detach() *  Q_r_ * mask) **2).sum()) /mask.sum()

        if noopt_loss2 is not None:
            noopt_loss = noopt_loss1 + noopt_loss2
        else:
            noopt_loss = noopt_loss1
        loss = self.args.qmix_loss * qmix_loss + self.args.noopt_loss * noopt_loss

        # Optimise
        self.optimiser.zero_grad()
        loss.backward()

        # Logging
        agent_norm = 0
        for p in self.mac_params:
            param_norm = p.grad.data.norm(2)
            agent_norm += param_norm.item() ** 2
        agent_norm = agent_norm ** (1. / 2)

        mixer_norm = 0
        for p in self.mixer_params:
            param_norm = p.grad.data.norm(2)
            mixer_norm += param_norm.item() ** 2
        mixer_norm = mixer_norm ** (1. / 2)
        self.mixer_norm = mixer_norm
        # self.mixer_norms.append(mixer_norm)

        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.grad_norm = grad_norm

        self.optimiser.step()

        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss", loss.item(), t_env)
            self.logger.log_stat("qmix_loss", qmix_loss.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm, t_env)
            self.logger.log_stat("mixer_norm", mixer_norm, t_env)
            self.logger.log_stat("agent_norm", agent_norm, t_env)
            if noopt_loss is not None:
                self.logger.log_stat("noopt_loss", noopt_loss.item(), t_env)
            if noopt_loss1 is not None:
                self.logger.log_stat("noopt_loss1", noopt_loss1.item(), t_env)
            if noopt_loss2 is not None:
                self.logger.log_stat("noopt_loss2", noopt_loss2.item(), t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat("td_error_abs", (masked_td_error.abs().sum().item()/mask_elems), t_env)
            self.logger.log_stat("q_taken_mean", (chosen_action_qvals * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat("target_mean", (targets * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            # self.logger.log_stat("central_loss", central_loss.item(), t_env)
            # self.logger.log_stat("w_to_use", w_to_use, t_env)
            self.log_stats_t = t_env

            # print estimated matrix
            if self.args.env.find("one_step_matrix_game")>=0:
                print_matrix_status(t_env, self.logger, batch, self.mixer, mac_out, hidden=None, max_q_i=None, is_wqmix=True, wqmix_central_mixer=self.central_mixer, rest_mixer=self.rest_mixer)
                # print_matrix_status(batch, self.central_mixer, mac_out)
                # print_matrix_status(batch, self.mixer, mac_out, hidden=None, max_q_i=None, is_wqmix=True, wqmix_central_mixer=self.central_mixer, rest_mixer=self.rest_mixer)

    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        self.rest_target_mac.load_state(self.rest_mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
            self.rest_target_mixer.load_state_dict(self.rest_mixer.state_dict())
        self.logger.console_logger.info("Updated target network")

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()
        self.rest_mac.cuda()
        self.rest_target_mac.cuda()
        if self.mixer is not None:
            self.mixer.cuda()
            self.target_mixer.cuda()
            self.rest_mixer.cuda()
            self.rest_target_mixer.cuda()
        # if self.central_mac is not None:
        #     self.central_mac.cuda()
        #     self.target_central_mac.cuda()
        # self.central_mixer.cuda()
        # self.target_central_mixer.cuda()

    # TODO: Model saving/loading is out of date!
    def save_models(self, path):
        self.mac.save_models(path)
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
        self.optimiser.load_state_dict(th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))
