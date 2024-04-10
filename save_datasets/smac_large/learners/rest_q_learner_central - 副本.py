import copy
from components.episode_buffer import EpisodeBatch
from modules.mixers.vdn import VDNMixer
from modules.mixers.qmix import QMixer
from modules.mixers.qmix_central_no_hyper import QMixerCentralFF
from utils.rl_utils import build_td_lambda_targets
import torch as th
from torch.optim import RMSprop
from collections import deque
from controllers import REGISTRY as mac_REGISTRY
from utils.th_utils import get_parameters_num
from envs.one_step_matrix_game import print_matrix_status

"""


v0的restq
这个是RESTQ，主要的实现是[w(s,a) Q_tot(s,a) + R_tot(s,a) - Q_jt]^2 最小
w(s,a) 当a是最大值的时候为1，其他时候为0
并且Q_tot在a不是最大值的时候， Q_tot为0或者为一个不大的值

v2的restq的
Q = Q_tot(s,a) + w_r(s,a) R(s,a) 
其中 w_r = 1 当a是a-hat是，否则w_r=0
并且限制要求R(s,a)大于0
这里的learner是有用了Central_mac
"""
class RestQLearnerCentral:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.logger = logger

        self.mac_params = list(mac.parameters())
        self.params = list(self.mac.parameters())

        self.last_target_update_episode = 0

        self.mixer = None
        assert args.mixer is not None
        if args.mixer is not None:
            if args.mixer == "vdn":
                self.mixer = VDNMixer()
            elif args.mixer == "qmix":
                self.mixer = QMixer(args)
            else:
                raise ValueError("Mixer {} not recognised.".format(args.mixer))
            self.mixer_params = list(self.mixer.parameters())
            self.params += list(self.mixer.parameters())
            self.target_mixer = copy.deepcopy(self.mixer)

        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        self.target_mac = copy.deepcopy(mac)

        # Central Q
        # TODO: Clean this mess up!
        self.central_mac = None
        if self.args.central_mixer in ["ff", "atten"]:
            if self.args.central_loss == 0:
                self.central_mixer = self.mixer
                self.central_mac = self.mac
                self.target_central_mac = self.target_mac
            else:
                if self.args.central_mixer == "ff":
                    self.central_mixer = QMixerCentralFF(args) # Feedforward network that takes state and agent utils as input
                    self.rest_mixer = QMixerCentralFF(args)#added for RESTQ
                # elif self.args.central_mixer == "atten":
                    # self.central_mixer = QMixerCentralAtten(args)
                else:
                    raise Exception("Error with central_mixer")

                assert args.central_mac == "basic_central_mac"
                self.central_mac = mac_REGISTRY[args.central_mac](scheme, args) # Groups aren't used in the CentralBasicController. Little hacky
                self.target_central_mac = copy.deepcopy(self.central_mac)
                self.params += list(self.central_mac.parameters())

                self.rest_mac = copy.deepcopy(self.mac) #added for RESTQ
                self.rest_target_mac = copy.deepcopy(self.rest_mac)#added for RESTQ
                self.params += list(self.rest_mac.parameters())#added for RESTQ

        else:
            raise Exception("Error with qCentral")
        self.params += list(self.central_mixer.parameters())
        self.params += list(self.rest_mixer.parameters())
        self.target_central_mixer = copy.deepcopy(self.central_mixer)
        self.rest_target_mixer = copy.deepcopy(self.rest_mixer) #added for RESTQ
        print('Mixer Size: ')
        print(get_parameters_num(list(self.mixer.parameters()) + list(self.central_mixer.parameters()) + list(self.rest_mixer.parameters())))

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
        target_mac_out = th.stack(target_mac_out[:], dim=1)  # Concat across time
        # Mask out unavailable actions
        target_mac_out[avail_actions[:, :] == 0] = -9999999  # From OG deepmarl

        # Calculate the Q-Values necessary for the target
        rest_target_mac_out = []
        self.rest_target_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            rest_target_agent_outs = self.rest_target_mac.forward(batch, t=t)
            rest_target_mac_out.append(rest_target_agent_outs)
        # We don't need the first timesteps Q-Value estimate for calculating targets
        rest_target_mac_out = th.stack(rest_target_mac_out[:], dim=1)  # Concat across time
        # Mask out unavailable actions
        rest_target_mac_out[avail_actions[:, :] == 0] = -9999999  # From OG deepmarl


        # Max over target Q-Values
        if self.args.double_q:  # Get actions that maximise live Q (for double q-learning)
            mac_out_detach = mac_out.clone().detach() #mac_out batch_size, seq_length, n_agents, n_commands
            mac_out_detach[avail_actions == 0] = -9999999
            cur_max_action_targets, cur_max_actions = mac_out_detach[:, :].max(dim=3, keepdim=True)             #(max, max_indices) = torch.max(input, dim, keepdim=False)
            target_max_agent_qvals = th.gather(target_mac_out[:,:], 3, cur_max_actions[:,:]).squeeze(3)

            rest_mac_out_detach = rest_mac_out.clone().detach()
            rest_mac_out_detach[avail_actions == 0] = -9999999
            rest_cur_max_action_targets, rest_cur_max_actions = cur_max_action_targets, cur_max_actions #这一点要注意，RestQ的argmax必须和Q_tot一样的
            rest_target_max_agent_qvals = th.gather(rest_target_mac_out[:,:], 3, rest_cur_max_actions[:,:]).squeeze(3)
        else:
            raise Exception("Use double q")

        # Central MAC stuff
        central_mac_out = []
        self.central_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agent_outs = self.central_mac.forward(batch, t=t)
            central_mac_out.append(agent_outs)
        central_mac_out = th.stack(central_mac_out, dim=1)  # Concat over time
        # print(central_mac_out.shape) #mac_out batch_size, seq_length, n_agents, n_commands, args.central_action_embed(default =1)
        # print(central_mac_out[:, :-1].shape) #torch.Size([128, 1, 2, 3, 1])
        # print(actions.shape) #torch.Size([128, 1, 2, 1])
        central_chosen_action_qvals_agents = th.gather(central_mac_out[:, :-1], dim=3, index=actions.unsqueeze(4).repeat(1,1,1,1,self.args.central_action_embed)).squeeze(3)  # Remove the last dim

        central_target_mac_out = []
        self.target_central_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            target_agent_outs = self.target_central_mac.forward(batch, t=t)
            central_target_mac_out.append(target_agent_outs)
        central_target_mac_out = th.stack(central_target_mac_out[:], dim=1)  # Concat across time
        # Mask out unavailable actions
        central_target_mac_out[avail_actions[:, :] == 0] = -9999999  # From OG deepmarl
        # Use the Qmix max actions
        #central_target_max_agent_qvals 这个比central_chosen_action_qvals_agents还长一个time_step，后面的build_td_lambda_targets会把多的time_step去掉。
        central_target_max_agent_qvals = th.gather(central_target_mac_out[:,:], 3, cur_max_actions[:,:].unsqueeze(4).repeat(1,1,1,1,self.args.central_action_embed)).squeeze(3)
        # ---

        # Mix
        chosen_action_qvals = self.mixer(chosen_action_qvals, batch["state"][:, :-1])
        target_max_qvals = self.target_central_mixer(central_target_max_agent_qvals, batch["state"])
        #central_target_max_agent_qvals.shape torch.Size([128, 2, 2, 1])
        # print(rest_chosen_action_qvals.shape) #torch.Size([128, 1, 2])
        rest_chosen_action_qvals_ = rest_chosen_action_qvals.unsqueeze(3).repeat(1,1,1,self.args.central_action_embed).squeeze(3)
        # print(rest_chosen_action_qvals_.shape)
        Q_r = rest_chosen_action_qvals = self.rest_mixer(rest_chosen_action_qvals_, batch["state"][:,:-1])#added for RESTQ
        # print(chosen_action_qvals.shape, rest_chosen_action_qvals.shape) #torch.Size([128, 1, 1])

        # We use the calculation function of sarsa lambda to approximate q star lambda
        #这个就是weighted qmix论文里面的y_i, y_i = r + \gamma Q^*(s', \tau, argmax_u'Q_tot)
        targets = build_td_lambda_targets(rewards, terminated, mask, target_max_qvals, 
                                    self.args.n_agents, self.args.gamma, self.args.td_lambda)

        # Td-error
        """should clean this 4 lines"""
        td_error = (chosen_action_qvals - (targets.detach()))
        mask = mask.expand_as(td_error)
        # 0-out the targets that came from padded data
        masked_td_error = td_error * mask
        """should clean this 4 lines"""

        # Training central Q
        central_chosen_action_qvals = self.central_mixer(central_chosen_action_qvals_agents, batch["state"][:, :-1]) #这个就是Q^*(s,\tau, u)
        central_td_error = (central_chosen_action_qvals - targets.detach())
        central_mask = mask.expand_as(central_td_error)
        central_masked_td_error = central_td_error * central_mask
        central_loss = (central_masked_td_error ** 2).sum() / mask.sum()

        # QMIX loss with weighting
        # ws = th.ones_like(td_error) * self.args.w
        # ws = th.zeros_like(td_error)
        if self.args.hysteretic_qmix: # OW-QMIX
            ws = th.where(td_error < 0, th.ones_like(td_error)*1, th.zeros_like(td_error)) # Target is greater than current max
            w_to_use = ws.mean().item() # For logging
        else: # CW-QMIX
            is_max_action = (actions == cur_max_actions[:, :-1]).min(dim=2)[0]
            max_action_qtot = self.target_central_mixer(central_target_max_agent_qvals[:, :-1], batch["state"][:, :-1])
            qtot_larger = targets > max_action_qtot
            if self.args.condition == "max_action":
                condition = is_max_action
            elif self.args.codition == "max_larger":
                condition = is_max_action | qtot_larger
            # ws = th.where(is_max_action | qtot_larger, th.ones_like(td_error)*1, ws) # Target is greater than current max
            if self.args.resq_version == "v0":
                ws = th.where(condition, th.zeros_like(td_error),
                              th.ones_like(td_error))  # Target is greater than current max
            elif self.args.resq_version == "v2_wrong":
                ws = th.where(condition, th.zeros_like(td_error), th.ones_like(td_error))  # w_r(u\hat) = 1, w_r(u) = 0
            elif self.args.resq_version == "v2":
                ws = th.where(condition, th.ones_like(td_error), th.zeros_like(td_error))  # w_r(u\hat) = 1, w_r(u) = 0
            elif self.args.resq_version == "v3":
                ws = th.where(condition, th.zeros_like(td_error), th.ones_like(td_error))  # 对于非最优的动作就要加权，让他变小
            w_to_use = ws.mean().item() # Average of ws for logging

        # print(chosen_action_qvals.shape, chosen_action_qvals.shape, ws.shape)
        td_error = (chosen_action_qvals + ws.detach() * rest_chosen_action_qvals - (targets.detach()))
        # print("td_error.shape", td_error.shape)
        mask = mask.expand_as(td_error)
        masked_td_error = td_error * mask        # 0-out the targets that came from padded data
        qmix_loss = (masked_td_error ** 2).sum() / mask.sum()
        # print("loss", qmix_loss.item())

        """约束下q_tot(\hat(u)) > R_tot"""
        # t = (1/w_to_use) * ws.detach() * th.clamp(rest_chosen_action_qvals.detach() - chosen_action_qvals, 0) * mask
        # condition_loss = (t**2).sum()/mask.sum()

        """约束Q_r>=0的部分"""
        if self.args.resq_version == "v2_wrong":
            ws2 = 1 - ws
        if self.args.resq_version == "v2":
            ws2 = ws
        if self.args.resq_version == "v3":
            Q_r_ = th.max(Q_r, 0)[0]  # 这样如果Q_r大于0的话，会被挑出来
            noopt_loss1 = (((Q_r_ * mask) ** 2).sum()) / mask.sum()  # 要求Q_r< 0
            noopt_loss = noopt_loss1
            loss = self.args.qmix_loss * qmix_loss + self.args.central_loss * central_loss + self.args.noopt_loss * noopt_loss
        elif self.args.resq_version.find("v2")>=0:
            Q_r_ = th.min(Q_r, 0)[0] #这样如果Q_r小于0的话，会被挑出来
            noopt_loss1 = (((ws2.detach() * Q_r_ * mask) **2).sum()) /mask.sum() #要求Q_r(max) > 0
            t = th.min(chosen_action_qvals - Q_r, 0)[0]#额外引入约束，约束Q_tot > Q_r, 这个是我在v2上额外增加的要求，应该不影响性质
            noopt_loss2 = ((ws2.detach() * t)**2).sum()/mask.sum() #要求Q_tot(max) > Q_r(max)
            noopt_loss = noopt_loss1 + noopt_loss2
            loss = self.args.qmix_loss * qmix_loss + self.args.central_loss * central_loss + self.args.noopt_loss * noopt_loss
        elif self.args.resq_version == "v0":
            # is_max_action = (actions == cur_max_actions[:, :-1]).min(dim=2)[0]
            # ws2 = th.where(is_max_action, th.zeros_like(td_error), th.ones_like(td_error))  # 对于非最优的动作就要加权，让他变小
            # noopt_loss = ((ws2*chosen_action_qvals * mask) **2).sum()/mask.sum()
            # The weightings for the different losses aren't used (they are always set to 1)
            loss = self.args.qmix_loss * qmix_loss + self.args.central_loss * central_loss
                   # + 0.001 * noopt_loss

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
            if self.args.resq_version == "v2":
                self.logger.log_stat("noopt_loss", noopt_loss.item(), t_env)
                self.logger.log_stat("noopt_loss1", noopt_loss1.item(), t_env)
                self.logger.log_stat("noopt_loss2", noopt_loss2.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm, t_env)
            self.logger.log_stat("mixer_norm", mixer_norm, t_env)
            self.logger.log_stat("agent_norm", agent_norm, t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat("td_error_abs", (masked_td_error.abs().sum().item()/mask_elems), t_env)
            self.logger.log_stat("q_taken_mean", (chosen_action_qvals * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat("target_mean", (targets * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat("central_loss", central_loss.item(), t_env)
            self.logger.log_stat("w_to_use", w_to_use, t_env)
            self.log_stats_t = t_env

            # print estimated matrix
            if self.args.env == "one_step_matrix_game":
                # print_matrix_status(batch, self.central_mixer, mac_out)
                # print_matrix_status(batch, self.mixer, mac_out, hidden=None, max_q_i=None, is_wqmix=True, wqmix_central_mixer=self.central_mixer, rest_mixer=self.rest_mixer)
                print_matrix_status(batch, self.mixer, mac_out, hidden=None, max_q_i=None, is_wqmix=True, wqmix_central_mixer=self.central_mixer, rest_mixer=self.rest_mixer, central_mac_out=central_mac_out, rest_mac_out=rest_mac_out)

    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        self.rest_target_mac.load_state(self.rest_mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
            self.rest_target_mixer.load_state_dict(self.rest_mixer.state_dict())
        if self.central_mac is not None:
            self.target_central_mac.load_state(self.central_mac)
        self.target_central_mixer.load_state_dict(self.central_mixer.state_dict())
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
        if self.central_mac is not None:
            self.central_mac.cuda()
            self.target_central_mac.cuda()
        self.central_mixer.cuda()
        self.target_central_mixer.cuda()

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
