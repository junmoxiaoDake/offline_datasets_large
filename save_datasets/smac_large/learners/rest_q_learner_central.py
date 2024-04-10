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
from modules.mixers.qatten import QattenMixer
from torch.optim import Adam

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

def get_ws(resq_version, condition, qvals):
    if resq_version == "v0":
        ws = th.where(condition, th.zeros_like(qvals),
                      th.ones_like(qvals))  # Target is greater than current max # w_r(u\hat) = 0, w_r(u) = 1
    elif resq_version == "v2_wrong":
        ws = th.where(condition, th.zeros_like(qvals), th.ones_like(qvals))  # w_r(u\hat) = 0, w_r(u) = 1
    elif resq_version in ["v2", "v4", "v5", "v6", "v5-"]:
        ws = th.where(condition, th.ones_like(qvals), th.zeros_like(qvals))  # w_r(u\hat) = 1, w_r(u) = 0
    elif resq_version == "v3":
        ws = th.where(condition, th.zeros_like(qvals), th.ones_like(qvals))  # 对于非最优的动作就要加权，让他变小
    return ws

class RestQLearnerCentral:

    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.logger = logger

        self.mac_params = list(mac.parameters())
        self.params = list(self.mac.parameters())

        self.last_target_update_episode = 0
        self.is_res_qmix = getattr(args, 'res', False) #added 20220825

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

        # Central Q
        # TODO: Clean this mess up!
        self.central_mac = None
        args.is_res_mixer = True;  # added 20220502
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

        args.is_res_mixer = False
        if self.args.central_mixer in ["ff", "atten", "vdn", "qmix"]:
            if self.args.central_loss == 0:
                self.central_mixer = self.mixer
                self.central_mac = self.mac
                self.target_central_mac = self.target_mac
            else:
                if self.args.central_mixer == "ff":
                    self.central_mixer = QMixerCentralFF(args) # Feedforward network that takes state and agent utils as input
                elif self.args.central_mixer == "vdn":
                    self.central_mixer = VDNMixer()
                elif self.args.central_mixer == "qmix":
                    self.central_mixer = QMixer(args)
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

        if hasattr(self, "optimizer"):
            if getattr(self, "optimizer") == "Adam":
                self.optimiser = Adam(params=self.params, lr=args.lr, eps=args.optim_eps)
        else:
            self.optimiser = RMSprop(params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)

        self.log_stats_t = -self.args.learner_log_interval - 1

        self.grad_norm = 1
        self.mixer_norm = 1
        self.mixer_norms = deque([1], maxlen=100)

    def softmax_weighting(self, q_vals):
        assert q_vals.shape[-1] != 1

        max_q_vals = th.max(q_vals, -1, keepdim=True)[0]
        norm_q_vals = q_vals - max_q_vals
        e_beta_normQ = th.exp(self.args.res_beta * norm_q_vals)

        numerators = e_beta_normQ
        denominators = th.sum(e_beta_normQ, -1, keepdim=True)

        softmax_weightings = numerators / denominators

        return softmax_weightings

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        share_Qi =  getattr(self.args, 'share_Qi', False)
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
        if not share_Qi:
            self.rest_mac.init_hidden(batch.batch_size)
            for t in range(batch.max_seq_length):
                rest_agent_outs = self.rest_mac.forward(batch, t=t)
                rest_mac_out.append(rest_agent_outs)
            rest_mac_out = th.stack(rest_mac_out, dim=1)  # Concat over time
            rest_chosen_action_qvals = th.gather(rest_mac_out[:, :-1], dim=3, index=actions).squeeze(3)
        else: #20220818, use share_qi
            rest_mac_out = mac_out;
            rest_chosen_action_qvals = chosen_action_qvals_agents;

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
        if not share_Qi:
            self.rest_target_mac.init_hidden(batch.batch_size)
            for t in range(batch.max_seq_length):
                rest_target_agent_outs = self.rest_target_mac.forward(batch, t=t)
                rest_target_mac_out.append(rest_target_agent_outs)
            # We don't need the first timesteps Q-Value estimate for calculating targets
            rest_target_mac_out = th.stack(rest_target_mac_out[:], dim=1)  # Concat across time
            # Mask out unavailable actions
            rest_target_mac_out[avail_actions[:, :] == 0] = -9999999  # From OG deepmarl
        else: #add 20220818
            rest_target_mac_out = target_mac_out;

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
        if not share_Qi:
            self.central_mac.init_hidden(batch.batch_size)
            for t in range(batch.max_seq_length):
                agent_outs = self.central_mac.forward(batch, t=t)
                central_mac_out.append(agent_outs)
            central_mac_out = th.stack(central_mac_out, dim=1)  # Concat over time
            # print(central_mac_out.shape) #mac_out batch_size, seq_length, n_agents, n_commands, args.central_action_embed(default =1)
            # print(central_mac_out[:, :-1].shape) #torch.Size([128, 1, 2, 3, 1])
            # print(actions.shape) #torch.Size([128, 1, 2, 1])
            central_chosen_action_qvals_agents = th.gather(central_mac_out[:, :-1], dim=3, index=actions.unsqueeze(4).repeat(1,1,1,1,self.args.central_action_embed)).squeeze(3)  # Remove the last dim
        else:
            central_mac_out = mac_out;
            central_chosen_action_qvals_agents = chosen_action_qvals_agents

        central_target_mac_out = []
        if not share_Qi:
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
        else:
            central_target_mac_out = target_mac_out;
            central_target_max_agent_qvals = target_max_agent_qvals
            # target_max_agent_qvals = th.gather(target_mac_out[:,:], 3, cur_max_actions[:,:]).squeeze(3)

        if self.is_res_qmix:  #modified from res-qmix 20220825
            central_mac_out_detach = central_mac_out.detach()
            all_counterfactual_actions_qvals = []
            all_counterfactual_actions_target_qvals = []
            for agent_idx in range(cur_max_actions.shape[2]):
                base_actions = copy.deepcopy(cur_max_actions) # cur_max_actions (batch, T, agents, 1)
                # total_batch_size, num_agents
                base_actions = base_actions.squeeze(-1).reshape(-1, cur_max_actions.shape[2]) #(batch_size * t, agents)

                # num_actions, 1
                all_actions_for_an_agent = th.tensor(
                    [action_idx for action_idx in range(self.args.n_actions)]).unsqueeze(0)
                # num_actions, total_batch_size: [[0, ..., 0], [1, ..., 1], ..., [4, ..., 4]]
                all_actions_for_an_agent = all_actions_for_an_agent.repeat(base_actions.shape[0], 1).transpose(1, 0) #(n_actions, batch_size * t)
                # formate to a column vector: total_batch_size x num_actions: [0, ..., 0, ...., 4, ..., 4]
                all_actions_for_an_agent = all_actions_for_an_agent.reshape(-1, 1).squeeze() #(n_actions * batch_size * t)

                # total_batch_size x num_agents, num_actions (repeat the actions for num_actions times)
                counterfactual_actions = base_actions.repeat(self.args.n_actions, 1).reshape(-1, base_actions.shape[1]) #(n_action*batch_size*t, agents)

                counterfactual_actions[:, agent_idx] = all_actions_for_an_agent

                counterfactual_actions_qvals, counterfactual_actions_target_qvals = [], []
                for action_idx in range(self.args.n_actions):
                    curr_counterfactual_actions = counterfactual_actions[
                                                  action_idx * base_actions.shape[0]: (action_idx + 1) *
                                                                                      base_actions.shape[0]]
                    curr_counterfactual_actions = curr_counterfactual_actions.reshape(cur_max_actions.shape[0],
                                                                                      cur_max_actions.shape[1],
                                                                                      cur_max_actions.shape[2], -1) #(batch_size, t, agents, 1)

                    # batch_size, episode_len, num_agents
                    # print("central_mac_out_detach.shape", central_mac_out_detach.shape) #central_mac_out_detach.shape torch.Size([32, 61, 3, 9, 1])
                    # print("curr_counterfactual_actions.shape", curr_counterfactual_actions.shape) #curr_counterfactual_actions.shape torch.Size([32, 61, 3, 1])
                    t = central_mac_out_detach.squeeze(-1);
                    curr_counterfactual_actions_qvals = th.gather(t, dim=3, index=curr_counterfactual_actions).squeeze(3)  # Remove the last dim
                    # print("central_target_mac_out.shape", central_target_mac_out.shape) #curr_counterfactual_actions.shape torch.Size([32, 61, 3, 1])
                    t = central_target_mac_out.squeeze(-1);
                    curr_counterfactual_actions_target_qvals = th.gather(t, dim=3, index=curr_counterfactual_actions).squeeze(3)  # Remove the last dim

                    # batch_size, episode_len, 1
                    curr_counterfactual_actions_qvals = self.central_mixer(
                        curr_counterfactual_actions_qvals, batch["state"]
                    )
                    curr_counterfactual_actions_qvals = curr_counterfactual_actions_qvals.reshape(
                        curr_counterfactual_actions_qvals.shape[0] * curr_counterfactual_actions_qvals.shape[1], 1
                    )
                    # print("curr_counterfactual_actions_target_qvals.mean", curr_counterfactual_actions_target_qvals.mean().detach().cpu().numpy())
                    curr_counterfactual_actions_target_qvals = self.target_central_mixer(
                        curr_counterfactual_actions_target_qvals, batch["state"]
                    )
                    # print("curr_counterfactual_actions_target_qvals.mean after", curr_counterfactual_actions_target_qvals.mean().detach().cpu().numpy())

                    curr_counterfactual_actions_target_qvals = curr_counterfactual_actions_target_qvals.reshape(
                        curr_counterfactual_actions_target_qvals.shape[0] *
                        curr_counterfactual_actions_target_qvals.shape[1], 1
                    )

                    counterfactual_actions_qvals.append(curr_counterfactual_actions_qvals)
                    counterfactual_actions_target_qvals.append(curr_counterfactual_actions_target_qvals)

                # batch_size x episode_len, num_actions
                counterfactual_actions_qvals = th.cat(counterfactual_actions_qvals, 1)
                counterfactual_actions_target_qvals = th.cat(counterfactual_actions_target_qvals, 1)

                all_counterfactual_actions_qvals.append(counterfactual_actions_qvals)
                all_counterfactual_actions_target_qvals.append(counterfactual_actions_target_qvals)

            # total_batch_size, num_agents, num_actions
            all_counterfactual_actions_qvals = th.stack(all_counterfactual_actions_qvals).permute(1, 0, 2)
            all_counterfactual_actions_target_qvals = th.stack(all_counterfactual_actions_target_qvals).permute(1, 0, 2)

            # total_batch_size, num_agents x num_actions
            all_counterfactual_actions_qvals = all_counterfactual_actions_qvals.reshape(all_counterfactual_actions_qvals.shape[0], -1)
            all_counterfactual_actions_target_qvals = all_counterfactual_actions_target_qvals.reshape(all_counterfactual_actions_target_qvals.shape[0], -1)

            softmax_weightings = self.softmax_weighting(all_counterfactual_actions_qvals)
            # print("softmax_weightings.shape", softmax_weightings.shape) #softmax_weightings.shape torch.Size([1856, 180])
            # print("softmax_weightings.mean", softmax_weightings.mean().detach().cpu().numpy())
            softmax_qtots = softmax_weightings * all_counterfactual_actions_target_qvals
            # print("all_counterfactual_actions_target_qvals.mean", all_counterfactual_actions_target_qvals.mean().detach().cpu().numpy())
            # print("softmax_qtots.mean", softmax_qtots.mean().detach().cpu().numpy(), "softmax_qtots.shape", softmax_qtots.shape)
            # softmax_qtots = th.sum(softmax_qtots, 1, keepdim=True)
            softmax_qtots = th.sum(softmax_qtots, 1, keepdim=True)
            # print("be", all_counterfactual_actions_target_qvals.max(axis=1)[0].mean().detach().cpu().numpy())
            # print("af", softmax_qtots.mean().detach().cpu().numpy())
            t = softmax_qtots.reshape(rewards.shape[0], rewards.shape[1]+1, rewards.shape[2])
            # softmax_qtots = t[:, :-1];
            target_max_qvals = t
            v2 = self.target_central_mixer(central_target_max_agent_qvals, batch["state"])
            # print("target_max_qvals.mean()", target_max_qvals.mean().detach().cpu().numpy(), "target_central_mixer.mean", v2.mean().detach().cpu().numpy());
                       # print("target_max_qvals.mean()", target_max_qvals.mean(), "target_central_mixer.mean", v2.mean());

        else: #added to here 20220825, it was the default implementation
             target_max_qvals = self.target_central_mixer(central_target_max_agent_qvals, batch["state"])
             # print("target_max_qvals.mean()", target_max_qvals.mean().detach().cpu().numpy());

        # Mix

        chosen_action_qvals = self.mixer(chosen_action_qvals, batch["state"][:, :-1])
        #central_target_max_agent_qvals.shape torch.Size([128, 2, 2, 1])
        # print(rest_chosen_action_qvals.shape) #torch.Size([128, 1, 2])
        rest_chosen_action_qvals_ = rest_chosen_action_qvals.unsqueeze(3).repeat(1,1,1,self.args.central_action_embed).squeeze(3)
        # print(rest_chosen_action_qvals_.shape)
        Q_r = rest_chosen_action_qvals = self.rest_mixer(rest_chosen_action_qvals_, batch["state"][:,:-1])#added for RESTQ
        negative_abs = getattr(self.args, 'residual_negative_abs', False)
        if negative_abs:
            Q_r = - Q_r.abs()

        # We use the calculation function of sarsa lambda to approximate q star lambda
        #这个就是weighted qmix论文里面的y_i, y_i = r + \gamma Q^*(s', \tau, argmax_u'Q_tot)

        # target_max_qvals = target_Q_tot + w_r_target * target_Q_r
        # targets = rewards + self.args.gamma * (1 - terminated) * target_max_qvals
        # print("target_max_qvals.shape", target_max_qvals.shape)
        targets = build_td_lambda_targets(rewards, terminated, mask, target_max_qvals,
                                    self.args.n_agents, self.args.gamma, self.args.td_lambda)
        # print("targets.shape", targets.shape)
        # print("rewards.shape", rewards.shape)
        # Td-error
        """should clean this 4 lines"""
        # td_error = (chosen_action_qvals - (targets.detach())) #this is actually useless
        mask = mask.expand_as(chosen_action_qvals)
        # 0-out the targets that came from padded data
        # masked_td_error = td_error * mask
        """should clean this 4 lines"""

        # Training central Q
        central_chosen_action_qvals = self.central_mixer(central_chosen_action_qvals_agents, batch["state"][:, :-1]) #这个就是Q^*(s,\tau, u)
        central_td_error = (central_chosen_action_qvals - targets.detach())
        central_mask = mask.expand_as(central_td_error)
        central_masked_td_error = central_td_error * central_mask
        central_loss = (central_masked_td_error ** 2).sum() / central_mask.sum()

        # QMIX loss with weighting
        # ws = th.ones_like(td_error) * self.args.w
        # ws = th.zeros_like(td_error)
        if self.args.hysteretic_qmix: # OW-QMIX
            print("use less") #commented out 20220805
            # w_r = th.where(td_error < 0, th.ones_like(td_error)*1, th.zeros_like(td_error)) # Target is greater than current max
            # w_to_use = w_r.mean().item() # For logging
        else: # CW-QMIX
            is_max_action = (actions == cur_max_actions[:, :-1]).min(dim=2)[0]
            max_action_qtot = self.target_central_mixer(central_target_max_agent_qvals[:, :-1], batch["state"][:, :-1])
            qtot_larger = targets > max_action_qtot
            if self.args.condition == "max_action":
                condition = is_max_action
            elif self.args.condition == "max_larger":
                condition = is_max_action | qtot_larger
            # ws = th.where(is_max_action | qtot_larger, th.ones_like(td_error)*1, ws) # Target is greater than current max
            nomask = getattr(self.args, 'nomask', False)
            if nomask:
                w_r = th.ones_like(chosen_action_qvals);
            else:
                w_r = get_ws(self.args.resq_version, condition, chosen_action_qvals)
            w_to_use = w_r.mean().item() # Average of ws for logging

        # print(chosen_action_qvals.shape, chosen_action_qvals.shape, ws.shape)
        current_Q_tot = chosen_action_qvals + w_r.detach() * rest_chosen_action_qvals;
        td_error = current_Q_tot - (targets.detach())
        # print("td_error.shape", td_error.shape)
        mask = mask.expand_as(td_error)
        masked_td_error = td_error * mask        # 0-out the targets that came from padded data
        qmix_loss = (masked_td_error ** 2).sum() / mask.sum()
        # print("loss", qmix_loss.item())

        """约束下q_tot(\hat(u)) > R_tot"""
        # t = (1/w_to_use) * ws.detach() * th.clamp(rest_chosen_action_qvals.detach() - chosen_action_qvals, 0) * mask
        # condition_loss = (t**2).sum()/mask.sum()

        """约束Q_r>=0的部分"""
        noopt_loss2 = None
        if self.args.resq_version == "v2_wrong":
            w_r2 = 1 - w_r
        if self.args.resq_version in ["v2", "v4", "v5"]:
            w_r2 = w_r
        if self.args.resq_version in ["v3"]:  #w_r(u\hat) = 0, w_r(非最优) = 1, 要求Q_r小于0
            #added 20220317，如果在Q_r里面用了abs的话， Q_r肯定是正的，所以，这里面的代码noopt_loss肯定是0了，还是不改learner的代码，避免有麻烦。
            Q_r_ = th.max(Q_r, th.zeros_like(Q_r))
            #邱梦薇发现，原来的Q_r_ = th.max(Q_r, 0)[0] 是错误的，原来这个语句是让
            noopt_loss1 = (((Q_r_ * mask) ** 2).sum()) / mask.sum()  # 要求Q_r< 0
            noopt_loss = noopt_loss1
            loss = self.args.qmix_loss * qmix_loss + self.args.central_loss * central_loss + self.args.noopt_loss * noopt_loss
        elif self.args.resq_version == "v5-": #这个版本就是actual_v2的版本，要求所有的Q_r>0，这个也是完美的和v3先对应的版本。
            Q_r_ = th.min(Q_r, 0)[0]  # 这样如果Q_r小于0的话，会被挑出来
            noopt_loss1 = (((Q_r_ * mask) ** 2).sum()) / mask.sum()  # 要求Q_r > 0
            noopt_loss = noopt_loss1
            loss = self.args.qmix_loss * qmix_loss + self.args.central_loss * central_loss + self.args.noopt_loss * noopt_loss
        elif self.args.resq_version in ["v2_wrong", "v2", "v4", "v5"]:
            gap = self.args.max_second_gap
            if self.args.condition_loss == "mse":
                Q_r_ = th.min(Q_r, gap)[0]  # 这样如果Q_r小于gap的话(v2里面gap是0，意味着挑小于0的)，会被挑出来
                t0 = w_r2.detach() * Q_r_ * mask  #ws2这个时候是将max action 对应的Q挑出来
                noopt_loss1 = ((t0 ** 2).sum()) /mask.sum() #要求Q_r(max) > 0
            elif self.args.condition_loss == "delta":
                t0 = th.where(Q_r<gap, th.ones_like(Q_r), th.zeros_like(Q_r))
                noopt_loss1 = (w_r2.detach() * t0 * mask).sum() * self.args.condition_loss_delta /mask.sum() #要求Q_r(max) > 0

            if self.args.resq_version in ["v2"]:
                if self.args.condition_loss == "mse":
                    t = th.min(chosen_action_qvals - Q_r, 0)[0]  # 额外引入约束，约束Q_tot > Q_r, 这个是我在v2上额外增加的要求，应该不影响性质
                    t1 = (w_r2.detach() * t)
                    noopt_loss2 = (t1**2).sum()/mask.sum() #要求Q_tot(max) > Q_r(max)  #TODO 这里有bug，其实根本没有约束上，这里是noopt_loss2
                elif self.args.condition_loss == "delta":
                    t0 = th.where( (chosen_action_qvals - Q_r)<0, th.ones_like(Q_r), th.zeros_like(Q_r))
                    noopt_loss1 = (w_r2.detach() * t0 * mask).sum() * self.args.condition_loss_delta /mask.sum() #要求Q_r(max) > 0

            if noopt_loss2 is not None:
                noopt_loss = noopt_loss1 + noopt_loss2
            else:
                noopt_loss = noopt_loss1
            loss = self.args.qmix_loss * qmix_loss + self.args.central_loss * central_loss + self.args.noopt_loss * noopt_loss
        elif self.args.resq_version == "v0":
            # is_max_action = (actions == cur_max_actions[:, :-1]).min(dim=2)[0]
            # ws2 = th.where(is_max_action, th.zeros_like(td_error), th.ones_like(td_error))  # 对于非最优的动作就要加权，让他变小
            # noopt_loss = ((ws2*chosen_action_qvals * mask) **2).sum()/mask.sum()
            # The weightings for the different losses aren't used (they are always set to 1)
            loss = self.args.qmix_loss * qmix_loss + self.args.central_loss * central_loss
                   # + 0.001 * noopt_loss

        if self.is_res_qmix:
            future_episode_return = batch["future_discounted_return"][:, :-1]
            q_return_diff = (central_chosen_action_qvals - future_episode_return.detach()) #central_chosen_action_qvals可能要改为targets?

            v_l2 = ((q_return_diff * mask) ** 2).sum() / mask.sum()
            # print()
            # print("v_l2 loss", v_l2.mean().detach().cpu().numpy(), "qmix loss", qmix_loss.mean().detach().cpu().numpy(), "central_loss", central_loss.mean().cpu().detach())
            loss += self.args.res_lambda * v_l2
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
            if noopt_loss is not None:
                self.logger.log_stat("noopt_loss", noopt_loss.item(), t_env)
            if noopt_loss1 is not None:
                self.logger.log_stat("noopt_loss1", noopt_loss1.item(), t_env)
            if noopt_loss2 is not None:
                self.logger.log_stat("noopt_loss2", noopt_loss2.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm, t_env)
            self.logger.log_stat("mixer_norm", mixer_norm, t_env)
            self.logger.log_stat("agent_norm", agent_norm, t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat("td_error_abs", (masked_td_error.abs().sum().item()/mask_elems), t_env)
            self.logger.log_stat("q_taken_mean", (chosen_action_qvals * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat("target_mean", (targets * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat("current Q_tot", (current_Q_tot * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat("central_loss", central_loss.item(), t_env)
            self.logger.log_stat("w_to_use", w_to_use, t_env)
            self.log_stats_t = t_env

            # print estimated matrix
            if self.args.env == "one_step_matrix_game":
                # print_matrix_status(batch, self.central_mixer, mac_out)
                # def print_matrix_status(t_env, logger_boss, batch, mixer, mac_out, hidden=None, max_q_i=None,
                #                         is_wqmix=False, wqmix_central_mixer=None, rest_mixer=None, central_mac_out=None,
                #                         rest_mac_out=None):
                print_matrix_status(t_env, self.logger, batch, self.mixer, mac_out, hidden=None, max_q_i=None, is_wqmix=True, wqmix_central_mixer=self.central_mixer, rest_mixer=self.rest_mixer, central_mac_out=central_mac_out, rest_mac_out=rest_mac_out)
                # print_matrix_status(t_env, self.logger, batch, self.mixer, mac_out, hidden=None, max_q_i=None, is_wqmix=True, wqmix_central_mixer=self.central_mixer, rest_mixer=self.rest_mixer)
                # print_matrix_status(batch, self.mixer, mac_out, hidden=None, max_q_i=None, is_wqmix=True, wqmix_central_mixer=self.central_mixer, rest_mixer=self.rest_mixer, central_mac_out=central_mac_out, rest_mac_out=rest_mac_out)

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
