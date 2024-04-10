from envs.multiagentenv import MultiAgentEnv
from utils.dict2namedtuple import convert
import numpy as np
import torch as th
from modules.mixers.qtran import QTranBase
from modules.mixers.dmaq_general import DMAQer
from modules.mixers.dmaq_si_weight import DMAQ_SI_Weight


from components.transforms import OneHot

# this non-monotonic matrix can be solved by qmix
# payoff_values = [[12, -0.1, -0.1],
#                     [-0.1, 0, 0],
#                     [-0.1, 0, 0]]

payoff_values = [[8, -12, -12],
                    [-12, 0, 0],
                    [-12, 0, 0]]

payoff_values = [[8, 7.9, -12],
                    [-12, 0, 0],
                    [-12, 0, 0]]
#cw qmix can reconstruct such matrix
payoff_values = [[8, -12, -12],
                    [8.1, 0, 0],
                    [-12, 0, 0]]
#cw qmix can reconstruct such matrix
payoff_values = [[8, -12, -12],
                    [-12, 0, 0],
                    [-12, 0, 7.9]]
#very interesting result, 8 is well constructed, but 7.9 is not, this suggest that it may suffer numerical instability
payoff_values = [[8, -12, -12],
                    [-12, 0, 0],
                    [-12, 0, 7.99]]
#very interesting result, 8 is well constructed, but 7.9 is not, this suggest that it may suffer numerical instability
# tensor([[  8.0, -11.9, -11.9],
#        [-11.9, -11.9, -11.9],
#        [-11.9, -11.9, -11.9]])

payoff_values = [[8, -12, -12],
                    [-12, 0, 0],
                    [-12, 0, 7.999]]

# tensor([[-11.9, -11.9, -11.9],
#         [-11.9, -3.3, 0.0],
#         [-11.9, 0.0, 8.0]])

payoff_values = [[8, -12, -12],
                    [-12, 0, 0],
                    [-12, 0, 9]]
#这个很容易, v2的可以重建
payoff_values = [[2.5, 0, -100],
                    [0, 2, 0],
                    [0, 0, 3]]
#这个qmix, ow qmix, cw qmix效果都不好，但是qtran和qplex都不错，qplex基本上恢复了

payoff_values = [[8, -12, -12],
                    [-12, 0, 0],
                    [-12, 0, 0]]
#这个是weight QMIX用来举例的，非monotonic的例子, restq_v2_central可以

payoff_values = [[2.5, 0, -100],
                    [0, 2, 0],
                    [-100, 0, 3]]
#只有Qtran cw成功， ow, qplex失败, rest_q_v2_wrong_central有时候可以, rest_v2_central不行, v3_central可以



# payoff_values = [[2.5, 0, -100],
#                     [0, 2, 0],
#                     [-100, -100, 3]]
#cw ow qplex qtran都失败了, restQ可以




payoff_values = [[8, -12, -12],
                    [-12, 0, 0],
                    [-12, 0, 0]]
#v2_wrong_central可以建模，v2_central可以建模, v3_central也可以

payoff_values = [[2.5, 0, -100],
                    [0, 2, 0],
                    [-100, 0, 3]]
#v2不能， v2 wrong也不能， v4 gap 1不行, v4 gap 2也不行。v3可以

payoff_values = [[2.5, 0, -100],
                    [0, 2, 0],
                    [-100, -100, 3]]
#cw qmix, ow qmix qplex, qmix都不行, qtran和v3可以


payoff_values = [[8, -12, 7.9],
                    [-12, 7.9, 0],
                    [-12, 0, 7.9]]
#qplex, v3, qtran可以，ow, cw不行

payoff_values = [[8, -12, -12],
                    [-12, 0, 0],
                    [0, 0, 0]]
#这个很简单, qplex很快搞定

payoff_values = [[8, -12, -12],
                    [-12, 0, 0],
                    [-12, 0, 0]]


payoff_values = [[8, -12, -12],
                    [-12, 7.999, 0],
                    [-12, 0, 0]]
#ow, qplex不可以，qtran, v3-b32-small, cw可以

payoff_values = [[8, -12, -12],
                    [-12, 0, 0],
                    [-12, 0, 7.999]]
#cw ow  qtran，qplex都失败了 v3-b32-small可以， resq_v2_central目前不行， v3_central可以

payoff_values = [[8, -12, -12],
                    [-12, 0, 7.999],
                    [-12, 0, 0]]
#cw可以, qtran, ow, cw不行, v3-b32-small有时候可以。v3-b32-small-delta01可以

payoff_values = [[8, -1000, 7],
                    [-12, 0, 0],
                    [-12, 0, 0]]
#qplex可以

payoff_values = [[8, -1000, -12],
                    [-12, 0, 0],
                    [-12, 0, 0]]
#qplex可以
payoff_values = [[8.1, -12, -12],
                    [-12, 7.9, 0],
                    [-12, 0, 8]]
#qplex, ow不行，cw, qtran, v3-b32-small, v3-b32-small-delta01可以

payoff_values = [[8.1, -12, -12],
                    [-12, 7.9, 0],
                    [8.01, 0, 8]]
#qplex, cw, ow不行， qtran, v32-b32-small v32-b32-small-delta01可以


payoff_values = [[8.1, -12, -12],
                    [-12, 7.9, -12],
                    [0, 0, 8]]
#这个比较容易


payoff_values = [[8, -12, -12],
                    [-12, -13, 0],
                    [-12, 0, 7.999]]
#qtran有时候可以，有时候不行，v3-c-b32-small有时候可以，有时候不行。


payoff_values = [[8, -12, -12],
                    [-12, 0, 0],
                    [-12, 0, 7.999]]

payoff_values = [[8, -12, -12],
                    [-12, 0, 0],
                    [-12, 0, 7.9]]


payoff_values = [[8000, -12000, -12000],
                    [-12000, 0, 0],
                    [-12000, 0, 7999]]

payoff_values = [[8, -12, -12],
                    [-12, 0, 0],
                    [-12, 0, 7.999]]

payoff_values = [[8, -12, -12],
                    [-12, 0, 0],
                    [-12, 0, 7.999]]


payoff_values = [[8000, -12000, -12000],
                    [-12000, 0, 0],
                    [-12000, 0, 7999]]

payoff_values = [[80000, -120000, -120000],
                    [-120000, 0, 0],
                    [-120000, 0, 79990]]

# payoff_values = [[8, -12, -12],
#                   [-12, 0, 0],
#                 [-12, 0, 7.9]]
#
payoff_values = [[8, -12, -120],
                  [-12, 7.8, 7.7],
                [-120, -130, 7.9]]

payoff_values = [[8000, -12000, -12000],
                    [-12000, 0, 0],
                    [-12000, 0, 7999]]

# payoff_values = [[8, -12, -12],
#                     [-12, 0, 0],
#                     [-12, 0, 7.999]]
payoff_values = [[8, -12, -12],
                  [-12, 7.8, 0],
                [-12, 0, 7.9]]
#QTran可以重建这个矩阵



# payoff_values = [[8, -12, -12],
#                     [-12, 0, 0],
#                     [-12, 0, 7.9]]


payoff_values = [[8, 6, 4],
                    [6, 4, 3],
                    [4, 3, 1]]

payoff_values = [[8, -12, -12],
                  [-12, 0, 0],
                [-12, 0, 7.9]]






payoff_values = [[8, -12, -12],
                  [-12, 0, 0],
                [-12, 0, 0]]

payoff_values = [[8, -12, -12],
                  [-12, 0, 0],
                [-12, 0, 7.9]]


# payoff_values = [[8, 4, 1],
#                   [-9, -4, -1],
#                 [0, 0, 0]]

payoff_values = [[2.5, 0, -100],
                    [0, 2, 0],
                    [-100, -100, 3]]


payoff_values = [[8, -12, -12],
                    [-12, 0, 0],
                    [-12, 0, 7.9]]

# payoff_values = [[2.5, 0, -100],
#                     [0, 2, 0],
#                     [-100, -100, 3]]


# payoff_values = [[8, -12, -120],
#                   [-12, 7.8, 7.7],
#                 [-120, -130, 7.9]]


std_values = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
#cw ow  qtran，qplex都失败了 v3-b32-small可以， resq_v2_central目前不行， v3-c-b32-small可以(有时候不行。。。)



#
# payoff_values = [[2.5, 0, -100],
#                     [0, 2, 0],
#                     [1, -100, 3]]

# payoff_values = [[12, -12, -12],
#                     [-12, 0, 0],
#                     [-12, 0, 0]]

# payoff_values = [[12, 0, 10],
#                     [0, 0, 10],
#                     [10, 10, 10]]


# payoff_values = [[8, -12, -12],
#                     [-12, 0, 0],
#                     [-12, 0, 7.999]]
#在CW，Ow下面，结果也不是不大好，有时候能够回复出来，有时候认为7.99才是argmax

# payoff_values = [[1, 0], [0, 1]]
# n_agents = 3
# payoff_values = np.zeros((n_agents, n_agents))
# for i in range(n_agents):
#     payoff_values[i, i] = 1

is_dist = True
class OneStepMatrixGame(MultiAgentEnv):
    def __init__(self, batch_size=None, **kwargs):
        # Define the agents
        self.n_agents = 2
        map_name = kwargs["map_name"]
        self.distributional = False
        if map_name.find("_dist") >=0:
            self.distributional = True
            is_dist = True
        # Define the internal state
        self.steps = 0
        self.n_actions = len(payoff_values[0])
        self.episode_limit = 1
        self.stds = np.array(std_values)

    def reset(self):
        """ Returns initial observations and states"""
        self.steps = 0
        return self.get_obs(), self.get_state()

    def step(self, actions):
        """ Returns reward, terminated, info """
        mean_reward = payoff_values[actions[0]][actions[1]]
        if self.distributional:
            reward = mean_reward + np.random.random() * std_values[actions[0]][actions[1]];
        else:
            reward = mean_reward
        self.steps = 1
        terminated = True

        info = {}
        return reward, terminated, info

    def get_obs(self):
        """ Returns all agent observations in a list """
        one_hot_step = np.zeros(2)
        # one_hot_step[self.steps] = 1  #之前这一步导致了问题所在，导致训练不好。#20220804，好像也不对
        return [np.copy(one_hot_step) for _ in range(self.n_agents)]

    def get_obs_agent(self, agent_id):
        """ Returns observation for agent_id """
        return self.get_obs()[agent_id]

    def get_obs_size(self):
        """ Returns the shape of the observation """
        return len(self.get_obs_agent(0))

    def get_state(self):
        return self.get_obs_agent(0)

    def get_state_size(self):
        """ Returns the shape of the state"""
        return self.get_obs_size()

    def get_avail_actions(self):
        avail_actions = []
        for agent_id in range(self.n_agents):
            avail_agent = self.get_avail_agent_actions(agent_id)
            avail_actions.append(avail_agent)
        return avail_actions

    def get_avail_agent_actions(self, agent_id):
        """ Returns the available actions for agent_id """
        return np.ones(self.n_actions)

    def get_total_actions(self):
        """ Returns the total number of actions an agent could ever take """
        return self.n_actions

    def get_stats(self):
        return None

    def render(self):
        raise NotImplementedError

    def close(self):
        pass

    def seed(self):
        raise NotImplementedError
    

def print_matrix_status_distributional(batch, mixer, mac_out, n_quantiles):
    batch_size = batch.batch_size
    matrix_size = len(payoff_values)
    Q_tot = th.zeros((matrix_size, matrix_size))
    Q_jt = th.zeros((matrix_size, matrix_size))
    Q_r = th.zeros((matrix_size, matrix_size))

    with th.no_grad():
        for i in range(matrix_size):
            for j in range(matrix_size):
                #i, j are the actions of the two agents
                actions = th.LongTensor([[[[i], [j]]]]).to(device=mac_out.device).repeat(batch_size, 1, 1, 1)
                # print("actions.shape", actions.shape) #torch.Size([128, 1, 2, 1])
                if len(mac_out.size()) == 5: # n qvals
                    actions = actions.unsqueeze(-1).repeat(1, 1, 1, 1, mac_out.size(-1)) # b, t, a, actions, n
                    print("new_action.shape", actions.shape)
                qvals = th.gather(mac_out[:batch_size, 0:1], dim=3, index=actions).squeeze(3)


# for mixer methods
# print_matrix_status(t_env, batch, self.mixer, mac_out, hidden=None, max_q_i=None, is_wqmix=True, wqmix_central_mixer=self.central_mixer, rest_mixer=self.rest_mixer)

def print_matrix_status(t_env, logger_boss, batch, mixer, mac_out, hidden=None, max_q_i=None, is_wqmix=False, wqmix_central_mixer=None, rest_mixer=None, central_mac_out=None, rest_mac_out=None):
    logger = logger_boss.console_logger
    batch_size = batch.batch_size
    matrix_size = len(payoff_values)
    Q_tot = th.zeros((matrix_size, matrix_size))
    central_results = th.zeros((matrix_size, matrix_size)) #for wqmix
    Q_r = th.zeros((matrix_size, matrix_size)) #for wqmix
    qtran = np.zeros((matrix_size, matrix_size))
    qatten_v = np.zeros((matrix_size, matrix_size))
    qatten_adv = np.zeros((matrix_size, matrix_size))
    with th.no_grad():
        for i in range(Q_tot.shape[0]):
            for j in range(Q_tot.shape[1]):
                #i, j are the actions of the two agents
                actions = th.LongTensor([[[[i], [j]]]]).to(device=mac_out.device).repeat(batch_size, 1, 1, 1)
                # print("actions.shape", actions.shape) #torch.Size([128, 1, 2, 1])
                if len(mac_out.size()) == 5: # n qvals
                    actions = actions.unsqueeze(-1).repeat(1, 1, 1, 1, mac_out.size(-1)) # b, t, a, actions, n
                    print("new_action.shape", actions.shape)
                # print("mac_out.shape", mac_out.shape) #torch.Size([128, 2, 2, 3], mac_out[:,1,:,:] is the end_of_episode token (useless)
                qvals = th.gather(mac_out[:batch_size, 0:1], dim=3, index=actions).squeeze(3)
                # print("mac_out.shape", mac_out.shape)
                if central_mac_out is not None:
                    if central_mac_out.shape[-1] == 1:
                        central_mac_out = central_mac_out.squeeze(4)
                    # print("central_mac_out.shape", central_mac_out.shape)
                    central_qvals = th.gather(central_mac_out[:batch_size, 0:1], dim=3, index=actions).squeeze(3)
                if rest_mac_out is not None:
                    # print("rest_mac_out.shape", rest_mac_out.shape)
                    rest_qvals = th.gather(rest_mac_out[:batch_size, 0:1], dim=3, index=actions).squeeze(3)
                # print("q values.shape", qvals.shape) # torch.Size([128, 1, 2])
                if isinstance(mixer, QTranBase): #QTran
                    # print("actions.shape", actions.shape) #actions.shape torch.Size([128, 1, 2, 1])
                    n_actions = Q_tot.shape[0]
                    one_hot = OneHot(n_actions)
                    one_hot_actions = one_hot.transform(actions)
                    joint_qs, joint_vs = mixer(batch[:,:-1], hidden[:,:-1], one_hot_actions)
                    global_q = (joint_qs).mean() #modified 20220805
                    # global_q = (joint_qs + joint_vs).mean()
                elif isinstance(mixer, DMAQer) or isinstance(mixer, DMAQ_SI_Weight): #QPlex
                    # def forward(self, agent_qs, states, actions=None, max_q_i=None, is_v=False):
                    n_actions = Q_tot.shape[0]
                    one_hot = OneHot(n_actions)
                    one_hot_actions = one_hot.transform(actions)
                    v = mixer(qvals, batch["state"][:batch_size, 0:1], actions=one_hot_actions, is_v=True)
                    q = mixer(qvals, batch["state"][:batch_size, 0:1], actions=one_hot_actions, max_q_i=max_q_i, is_v=False)
                    qatten_v[i][j] = v.mean().item()
                    qatten_adv[i][j] = q.mean().item()
                    global_q = (v + q).mean()
                else:#qmix, ow qmix cw qmix, restQ
                    global_q = mixer(qvals, batch["state"][:batch_size, 0:1]).mean()  #for qmix, first arg is qvals, second arg is states
                    if wqmix_central_mixer is None:
                        central_results[i][j] = 0
                    else:
                        if central_mac_out is not None:
                            central_results[i][j] = wqmix_central_mixer(central_qvals, batch["state"][:batch_size, 0:1]).mean().item()  # for qmix, first arg is qvals, second arg is states
                        else:
                            central_results[i][j] = wqmix_central_mixer(qvals, batch["state"][:batch_size, 0:1]).mean().item()  # for qmix, first arg is qvals, second arg is states
                    if rest_mixer is not None:
                        if rest_mac_out is not None:
                            Q_r[i][j] = rest_mixer(rest_qvals, batch["state"][:batch_size, 0:1]).mean().item()
                        else:
                            Q_r[i][j] = rest_mixer(qvals, batch["state"][:batch_size, 0:1]).mean().item()

                Q_tot[i][j] = global_q.item()

    # th.set_printoptions(2, sci_mode=False)
    if wqmix_central_mixer is not None:
        # logger.log_stat("reconstructed_q_tot", Q_tot.numpy(), t_env)
        logger.info("reconstructed_q_tot\n" + str(Q_tot.numpy()))
        # print("reconstructed q_tot\n", Q_tot.numpy())  #这里是给Q_tot的，比如给WQMIX或者ResQ的输出Q_tot的
        if rest_mixer is not None:
            logger.info("reconstructed_rest\n" +str(Q_r.numpy()))
            # print("reconstructed rest\n", Q_r.numpy()) #如果这个是ResQ的话，应该是Q_r
            # ============== added 20220430 ===============================
            p = np.array(payoff_values)
            idx = np.where(p == np.max(p))
            actual_Q_r = Q_r.numpy()
            actual_Q_r[idx] = 0
            Q_jt = Q_tot.numpy() + actual_Q_r;
            # print("Q_jt\n", Q_jt);#这个才是ResQ的Q_jt
            logger.info("Q_jt\n"+str(Q_jt))

            logger.info("reconstructed_q^\n" + str(central_results.numpy()))

            # ============== added 20220430 ===============================
        else:
            # print("reconstructed q^\n",central_results.numpy()) #这个不是ResQ的，是WQMIX的应该是Q_jt
            logger.info("reconstructed_q^\n"+str(central_results.numpy()))

        # print("original\n", payoff_values)
        logger.info("original\n"+str(payoff_values))
        if is_dist:
            logger.info("std\n"+str(std_values))
            # print("std\n", std_values)
    else:
        if isinstance(mixer, QTranBase):  # QTran
            t = mac_out.mean(axis=0)
            # print(t.shape)
            q_i = t[0, :, :]  # the q values of Q_i
            q_i = q_i.cpu().detach().numpy()
            v = joint_vs.mean(axis=0).detach().cpu().numpy()
            for i in range(Q_tot.shape[0]):
                for j in range(Q_tot.shape[1]):
                    qtran[i][j] = q_i[0,i] +q_i[1,j] + v;
            logger.info("Q_{tran}\n" + str(qtran))
        if isinstance(mixer, DMAQer) or isinstance(mixer, DMAQ_SI_Weight):
            logger.info("Q_{plex_Adv}\n" + str((qatten_adv)))
            logger.info("Q_{plex_V}\n" + str((qatten_v)))
        logger.info("reconstructed\n"+str(Q_tot.numpy()))
        # print("reconstructed\n", Q_tot.numpy())
        if rest_mixer is not None:
            logger.info("rest\n"+str(Q_r.numpy()))
            # print("rest\n", Q_r.numpy())
        logger.info("original\n"+str(payoff_values)) #
        if is_dist:
            logger.info("std\n"+str(std_values))
        # print("original\n", payoff_values) #
    if len(mac_out.size()) == 5:
        mac_out = mac_out.mean(-1)

    t = mac_out.mean(axis=0)
    q_i = t[0, :, :]  # the q values of Q_i
    # th.set_printoptions(5, sci_mode=False)
    logger.info("Q_i\n"+str(q_i.detach().cpu().numpy()))
    # print("Q_i\n", t2.detach().cpu().numpy())
    logger.info("max_q_i\n"+str(th.argmax(q_i, dim=1)))
    # print("max_q_i\n", th.argmax(t2, dim=1))

    # print(mac_out.mean(dim=(0, 1)).detach().cpu())
    # th.set_printoptions(2, sci_mode=False)