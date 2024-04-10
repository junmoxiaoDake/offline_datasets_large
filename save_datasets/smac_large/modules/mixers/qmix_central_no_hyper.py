import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class QMixerCentralFF(nn.Module):
    def __init__(self, args):
        super(QMixerCentralFF, self).__init__()

        self.args = args
        self.is_residual_mixer = getattr(self.args, 'is_res_mixer', False)  #20220502 added
        # self.abs = getattr(self.args, 'residual_abs', False)  #20220317 added
        self.negative_abs = getattr(self.args, 'residual_negative_abs', False)  #20220430 added
        self.negative_relu = getattr(self.args, 'residual_negative_relu', False)  #20220430 added

        self.n_agents = args.n_agents
        self.state_dim = int(np.prod(args.state_shape))

        self.input_dim = self.n_agents * self.args.central_action_embed + self.state_dim
        self.embed_dim = args.central_mixing_embed_dim

        non_lin = nn.ReLU

        self.net = nn.Sequential(nn.Linear(self.input_dim, self.embed_dim),
                                 non_lin(),
                                 nn.Linear(self.embed_dim, self.embed_dim),
                                 non_lin(),
                                 nn.Linear(self.embed_dim, self.embed_dim),
                                 non_lin(),
                                 nn.Linear(self.embed_dim, 1))

        # V(s) instead of a bias for the last layers
        self.V = nn.Sequential(nn.Linear(self.state_dim, self.embed_dim),
                               non_lin(),
                               nn.Linear(self.embed_dim, 1))

    def forward(self, agent_qs, states):
        bs = agent_qs.size(0)
        states = states.reshape(-1, self.state_dim)
        agent_qs = agent_qs.reshape(-1, self.n_agents * self.args.central_action_embed)

        inputs = th.cat([states, agent_qs], dim=1)

        advs = self.net(inputs)
        vs = self.V(states)

        # if self.abs:
        #     advs = advs.abs()  #20220317 added
        #     vs = vs.abs()  #20220317 added
        #     y = advs + vs

        y = advs + vs

        if self.negative_abs and self.is_residual_mixer: #20220430 added
            y =  - y.abs()

        if self.negative_relu and self.is_residual_mixer:#20220504 added
            y = - F.leaky_relu(y)

        q_tot = y.view(bs, -1, 1)
        return q_tot
