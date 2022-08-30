import numpy as np
import torch
from torch_scatter import scatter_max
from torch.nn import Sequential as Seq, Linear, ReLU, BatchNorm1d as BN, Dropout
from models.tools import GCU, MLP


class skinnet(torch.nn.Module):
    def __init__(self, nearest_bone, use_Dg, use_Lf, aggr='max'):
        super(skinnet, self).__init__()
        self.num_nearest_bone = nearest_bone
        self.use_Dg = use_Dg
        self.use_Lf = use_Lf
        if self.use_Dg and self.use_Lf:
            input_dim = 3 + self.num_nearest_bone * 8
        elif self.use_Dg and not self.use_Lf:
            input_dim = 3 + self.num_nearest_bone * 7
        elif self.use_Lf and not self.use_Dg:
            input_dim = 3 + self.num_nearest_bone * 7
        else:
            input_dim = 3 + self.num_nearest_bone * 6
        self.mlp1 = MLP([input_dim, 128, 64])
        self.gcu1 = GCU(in_channels=64, out_channels=256, aggr=aggr)
        self.gcu2 = GCU(in_channels=256, out_channels=512, aggr=aggr)
        self.gcu3 = GCU(in_channels=512, out_channels=512, aggr=aggr)
        self.mlp2 = MLP([256, 256, 1024])

        self.cls_branch = Seq(Linear(1024 + 512, 1024), ReLU(), BN(1024), Linear(1024, 512), ReLU(), BN(512),
                              Linear(512, 256), ReLU(), BN(256),
                              Linear(256, self.num_nearest_bone))

    def forward(self, data):
        samples = data.skin_input
        samples = data.skin_input
        if self.use_Dg and self.use_Lf:
            samples = samples[:, 0: 8 * self.num_nearest_bone]
        elif self.use_Dg and not self.use_Lf:
            samples = samples[:, np.arange(samples.shape[1]) % 8 != 7]
            samples = samples[:, 0: 7 * self.num_nearest_bone]
        elif self.use_Lf and not self.use_Dg:
            samples = samples[:, np.arange(samples.shape[1]) % 8 != 6]
            samples = samples[:, 0: 7 * self.num_nearest_bone]
        else:
            samples = samples[:, np.arange(samples.shape[1]) % 8 != 7]
            samples = samples[:, np.arange(samples.shape[1]) % 7 != 6]
            samples = samples[:, 0: 6 * self.num_nearest_bone]

        raw_input = torch.cat([data.pos, samples], dim=1)

        input0 = self.mlp1(raw_input)
        input1 = self.gcu1(input0, data.tpl_edge_index, data.geo_edge_index)

        global_input = self.mlp2(input1)
        global_input, _ = scatter_max(global_input, data.batch, dim=0)
        global_input = torch.repeat_interleave(global_input, torch.bincount(data.batch), dim=0)

        input2 = self.gcu2(input1, data.tpl_edge_index, data.geo_edge_index)
        input3 = self.gcu3(input2, data.tpl_edge_index, data.geo_edge_index)
        input4 = torch.cat([input3, global_input], dim=1)

        output = self.cls_branch(input4)
        return output


def run_skinnet(nearest_bone):
    model = skinnet(nearest_bone=nearest_bone)
    return model
