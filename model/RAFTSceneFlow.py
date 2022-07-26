import torch
import torch.nn as nn

from model.extractor import FlotEncoder
from model.corr import CorrBlock
from model.update import UpdateBlock
from model.refine import FlotRefine

print("begin!")

class RSF(nn.Module):
    def __init__(self, args):
        super(RSF, self).__init__()
        
        self.hidden_dim = 64
        self.context_dim = 64
        self.feature_extractor = FlotEncoder()
        self.context_extractor = FlotEncoder()
        
        self.corr_block = CorrBlock(num_levels=args.corr_levels, base_scale=args.base_scales,
                                    resolution=3, truncate_k=args.truncate_k)
        
        self.update_block = UpdateBlock(hidden_dim=self.hidden_dim)
        # self.refine_block = FlotRefine()
        # print("init end!")
    def forward(self, p, num_iters=12):
        # feature extraction
        
        [xyz1, xyz2] = p
        
        fmap1, graph = self.feature_extractor(p[0])
        
        fmap2, _ = self.feature_extractor(p[1])

        
        self.corr_block.init_module(fmap1, fmap2, xyz2)

       
        fct1, graph_context = self.context_extractor(p[0])

        # （1）net：hidden features（2）inp：context features
        net, inp = torch.split(fct1, [self.hidden_dim, self.context_dim], dim=1)
        net = torch.tanh(net)
        inp = torch.relu(inp)

        # initial
        coords1, coords2 = xyz1, xyz1
        flow_predictions = []

        # GRU
        for itr in range(num_iters):
            coords2 = coords2.detach()
            # 
            corr = self.corr_block(coords=coords2)
            # flow
            flow = coords2 - coords1
            # corr
            net, delta_flow = self.update_block(net, inp, corr, flow, graph_context, xyz1)
            
            coords2 = coords2 + delta_flow
            flow_predictions.append(coords2 - coords1)
        # refined_flow = self.refine_block(coords2 - coords1, graph)
        # flow_predictions.append(refined_flow)

        return flow_predictions

