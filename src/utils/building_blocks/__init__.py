from utils.building_blocks.bb_collections import NewBuildingBlocks, BuildingBlockDict
from utils.building_blocks.extended_bb_collections import ExtendedBuildingBlocks, ExtendedBuildingBlockDict
from utils.building_blocks.building_block import NewBuildingBlock
from utils.building_blocks.extended_bb import ExtendedBuildingBlock
from utils.misc import get_degree
import torch
def convert_to_bb(args, bb_deg2s, max_degree, feat_dim, hidden_size):
    
    new7 = max_degree+1
    new5 = max_degree-1

    list_of_bbs = []
    
    for index in range(bb_deg2s.shape[0]):
        bb = bb_deg2s[index]
        non_zero_rows = torch.any(bb != 0, dim=1)
        num_nodes = torch.sum(non_zero_rows).item()
        A = torch.zeros((num_nodes, num_nodes)).cuda()
        X = bb[non_zero_rows].cuda()
        degrees = get_degree(args, X.unsqueeze(0))
        degrees = degrees.squeeze(0)
        middle_idx = 0
        num_neighbors = 0
        for i in range(1,new7):
            if torch.any(bb[i] != 0):
                A[0][i] = 1
                A[i][0] = 1
                num_neighbors+=1
        connections = torch.where(degrees[num_neighbors+1:]>1)[0]
        connections = connections + (num_neighbors + 1)
        curr = num_neighbors+1
        for i in range(1,new7):
            for j in range(new5*i+2, new5*i+new7):
                if torch.any(bb[j] != 0):
                    A[i][curr] = 1
                    A[curr][i] = 1
                    curr+=1
        for i in range(num_nodes):
            A[i][i] = 1
        
        list_of_bbs.append(NewBuildingBlock(args, A, X, middle_idx, connections, 2, feat_dim, hidden_size, max_degree))
        
    return list_of_bbs