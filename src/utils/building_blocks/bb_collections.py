from collections.abc import Iterable
from utils.building_blocks.building_block import NewBuildingBlock   
from utils.chemical import mol_to_adj, mol_to_features
from typing import List, Dict, Tuple
from rdkit import Chem
import torch


class NewBuildingBlocks(list):
    def __init__(self, blocks: List[NewBuildingBlock]) -> None:
        super().__init__(blocks)
    
    @classmethod
    def from_molecule(args, cls, mol: Chem.rdchem.Mol, degree: int, no_dup=True, feature_onehot_encoding=None) -> 'NewBuildingBlocks':
        
        A = mol_to_adj(mol)
        if feature_onehot_encoding is None:
            X = mol_to_features(mol)
        else:
            X = mol_to_features(mol, feature_onehot_encoding = feature_onehot_encoding)
        
        blocks = []
        
        for k in range(len(X)):
            # get all idxs that are connected to middle_idx with degree hops
            idxs = set([k])
            new_idxs = set([k])

            for d in range(degree):
                neighbor_idxs = [neighbor.GetIdx() for idx in new_idxs for neighbor in mol.GetAtomWithIdx(idx).GetNeighbors()]
                new_idxs = set(neighbor_idxs) - idxs
                idxs = idxs.union(new_idxs)
                
            # sort idx and get middle_idx
            idxs = sorted(idxs)
            new_idxs = sorted(new_idxs)
            middle_idx = idxs.index(k)

            sub_A = A[idxs,:][:,idxs]
            sub_X = X[idxs,:]

            for p in range(len(sub_A)):
                for q in range(p):
                    if sub_A[middle_idx][p] == 0 and sub_A[middle_idx][q] == 0:
                        sub_A[p][q] = 0
                        sub_A[q][p] = 0

            if len(sub_A)>1:                
                for i in range(len(sub_A)):
                    assert(sum(sub_A[i])>=2)

            connections = []
            for sub_idx, idx in enumerate(idxs): 
                if idx in new_idxs:
                    connections.append(bool(A[idx].sum() > sub_A[sub_idx].sum()))
                else:
                    connections.append(False)

            connection_idxs = [idx for idx, val in enumerate(connections) if val]
            
            blocks.append(NewBuildingBlock(args, sub_A, sub_X, middle_idx, connection_idxs, degree))

        if no_dup:
            blocks = list(set(blocks))
            
            #Note: for some reason at some point I had commented the below. Note if it ever causes a problem. 
            for block in blocks:
                assert(block.middle_idx==0)
                
        blocks_dedup = []
        
        for b1 in blocks:
            if b1 not in blocks_dedup:
                blocks_dedup.append(b1)
        
        blocks = blocks_dedup
        
        return cls(blocks)

    
    def __iadd__(self, other: 'NewBuildingBlocks') -> 'NewBuildingBlocks':
        for block in other:
            if block not in self:
                self.append(block)
        return self

    def __str__(self) -> str:
        return '\n'.join(str(block) for block in self)

    def to_y_bb_dictionary(self) -> Dict[Tuple, List[NewBuildingBlock]]:
        y_bb_dict = {}

        for block in self:
            if block.y in y_bb_dict:
                y_bb_dict[block.y].append(block)
            else:
                y_bb_dict[block.y] = [block]
        
        return y_bb_dict


class BuildingBlockDict():
    def __init__(self, y_bb_dict: Dict[Tuple, List[NewBuildingBlock]]) -> None:
        self.y_bb_dict = y_bb_dict
    
    @classmethod
    def from_new_building_blocks(cls, building_blocks: NewBuildingBlocks) -> 'BuildingBlockDict':
        y_bb_dict = building_blocks.to_y_bb_dictionary()
        return cls(y_bb_dict)
    
    # gets Y and returns multiplicities
    def get_multiplicities(self, Y: torch.Tensor) -> Dict[Tuple, int]:
        return_dict: Dict[Tuple[float], int] = {}
            
        for y_ in Y:
            y = tuple(y_.numpy())
            if y in return_dict:
                return_dict[y] += 1
            else:
                return_dict[y] = 1

        return return_dict

    def __getitem__(self, y: Tuple) -> List[NewBuildingBlock]:
        return self.y_bb_dict[y]
    
    def __str__(self) -> str:
        return '\n'.join(f'y: {y[0]}: bb_list: {len(bb_list)}' for y, bb_list in self.y_bb_dict.items())
    
    def __repr__(self) -> str:
        return self.__str__()
