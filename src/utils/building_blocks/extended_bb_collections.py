from collections.abc import Iterable
from utils.building_blocks.extended_bb import ExtendedBuildingBlock

from typing import List, Dict, Tuple
import torch

class ExtendedBuildingBlocks(list):
    def __init__(self, blocks: List[ExtendedBuildingBlock]) -> None:
        super().__init__(blocks)
    
    @classmethod
    
    def __iadd__(self, other: 'ExtendedBuildingBlocks') -> 'ExtendedBuildingBlocks':
        for block in other:
            if block not in self:
                self.append(block)
        return self

    def __str__(self) -> str:
        return '\n'.join(str(block) for block in self)

    def to_y_bb_dictionary(self) -> Dict[Tuple, List[ExtendedBuildingBlock]]:
        y_bb_dict = {}
        for block in self:
            if tuple(block.y) in y_bb_dict:
                y_bb_dict[block.y].append(block)
            else:
                y_bb_dict[block.y] = [block]
        
        return y_bb_dict


class ExtendedBuildingBlockDict():
    def __init__(self, y_bb_dict: Dict[Tuple, List[ExtendedBuildingBlock]]) -> None:
        self.y_bb_dict = y_bb_dict
    
    @classmethod
    def from_new_building_blocks(cls, building_blocks: ExtendedBuildingBlocks) -> 'ExtendedBuildingBlockDict':
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

    def __getitem__(self, y: Tuple) -> List[ExtendedBuildingBlock]:
        return self.y_bb_dict[y]
    
    def __str__(self) -> str:
        return '\n'.join(f'y: {y[0]}: bb_list: {len(bb_list)}' for y, bb_list in self.y_bb_dict.items())
    
    def __repr__(self) -> str:
        return self.__str__()
