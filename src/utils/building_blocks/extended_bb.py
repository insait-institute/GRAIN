import torch
from rdkit import Chem
from typing import List, Tuple, Set 

from utils import tensor2d_to_tuple, properly_structure_deg2_grouped, get_partial_deg2_bb
    
pt = Chem.GetPeriodicTable()


def get_atom_symbol(atomic_num):
    return pt.GetElementSymbol(atomic_num)
        

class ExtendedBuildingBlock:
    W1 = torch.rand(8, 8)
    activation = torch.relu
    comparison_mode = 'symbol' # choose from 'full' and 'symbol'

    def __init__(self, 
                 args,
                 A: torch.Tensor, 
                 X: torch.Tensor, 
                 connections: List[int],
                 ) -> None:
        # sort the rows of X
        # Calculate the key y: 
        self.args = args
        # put the middle_idx to the first position
        permutation = sorted(range(len(X)), key=lambda k: tuple(X[k,:]))
        permutation = [idx for idx in permutation]
        
        # apply permutations to input
        self.X = X[permutation,:]
        self.A = A[permutation,:][:,permutation]
        # Question: does it matter how the middle_idxs are sorted here? I hope not.
        self.connections = tuple(sorted([permutation.index(idx) for idx in connections]))
        
        # Save as tuple to make it hashable
        self.X = tensor2d_to_tuple(self.X)
        self.A = tensor2d_to_tuple(self.A)
    
    def ext_neighbors(self, idx: List[int]) -> List[int]:
        neighbor_indices = []
        for i in idx:
            for idx1 in range(len(self.A[i])):
                if self.A[i][idx1]==1 and idx1!=i:
                    neighbor_indices.append(idx1)
        return neighbor_indices
    
    def connections_bool(self) -> List[int]:
        bool_list = [1 if i in self.connections else 0 for i in range(len(self.A))]
        return bool_list
    
    def set_x(self, idxs: Set[int]) -> Set[Tuple[float]]:
        if isinstance(idxs, Set): #add that the elements are int (or at least one is)
            return {self.X[idx] for idx in idxs}
        else:
            return TypeError # Type Not correct

    def submol(self) -> Chem.rdchem.Mol:
        # Build molecule from A and X. Mark the middle atom and the connections
        mol = Chem.RWMol()
        for k, atom in enumerate(self.X):
            mol.AddAtom(Chem.Atom(get_atom_symbol(int(atom[0]))))

        for k, row in enumerate(self.A):
            for l, conn in enumerate(row):
                if conn == 1 and k < l:
                    mol.AddBond(k, l, Chem.BondType.UNSPECIFIED)

        # Highlight the middle atom and the connections
        
        return mol
    
    def draw(self) -> None:
        # Draw the molecule, but the connections need to be dashed
        drawing = Chem.Draw.MolToImage(self.submol(), highlightAtoms=self.connections)

        return drawing
    
    def ordered_X(self, max_degree) -> None:
        if self.X_ordered is None:
            X_structured, former_idxs = get_partial_deg2_bb(self, 0, max_degree)
            X_with_nans = torch.cat((self.X[0].unsqueeze(0), X_structured), dim=0)
            self.X_ordered, self.ordered_idxs = properly_structure_deg2_grouped(self.args, X_with_nans.unsqueeze(0), former_idxs.unsqueeze(0))
            self.X_ordered = self.X_ordered.squeeze()
            self.ordered_idxs = self.ordered_idxs.squeeze()
        
        return self.X_ordered, self.ordered_idxs
        
    def to_tensor(self) -> None:
        if torch.cuda.is_available():
            self.X = torch.tensor(self.X).cuda()
            self.A = torch.tensor(self.A).cuda()
            
    def to_tuple(self) -> None:
        self.X = tensor2d_to_tuple(self.X.cpu())
        self.A = tensor2d_to_tuple(self.A.cpu())

    def __hash__(self) -> int:
        if ExtendedBuildingBlock.comparison_mode == 'full':
            return hash((self.X, self.A, self.connections))
        elif ExtendedBuildingBlock.comparison_mode == 'symbol':
            X_ = tuple(torch.tensor(self.X)[:,0].numpy())
            return hash((X_, self.A, self.connections))
        else:
            raise NotImplementedError


    def __eq__(self, other: 'ExtendedBuildingBlock') -> bool:
        if ExtendedBuildingBlock.comparison_mode == 'full':
            return (self.X == other.X and 
                    self.A == other.A and 
                    self.connections == other.connections)
        elif ExtendedBuildingBlock.comparison_mode == 'symbol':
            return (
                    self.connections == other.connections and
                    self.A == other.A and
                    tuple(torch.tensor(self.X)[:,0].numpy()) == tuple(torch.tensor(other.X)[:,0].numpy())
                    )
        else:
            raise NotImplementedError

    def __str__(self) -> str:
        ret_str = f'    {torch.tensor(self.X)} \n'
        ret_str += f'    {torch.tensor(self.A)} \n'
        return ret_str
