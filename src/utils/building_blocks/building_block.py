import torch
from rdkit import Chem
from typing import Generic, TypeVar, List, Union, Tuple, Dict, Set 
from utils.reconstruction import tensor2d_to_tuple, get_partial_deg2_bb, properly_structure_deg2_grouped
from utils.chemical import get_atom_symbol
import networkx as nx
import numpy as np
import yaml
import json

pt = Chem.GetPeriodicTable()

class NewBuildingBlock:

    W = None
    activation = torch.relu
    comparison_mode = 'full' # choose from 'full' and 'symbol' and 'struct'
    
    def __init__(self, 
                 args,
                 A: torch.Tensor, 
                 X: torch.Tensor, 
                 middle_idx: int, 
                 connections: List[int],
                 degree: int,
                 feat_dim: int,
                 hidden_size: int,
                 max_deg: int
                 ) -> None:
        
        if NewBuildingBlock.W is None:
            NewBuildingBlock.W = torch.rand(feat_dim, hidden_size).cuda()

        self.args = args
        self.X_ordered = None
        self.feat_dim = feat_dim
        self.hidden_size = hidden_size
        self.max_deg = max_deg
        
        # sort the rows of X
        
        self.degree = degree
        # Calculate the key y: 
        if self.degree == 1:
            y = A[middle_idx] @ X
        elif self.degree == 2:
            y = A[middle_idx] @ NewBuildingBlock.activation(A @ X @ NewBuildingBlock.W)
        else:
            print(f'Problem is {self.degree}')
            raise NotImplementedError

        # put the middle_idx to the first position
        permutation = sorted(range(len(X)), key=lambda k: tuple(X[k,:]))
        permutation = [middle_idx] + [idx for idx in permutation if idx != middle_idx]

        # apply permutations to input
        self.X = X[permutation,:]
        self.A = A[permutation,:][:,permutation]
        self.middle_idx = permutation.index(middle_idx)
        
        self.connections = tuple(sorted([permutation.index(idx) for idx in connections]))

        perm2 = [0] + [c for c in self.connections] + [idx for idx in range(1, len(self.X)) if idx not in self.connections]

        self.X = self.X[perm2,:]
        self.A = self.A[perm2,:][:,perm2]
        self.middle_idx = perm2.index(self.middle_idx)        
        self.connections = tuple(sorted([perm2.index(idx) for idx in self.connections]))

        assert(self.middle_idx == 0)
        assert(self.connections == tuple(range(1, len(self.connections)+1)))


        # Asserts
        # middle_idx stays invariant
        assert(self.middle_idx == 0)
        # y stays invariant
        if self.degree == 2:
            y_after_permutation = A[middle_idx] @ NewBuildingBlock.activation(A @ X @ NewBuildingBlock.W)

        elif self.degree == 1:
            y_after_permutation = A[middle_idx] @ X
        else:
            raise NotImplementedError
        assert(torch.allclose(y, y_after_permutation))
        

        # Save as tuple to make it hashable
        self.X = tensor2d_to_tuple(self.X)
        self.A = tensor2d_to_tuple(self.A)
        self.y = tuple(y.tolist())

    def to_dict(self):
        return {"A": self.A, "X": self.X, "middle_idx": self.middle_idx, "connections": self.connections, "degree": self.degree}

    @staticmethod
    def from_dict(args, data):
        return NewBuildingBlock(args, torch.tensor(data["A"]), torch.tensor(data["X"]), int(data["middle_idx"]), list(data["connections"]), data["degree"])

    @staticmethod
    
    def from_tensor_dict(args, tensor_dict: dict) -> 'NewBuildingBlock':
        
        A = torch.tensor(tensor_dict['A'], dtype=torch.float32)
        A = A.view(-1, A.size(-1))
        X = torch.tensor(tensor_dict['X'], dtype=torch.float32)
        X = X.view(-1, X.size(-1))
        middle_idx = int(tensor_dict['middle_idx'].item())  # Extract the int value from the tensor
        connections = [int(x) for x in tensor_dict['connections'].flatten().tolist()]  # Convert connections tensor to list of ints
        
        degree = tensor_dict['degree'].item()  # Extract the int value from the tensor
                
        return NewBuildingBlock(args, A, X, middle_idx, connections, degree)

    def to_tensor(self):
        # Convert the NewBuildingBlock object to a dictionary of tensors
        return {
            'A': torch.tensor(self.A, dtype=torch.float32),
            'X': torch.tensor(self.X, dtype=torch.float32),
            'middle_idx': torch.tensor(self.middle_idx, dtype=torch.float32),
            'connections': torch.tensor(self.connections, dtype=torch.float32),
            'degree': torch.tensor(self.degree, dtype=torch.float32)
        }
    
    def to_json(self):
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(args, cls, json_str):
        data = json.loads(json_str)
        return cls.from_dict(args, data)
    
    def center_neighbors(self) -> List[int]:
        return self.neighbors(self.middle_idx)
    
    def neighbors(self, idx: int) -> List[int]:
        neighbor_indices = [idx1 for idx1, val in enumerate(self.A[idx]) if val and idx1!=idx]
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
        mol.GetAtomWithIdx(self.middle_idx).SetProp('atomNote', 'middle')

        return mol
    
    def draw(self) -> None:
        # Draw the molecule, but the connections need to be dashed
        drawing = Chem.Draw.MolToImage(self.submol(), highlightAtoms=self.connections)

        return drawing

    def __hash__(self) -> int:
        if NewBuildingBlock.comparison_mode == 'full':
            return hash((self.X, self.A, self.middle_idx, self.connections, self.degree))
        elif NewBuildingBlock.comparison_mode == 'symbol':
            X_ = tuple(torch.tensor(self.X)[:,0].numpy())
            return hash((X_, self.A, self.middle_idx, self.connections, self.degree))
        elif NewBuildingBlock.comparison_mode == 'struct':
            X_ = tuple(map(tuple, torch.tensor(self.X)[:, [0, 2]].numpy()))

            return hash((X_, self.A, self.middle_idx, self.connections, self.degree))
        else:
            raise NotImplementedError


    def __eq__(self, other: 'NewBuildingBlock') -> bool:
        
        if NewBuildingBlock.comparison_mode == 'full':
            
            if len(self.A)!=len(other.A):
                return False
            
            if self.degree != other.degree:
                return False
            
            if torch.any( torch.abs(torch.tensor(self.X[self.middle_idx]) - torch.tensor(other.X[other.middle_idx])) > 0.01):
                return False
            
            A1 = torch.tensor(self.A)
            A2 = torch.tensor(other.A)
            
            X1 = torch.tensor(self.X)
            X2 = torch.tensor(other.X)
            
            sorted_X1 = torch.stack(sorted(X1, key=lambda row: tuple(row.tolist())))
            sorted_X2 = torch.stack(sorted(X2, key=lambda row: tuple(row.tolist())))
            
            if torch.any( torch.abs(sorted_X1 - sorted_X2) > 0.01):
                return False
        
            if self.degree==1 and other.degree==1:
                return True
            
            y1 = A1 @ A1 @ X1
            y2 = A2 @ A2 @ X2
            
            if torch.any( torch.abs(y1[0] - y2[0]) > 0.01):
                return False
            
            if (torch.all(torch.eq(X1,X2)) and 
                    self.A == other.A and 
                    self.middle_idx == other.middle_idx and 
                    self.connections == other.connections and 
                    self.degree == other.degree):
                return True
                        
            G1 = nx.from_numpy_matrix(np.array(self.A))
            G2 = nx.from_numpy_matrix(np.array(other.A))

            for i in range(len(self.X)):
                G1.nodes[i]['attr'] = X1[i]
                G2.nodes[i]['attr'] = X2[i]

            # Define a node_match function to compare node attributes
            def node_match(n1, n2):
                return np.array_equal(n1['attr'], n2['attr'])

            # Check for graph isomorphism with node attributes
            isom = nx.is_isomorphic(G1, G2, node_match=node_match)
            return isom
        
        elif NewBuildingBlock.comparison_mode == 'symbol':
            
            if len(self.A)!=len(other.A):
                return False
            
            if self.degree != other.degree:
                return False
            
            if torch.any( torch.abs(torch.tensor(self.X[self.middle_idx]) - torch.tensor(other.X[other.middle_idx])) > 0.01):
                return False
            
            A1 = torch.tensor(self.A)
            A2 = torch.tensor(other.A)
            
            X1 = torch.tensor(self.X)
            X2 = torch.tensor(other.X)
            
            X1 = X1[:, [0]]
            X2 = X2[:, [0]]
            
            if torch.any( torch.abs(X1 - X2) > 0.01):
                return False
            
            y1 = A1 @ A1 @ X1
            y2 = A2 @ A2 @ X2
            
            if torch.any( torch.abs(y1[0] - y2[0]) > 0.01):
                return False
            
            if (torch.all(torch.eq(X1,X2)) and 
                    self.A == other.A and 
                    self.middle_idx == other.middle_idx and 
                    self.connections == other.connections and 
                    self.degree == other.degree):
                return True
                        
            G1 = nx.from_numpy_matrix(np.array(self.A))
            G2 = nx.from_numpy_matrix(np.array(other.A))

            for i in range(len(self.X)):
                G1.nodes[i]['attr'] = X1[i]
                G2.nodes[i]['attr'] = X2[i]

            # Define a node_match function to compare node attributes
            def node_match(n1, n2):
                return np.array_equal(n1['attr'], n2['attr'])

            # Check for graph isomorphism with node attributes
            isom = nx.is_isomorphic(G1, G2, node_match=node_match)
            return isom
        
        elif NewBuildingBlock.comparison_mode == 'struct':
            
            if len(self.A)!=len(other.A):
                return False
            
            if self.degree != other.degree:
                return False
            
            if torch.any( torch.abs( torch.tensor(self.X[self.middle_idx]) - torch.tensor(other.X[other.middle_idx])) > 0.01):
                return False
            
            A1 = torch.tensor(self.A)
            A2 = torch.tensor(other.A)
            
            X1 = torch.tensor(self.X)
            X2 = torch.tensor(other.X)
            
            X1 = X1[:, [0, 2]]
            X2 = X2[:, [0, 2]]
            
            sorted_X1 = torch.stack(sorted(X1, key=lambda row: tuple(row.tolist())))
            sorted_X2 = torch.stack(sorted(X2, key=lambda row: tuple(row.tolist())))
            
            if torch.any( torch.abs(sorted_X1 - sorted_X2) > 0.01):
                return False
            
            y1 = A1 @ A1 @ X1
            y2 = A2 @ A2 @ X2
            
            if torch.any( torch.abs(y1[0] - y2[0]) > 0.01):
                return False
            
            if (torch.all(torch.eq(X1,X2)) and 
                    self.A == other.A and 
                    self.middle_idx == other.middle_idx and 
                    self.connections == other.connections and 
                    self.degree == other.degree):
                return True
                        
            G1 = nx.from_numpy_matrix(np.array(self.A))
            G2 = nx.from_numpy_matrix(np.array(other.A))

            for i in range(len(self.X)):
                G1.nodes[i]['attr'] = X1[i]
                G2.nodes[i]['attr'] = X2[i]

            # Define a node_match function to compare node attributes
            def node_match(n1, n2):
                return np.array_equal(n1['attr'], n2['attr'])

            # Check for graph isomorphism with node attributes
            isom = nx.is_isomorphic(G1, G2, node_match=node_match)
            return isom
        else:
            raise NotImplementedError
        
    def ordered_X(self) -> None:
        if self.X_ordered is None:
            X_structured, former_idxs = get_partial_deg2_bb(self, 0, self.max_deg)
            X_with_nans = torch.cat((self.X[0].unsqueeze(0), X_structured), dim=0)
            self.X_ordered, self.ordered_idxs = properly_structure_deg2_grouped(self.args, X_with_nans.unsqueeze(0), former_idxs.unsqueeze(0), self.max_deg, self.feat_dim)
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

    def __str__(self) -> str:
        ret_str = f'{get_atom_symbol(int(self.X[self.middle_idx][0]))} \n'
        ret_str += f'    middle: {self.middle_idx} -- connections: {self.connections} -- degree: {self.degree} \n'
        ret_str += f'    {torch.tensor(self.X)} \n'
        ret_str += f'    {torch.tensor(self.A)} \n'
        ret_str += f'    {self.y} \n'
        return ret_str
