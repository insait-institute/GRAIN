from rdkit.Chem.rdmolops import GetAdjacencyMatrix
from rdkit import Chem
import torch
import pandas as pd
from utils.misc import possible_feature_values
from rdkit.Chem.Draw import MolToImage


def draw_atom(X, A):
    mol = Chem.RWMol()
    for k, atom in enumerate(X):
        mol.AddAtom(features_to_mol(atom))

    for k, row in enumerate(A):
        for l, conn in enumerate(row):
            if conn == 1 and k < l:
                mol.AddBond(k, l, Chem.BondType.UNSPECIFIED)
                
    drawing = MolToImage(mol, size=(1200, 1200))

    return drawing

def get_atom_symbol(atomic_num):
    pt = Chem.GetPeriodicTable()
    return pt.GetElementSymbol(atomic_num)

def features_to_mol(X):
    atom = Chem.Atom(get_atom_symbol(int(X[0])))
    atom.SetFormalCharge(int(X[1]))
    atom.SetNoImplicit(True)
    atom.SetNumExplicitHs(int(X[4]))
    atom.SetIsAromatic(bool(X[5]))
    
    hybridization = int(X[6])  # Atom hybridization
    if hybridization==0:
        hybridization = Chem.HybridizationType.UNSPECIFIED
    elif hybridization==1:
        hybridization = Chem.HybridizationType.S
    elif hybridization==2:
        hybridization = Chem.HybridizationType.SP
    elif hybridization==3:
        hybridization = Chem.HybridizationType.SP2
    elif hybridization==4:
        hybridization = Chem.HybridizationType.SP3
    elif hybridization==5:
        hybridization = Chem.HybridizationType.SP3D
    elif hybridization==6:
        hybridization = Chem.HybridizationType.SP3D2
    atom.SetHybridization(hybridization)

    return atom

def mol_to_adj(mol: Chem.rdchem.Mol) -> torch.Tensor:
    adj = GetAdjacencyMatrix(mol)
    adj_mat = torch.tensor(adj, dtype=torch.float32)
    adj_mat += torch.eye(adj.shape[0]) # Adding the self-loops to the adjacency matrix
    return adj_mat

def mol_to_features(args, mol: Chem.rdchem.Mol, feature_onehot_encoding=True) -> torch.Tensor:
    
    ft_mat = torch.rand(mol.GetNumAtoms(),8)

    for k, atom in enumerate(mol.GetAtoms()):
        # Chirality
        chiral_tag = atom.GetChiralTag()  # Chirality
        if chiral_tag == Chem.ChiralType.CHI_UNSPECIFIED:
            chiral_tag = 0

        # Hybridization            
        hybridization = atom.GetHybridization()  # Atom hybridization
        if hybridization==Chem.HybridizationType.S:
            hybridization = 1
        elif hybridization==Chem.HybridizationType.SP:
            hybridization = 2
        elif hybridization==Chem.HybridizationType.SP2:
            hybridization = 3
        elif hybridization==Chem.HybridizationType.SP3:
            hybridization = 4
        elif hybridization==Chem.HybridizationType.SP3D:
            hybridization = 5
        elif hybridization==Chem.HybridizationType.SP3D2:
            hybridization = 6

        ft_mat[k,:] = torch.tensor([
            atom.GetAtomicNum(),  # Atomic number
            atom.GetFormalCharge(), # Formal charge
            atom.GetDegree(), # Number of bonds (not counting Hs) 
            chiral_tag, 
            atom.GetTotalNumHs(),  # Number of explicit Hs
            atom.GetMass()/100,  # Atomic mass
            int(atom.GetIsAromatic()), 
            hybridization
            # atom.GetHybridization()  # Atom hybridization
        ], dtype=torch.float32)

    if not feature_onehot_encoding:
        return ft_mat
    
    lists = possible_feature_values(args)
    
    one_hots = []
    
    for i in range(len(ft_mat)):
        x = ft_mat[i]
        one_hot = torch.zeros(sum(len(l) for l in lists))
        so_far = 0
        for j in range(8):
            assert(x[j] in lists[j])
            one = so_far + lists[j].index(x[j])
            one_hot[one] = 1
            so_far+=len(lists[j])
        assert(one_hot.size(0)==140)
        one_hots.append(one_hot)
    
    ft_mat_encoded = torch.stack(one_hots)
    
    if feature_onehot_encoding==False:
        return ft_mat
    if feature_onehot_encoding==True:
        return ft_mat_encoded

def mol_df(mol: Chem.rdchem.Mol) -> pd.DataFrame:
    df = pd.DataFrame(columns=['Atom', 'Charge', '#Bonds', 'Chirality', '#Hs', 'mass', 'arom', 'hyb'])
    for k, atom in enumerate(mol.GetAtoms()):
        df.loc[k] = [
            atom.GetSymbol(), # Symbol
            atom.GetFormalCharge(), # Formal charge
            atom.GetDegree(), # Number of bonds (not counting Hs) 
            str(atom.GetChiralTag()), 
            atom.GetTotalNumHs(),  # Number of explicit Hs
            atom.GetMass(),  # Atomic mass
            atom.GetIsAromatic(), 
            str(atom.GetHybridization())
        ]
    return df

def mol_to_out(mol) -> torch.Tensor:
    adj_mat = mol_to_adj(mol)
    ft_mat = mol_to_features(mol)    
    Y_mol = adj_mat@ft_mat
    return Y_mol  