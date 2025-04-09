import torch
import torch.nn as nn

class GCNBackbone(nn.Module):
    """
    Graph Convolutional Network based on https://arxiv.org/abs/1609.02907

    """

    def __init__(self, feat_dim, hidden_dim1, hidden_dim2, n_layers, dropout, activation, is_sparse=False):
        """Dense version of GAT."""
        super(GCNBackbone, self).__init__()
        
        self.n_layers = n_layers
        
        W_list = []
         
        if n_layers >= 1:
            W_list.append(nn.Parameter(torch.FloatTensor(feat_dim, hidden_dim1)))

        if n_layers >= 2:
            W_list.append(nn.Parameter(torch.FloatTensor(hidden_dim1, hidden_dim2)))
            
        for _ in range(n_layers - 2):
            W_list.append(nn.Parameter(torch.FloatTensor(hidden_dim2, hidden_dim2)))
            
        self.W_list = nn.ParameterList(W_list)

        self.act = activation()
        self.dropout = nn.Dropout(p=dropout)

        for module in self.W_list:
            nn.init.xavier_uniform_(module)

        self.is_sparse = is_sparse

        self.first_iteration = 1
        
    def forward(self, x, adj, output_z = False):
        adj = adj.to(torch.float)
        z = torch.tensor([]).cuda()
        # Layer 1
        if torch.cuda.is_available():      
            x=x.cuda()
            adj=adj.cuda()
            self.W_list = self.W_list.cuda()
            
        for i in range(self.n_layers - 1):
            
            # breakpoint()
            
            support = torch.mm(x, self.W_list[i]).cuda()
            
            if i == 0:
                z = support

            embeddings = (
                torch.sparse.mm(adj, support) if self.is_sparse else torch.mm(adj, support)
            ).cuda()
            
            embeddings = self.dropout(embeddings)

            x = self.act(embeddings)
        
        # Layer 2
        support = torch.mm(x, self.W_list[-1])
        embeddings = (
            torch.sparse.mm(adj, support) if self.is_sparse else torch.mm(adj, support)
        )
        
        # print(f'Graph has {adj.size(dim=1)} vertices, size after second layer is {embeddings.size()} and should be {adj.size(dim=0)}x{5}')        

        return {'emb': embeddings, 'z': z}  
    
    def get_features(self, x, adj):
        
        adj = adj.to(torch.float)
        
        # Layer 1
        if torch.cuda.is_available():      
            x=x.cuda()
            adj=adj.cuda()
            self.W_list = self.W_list.cuda()
            
        feature_list = []
            
        for i in range(self.n_layers - 1):
            
            support = torch.mm(x, self.W_list[i]).cuda()

            embeddings = (
                torch.sparse.mm(adj, support) if self.is_sparse else torch.mm(adj, support)
            ).cuda()
            
            embeddings = self.dropout(embeddings)

            x = self.act(embeddings)
            
            feature_list.append(x.clone())
        
        # Layer 2
        support = torch.mm(x, self.W_list[-1])
        embeddings = (
            torch.sparse.mm(adj, support) if self.is_sparse else torch.mm(adj, support)
        )
        
        feature_list.append(embeddings.clone())
        
        return feature_list   

class Readout(nn.Module):
    """
    This module learns a single graph level representation for a molecule given GNN generated node embeddings
    """

    def __init__(self, attr_dim, embedding_dim, hidden_dim, output_dim, n_layers, num_cats, activation, graph_class = True):
        super(Readout, self).__init__()
        self.attr_dim = attr_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_cats = num_cats
        self.n_layers = n_layers
        self.graph_class = graph_class

        self.layers = []
        self.layers.append(nn.Linear(attr_dim + embedding_dim, hidden_dim))
        self.layers.append(activation())
        
        for _ in range(n_layers - 2):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.layers.append(activation())

        self.layers.append(nn.Linear(hidden_dim, output_dim))
        self.layers.append(activation())
        
        self.layers = nn.Sequential(*self.layers)
        
        self.output = nn.Linear(output_dim, num_cats)

        self.first_iteration = 1

    def forward(self, node_features, node_embeddings):
        
        node_features = node_features.cuda()
        node_embeddings = node_embeddings.cuda()
        
        combined_rep = torch.cat(
            (node_features, node_embeddings), dim=1
        )  # Concat initial node attributed with embeddings from sage
        
        graph_rep = self.layers(combined_rep)

        logits = self.output(graph_rep)
        if self.graph_class:
            return torch.mean(logits, dim=0)  # Generated logits for multilabel classification

        return logits

class GCN(nn.Module):
    """
    Network that consolidates GCN + Readout into a single nn.Module
    """

    def __init__(
        self,
        feat_dim,
        hidden_dim,
        node_embedding_dim,
        dropout,
        readout_hidden_dim,
        graph_embedding_dim,
        num_categories,
        n_layers_gcn,
        n_layers_readout,
        activation,
        graph_class=True,
        sparse_adj=False,
    ):
        super(GCN, self).__init__()
        self.gcn = GCNBackbone(
            feat_dim, hidden_dim, node_embedding_dim, n_layers_gcn, dropout, activation,  is_sparse=sparse_adj
        )
        self.readout = Readout(
            feat_dim,
            node_embedding_dim,
            readout_hidden_dim,
            graph_embedding_dim,
            n_layers_readout,
            num_categories,
            activation,
            graph_class
        )

    def forward(self, adj_matrix, feature_matrix):
        self.node_embeddings = self.gcn(feature_matrix, adj_matrix)['emb']
        logits = self.readout(feature_matrix, self.node_embeddings)
        return logits
    
    def get_features(self, adj_matrix, feature_matrix):
        return self.gcn.get_features(feature_matrix, adj_matrix)