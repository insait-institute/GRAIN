from data_loading.graph_dataset import GraphDataset
from torch.utils.data import DataLoader
import torch
import random

class SubgraphDataset(GraphDataset):
    def __init__(self, args, features, edge_info, neighbors_list=None, classes=None):
        self.args = args

        self.use_degree = args.use_degree
        self.num_extra_features = args.num_extra_features
        self.max_extra_value = args.max_extra_value
        if classes is not None:
            self.num_classes = torch.max(classes)+1
        else:
            self.num_classes = 6

        num_nodes = len(features)
        self.num_nodes = num_nodes

        if self.num_extra_features!=0:
            extra_random = torch.randint(0, self.max_extra_value, (features.shape[0], 20), dtype=torch.float32) 
            extra_random = extra_random[:,-self.num_extra_features:]
            features = torch.cat((extra_random,features), dim=1)
            

        self.bigX = features
        if args.dataset == 'pokec':
            self.__adj_from_list_of_neighbors(edge_info)
            self.neighbors_list = neighbors_list
            
        else:
            self.__adj_from_edge_list(edge_info)
            self.powers = self.__compute_adj_powers(max_power=4)
            max_degree = int((torch.max(torch.sum(self.bigA,dim=0)) - 1).item())
            self.max_degree = max(max_degree, 99)
        self.bigY = classes        


        
        
        self.list_of_neighbors = neighbors_list
        self.X, self.A, self.y = self.__generate_dataset_graphs(args.n_inputs, args.max_nodes, args.min_nodes)

    def __adj_from_list_of_neighbors(self, edges_list):
        edge_tensor = torch.tensor(edges_list).T
        edge_tensor = edge_tensor.to(int)

        mask = ~torch.isnan(edge_tensor).any(dim=0)
        edge_tensor = edge_tensor[:, mask]

        self_loops = edge_tensor[0, :] == edge_tensor[1, :]
        edge_tensor = edge_tensor[:, ~self_loops]
        edge_tensor = edge_tensor.T

        flat_tensor = edge_tensor.flatten()
        _, degrees = torch.unique(flat_tensor.cuda(), return_counts=True)
        most_common_count = degrees.max()

        max_degree = most_common_count
        self.max_degree = min(max_degree, 99) 

        self.bigA = edge_tensor

    def __adj_from_edge_list(self, edges):
        edge_tensor = torch.tensor(edges).T
        edge_tensor = edge_tensor.to(int)

        mask = ~torch.isnan(edge_tensor).any(dim=0)
        edge_tensor = edge_tensor[:, mask]

        self_loops = edge_tensor[0, :] == edge_tensor[1, :]
        edge_tensor = edge_tensor[:, ~self_loops]

        self.bigA = torch.zeros((self.num_nodes, self.num_nodes), dtype=torch.float32)
        self.bigA[edge_tensor[1],edge_tensor[0]] = 1
        self.bigA += torch.eye(self.num_nodes, self.num_nodes)

        if not self.args.directed:
            self.bigA = (self.bigA + self.bigA.T) / 2
            self.bigA = torch.where(self.bigA > 0, torch.tensor(1), self.bigA)

    def __compute_adj_powers(self, max_power):
        list = [torch.eye(len(self.bigA), dtype=self.bigA.dtype)]
        for _ in range(1, max_power+1):
            last = list[len(list)-1]
            next = last@self.bigA
            list += [next]
        return torch.stack(list, dim=0)


    def __reachable_within_k(self, center_node, k):
        reachability_matrix = torch.zeros_like(self.bigA)
        for i in range(1, k + 1):
            reachability_matrix += self.powers[i]
            reachable_nodes = torch.where(reachability_matrix[center_node] > 0)[0]
        return reachable_nodes

    def __sample_neighborhood(self, allowed_nodes, k):
        center = random.choice(allowed_nodes)
        # center = 2617
        print(f'We chose {center}')
        all_nodes = self.__reachable_within_k(center, k)
        # breakpoint()
        all_nodes = list(all_nodes)
        all_nodes = [u.item() for u in all_nodes]
        return all_nodes

    def __sample_graph(self, k, picking, max_nodes_threshold=None, min_num_nodes_threshold=None):
        union = []
        
        while len(union)==0 or len(union)>max_nodes_threshold or len(union)<min_num_nodes_threshold:
            center = random.choice(range(self.num_nodes))
            # center = 2386
            initial_neghborhood_nodes = self.__reachable_within_k(center, k)
            union = list(initial_neghborhood_nodes)
            union = [u.item() for u in union]
        
        print(f"Initial neighborhood: {union}")
        
        for _ in range(1, picking):
            union_new = list(set(list(self.__sample_neighborhood(union, k))) | set(union))
            # breakpoint()
            if max_nodes_threshold is not None and len(union_new)<=max_nodes_threshold:
                union = union_new
                print(f'After picking: {len(union_new)}')

        print(f'Final number of nodes is {len(union)}')
        
        nn = len(union)
        
        subadjacency = self.bigA[torch.tensor(union)][:,torch.tensor(union)]
        degrees = torch.sum(subadjacency, dim=1)-1
        
        print(f"max deg is {torch.max(degrees)}")
        
        indices = degrees.reshape(nn, 1)
        indices = indices.to(torch.int)
        if self.args.use_degree:
            one_hot_tensor = torch.zeros((nn, self.max_degree + 1), dtype=torch.float32)
            one_hot_tensor[torch.arange(nn), indices.squeeze()] = 1        
            sub_features = torch.cat((self.bigX[torch.tensor(union)], one_hot_tensor), dim=1)
        else:
            sub_features = self.bigX[torch.tensor(union)]

        sub_classes = self.bigY[torch.tensor(union)]
        unique_elements, counts = torch.unique(sub_classes, return_counts=True)
        max_count = torch.max(counts)
        most_frequent_elements = unique_elements[counts == max_count]
        gt_label = torch.zeros(int(self.num_classes))
        gt_label[int(most_frequent_elements[0].item())]=1

        return sub_features, subadjacency, gt_label
    
    
    def __sample_graph_sequential(self, min_num_nodes_threshold, max_nodes_threshold):
        assert(not self.args.directed)

        target_nn = random.choice(range(min_num_nodes_threshold, max_nodes_threshold))
        print(f'Target nn:{target_nn}')
        center = random.choice(range(1,self.num_nodes+1))
        nodes_list = [center]
        while len(nodes_list)<target_nn:
            new_node = random.choice([self.list_of_neighbors[i][j] for i in nodes_list for j in range(len(self.list_of_neighbors[i]))])
            if new_node not in nodes_list:
                nodes_list.append(new_node)

        my_tensor = torch.tensor(nodes_list)
        mask = torch.isin(self.bigA, my_tensor).all(dim=1)
        filtered_rows = self.bigA[mask]

        number_to_index = {num: idx for idx, num in enumerate(nodes_list)}
        adj = torch.zeros((len(nodes_list), len(nodes_list)), dtype=torch.float32)
        for num1, num2 in filtered_rows:
            idx1 = number_to_index[num1.item()]
            idx2 = number_to_index[num2.item()]
            adj[idx1, idx2] = 1

        
        adj = torch.max(adj, adj.T)
        adj += torch.eye(adj.shape[0])

        sub_degrees = torch.sum(adj, dim=0) - 1
        sub_degrees = sub_degrees.reshape(target_nn, 1)
        sub_degrees = sub_degrees.to(torch.int)

        one_hot_tensor = torch.zeros((target_nn, self.max_degree + 1), dtype=torch.float32)
        one_hot_tensor[torch.arange(target_nn), sub_degrees.squeeze()] = 1        

        sub_features = torch.cat((self.bigX[my_tensor], one_hot_tensor), dim=1)

        assert(sub_features.shape[0]==target_nn)

        label = torch.zeros(self.num_classes, dtype=torch.int32)
        random_index = torch.randint(0, self.num_classes, (1,)).item()
        label[random_index] = 1

        return sub_features, adj, label.float()
    
    def __generate_dataset_graphs(self, num_graphs, max_nodes_threshold = None, min_num_nodes_threshold=None):
        
        Xs, Ys, As = [], [], []
        
        for _ in range(num_graphs):
            if self.args.dataset == 'citeseer':
                new_graph = self.__sample_graph(k=3, picking=3, max_nodes_threshold=max_nodes_threshold, min_num_nodes_threshold=min_num_nodes_threshold)
            else:
                new_graph = self.__sample_graph_sequential(max_nodes_threshold=max_nodes_threshold, min_num_nodes_threshold=min_num_nodes_threshold)
            Xs.append(new_graph[0])
            As.append(new_graph[1])
            Ys.append(new_graph[2])
        
        return Xs, As, Ys

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return (self.X[idx].to(self.args.device), self.A[idx].to(self.args.device), self.y[idx].to(self.args.device), None)