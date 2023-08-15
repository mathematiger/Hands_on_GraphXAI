import random as rnd
from collections import defaultdict
import time
import mlflow
import networkx as nx
import numpy as np
import torch
from torch_geometric.utils import from_networkx
from tqdm import tqdm as tq
from torch_geometric.nn import HeteroConv, SAGEConv, Linear, GCNConv, GINConv, GCN
from graphxai.explainers import *

device = None


# -- adjusted from https://github.com/m30m/gnn-explainability/blob/main/benchmarks/infection.py
def create_dataset(max_dist):
    #max_dist = self.num_layers  # anything larger than max_dist has a far away label
    number_features = 2
    g = nx.erdos_renyi_graph(1000, 0.004, directed=True)
    N = len(g.nodes())
    labels = [0 for _ in range(N)]
    infected_nodes = rnd.sample(g.nodes(), 80)
    #alternate way to get node labels
    if max_dist == 1:
        for _ in range(N):
            labels[_] = 2
        for u in infected_nodes:
            labels[u] = 0
        for u in infected_nodes:
            for _ in g.neighbors(u):
                #print(58, _)
                if labels[_] >= 1:
                    labels[_] = 1
    if max_dist == 2:
        for _ in range(N):
            labels[_] = 3
        for u in infected_nodes:
            labels[u] = 0
        for u in infected_nodes:
            for _ in g.neighbors(u):
                if labels[_] >= 1:
                    labels[_] = 1
                for __ in g.neighbors(_):
                    if labels[__] >= 2:
                        labels[__] = 2
    features = np.zeros((N, number_features)).tolist()
    for _ in range(N):
        if labels[_] == 0:
            features[_] = [0.0,1.0]
        else:
            features[_] = [2.0, 0.0]

    data = from_networkx(g)
    data.x = torch.tensor(features, dtype=torch.float)
    data.y = torch.tensor(labels)
    data.num_classes = 1 + max_dist + 1
    
    edges_tensor = data.edge_index
    list1 = edges_tensor[0, :].tolist()
    list2 = edges_tensor[1, :].tolist()
    clist1 = list1+list2
    clist2 = list2+list1
    list1=clist1
    list2=clist2
    filtered_pairs = list(set([(x, y) for x, y in zip(list1, list2)]))
    list1 = [pair[0] for pair in filtered_pairs]
    list2 = [pair[1] for pair in filtered_pairs]
    sorted_lists = sorted(zip(list1, list2))
    sorted_list1, sorted_list2 = zip(*sorted_lists)
    edges_tensor =  torch.tensor([sorted_list1, sorted_list2])
    data.edge_index = edges_tensor
    # Create train, validation, and test masks
    train_mask = torch.zeros(N, dtype=torch.bool)
    val_mask = torch.zeros(N, dtype=torch.bool)
    test_mask = torch.zeros(N, dtype=torch.bool)
    for _ in range(N):
        choose_which_mask = rnd.random()
        if choose_which_mask < 0.4:
            train_mask[_] = True
        elif choose_which_mask < 0.6:
            val_mask[_] = True
        else:
            test_mask[_] = True
        if labels[_] == 0:
            train_mask[_] = False
            val_mask[_] = False
            test_mask[_] = False
            if choose_which_mask < 0.6:  # 60% of unique_solution_nodes for training
                train_mask[_] = True
            else:
                val_mask[_] = True  # Remaining for validation
    data.train_mask = torch.tensor(train_mask)
    data.val_mask = torch.tensor(val_mask)
    data.test_mask = torch.tensor(test_mask)
    data.n_features = number_features
    print('created new dataset')
    return data



class GNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels, dropout = 0.1)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels, dropout = 0.1)
        self.conv3 = SAGEConv(hidden_channels, hidden_channels, dropout = 0.1)
        self.conv4 = SAGEConv(hidden_channels, hidden_channels, dropout = 0.1)
        self.lin = torch.nn.Linear(hidden_channels, out_channels)
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        x = torch.relu(x)
        x = self.conv3(x, edge_index)
        x = torch.relu(x)
        x = self.conv4(x, edge_index)
        x = torch.relu(x)
        x = self.lin(x)
        #x = self.conv3(x)  # Final layer for node classification
        return x

class GNN_GCN_2(torch.nn.Module):
    '''
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels, improved = True, add_selve_loops = True, bias = True, dropout = 0.1)
        self.conv2 = GCNConv(hidden_channels, out_channels, improved = True, add_selve_loops = True, bias = False, dropout = 0.1)
        self.conv3 = GCNConv(hidden_channels, out_channels, improved = True, add_selve_loops = True, bias = False, dropout = 0.1)
        self.conv4 = GCNConv(hidden_channels, hidden_channels)
        self.lin = torch.nn.Linear(hidden_channels, out_channels)
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        #x = torch.relu(x)
        #x = self.lin(x)
        #x = self.conv3(x)  # Final layer for node classification
        return x
    '''
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
        super().__init__()
        self.mlp_gin1 = torch.nn.Linear(in_channels, hidden_channels)
        self.gin1 = GINConv(self.mlp_gin1)
        self.mlp_gin2 = torch.nn.Linear(hidden_channels, out_channels)
        self.gin2 = GINConv(self.mlp_gin2)
        self.mlp_gin3 = torch.nn.Linear(hidden_channels, out_channels)
        self.gin3 = GINConv(self.mlp_gin3)

    def forward(self, x, edge_index):
        # NOTE: our provided testing function assumes no softmax
        #   output from the forward call.
        x = self.gin1(x, edge_index)
        x = x.relu()
        x = self.gin2(x, edge_index)
        #x = x.relu()
        #x = self.conv3(x, edge_index)
        return x


class GNN_GCN_3(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, out_channels)
        self.conv4 = GCNConv(hidden_channels, hidden_channels)
        self.lin = torch.nn.Linear(hidden_channels, out_channels)
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        x = torch.relu(x)
        x = self.conv3(x, edge_index)
        #x = torch.relu(x)
        #x = self.lin(x)
        #x = self.conv3(x)  # Final layer for node classification
        return x





class GNN_GCN_4(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.conv4 = GCNConv(hidden_channels, out_channels)
        self.lin = torch.nn.Linear(hidden_channels, out_channels)
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        x = torch.relu(x)
        x = self.conv3(x, edge_index)
        x = torch.relu(x)
        x = self.conv4(x, edge_index)
        #x = torch.relu(x)
        #x = self.lin(x)
        #x = self.conv3(x)  # Final layer for node classification
        return x

class GNN_GCN_5(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.conv4 = GCNConv(hidden_channels, hidden_channels)
        self.conv5 = GCNConv(hidden_channels, out_channels)
        self.lin = torch.nn.Linear(hidden_channels, out_channels)
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        x = torch.relu(x)
        x = self.conv3(x, edge_index)
        x = torch.relu(x)
        x = self.conv4(x, edge_index)
        x = torch.relu(x)
        x = self.conv5(x, edge_index)
        #x = torch.relu(x)
        #x = self.lin(x)
        #x = self.conv3(x)  # Final layer for node classification
        return x




class GIN(torch.nn.Module):
    def __init__(self,input_feat, hidden_channels, classes = 2):
        super(GIN, self).__init__()
        self.mlp_gin1 = torch.nn.Linear(input_feat, hidden_channels)
        self.conv1 = GINConv(self.mlp_gin1)
        self.mlp_gin2 = torch.nn.Linear(hidden_channels, hidden_channels)
        self.conv2 = GINConv(self.mlp_gin2)
        self.mlp_gin3 = torch.nn.Linear(hidden_channels, classes)
        self.conv3 = GINConv(self.mlp_gin3)

    def forward(self, x, edge_index):
        # NOTE: our provided testing function assumes no softmax
        #   output from the forward call.
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)
        return x



def choose_xnode(data, label, pred):
    boolean_tensor = data.test_mask.tolist()
    #true_indices = torch.nonzero(boolean_tensor).squeeze()
    #random_true_index = torch.randint(len(true_indices), size=(1,)).item()
    #list of indices of label label and in test_mask:
    list_of_possible_indices = []
    for _ in range(len(data.y.tolist())):
        if pred.tolist()[_] == label and boolean_tensor[_] == True:
            list_of_possible_indices.append(_)
    #selected_index = true_indices[random_true_index]
    random_index = rnd.choice(list_of_possible_indices)
    #return selected_index.item()
    #print(127, 'number of elements in the test-set of label ', label, ' : ', len(data.y.tolist()))
    return random_index


def get_exp_method(method, model, criterion, pred_class, data, layers_of_gnn, node_idx):  
    start_time = time.time()
    if method=='gnnex':
        exp_method = GNNExplainer(model)
        forward_kwargs={'x': data.x,
                        'node_idx': int(node_idx),
                        'edge_index': data.edge_index}
        method_exp = exp_method.get_explanation_node(**forward_kwargs)
        #acc = graph_exp_acc(gt_exp = gt_exp[0], generated_exp = method_exp)
        feedback = "GNNExplainer"
    elif method=='grad':
        exp_method = GradExplainer(model, criterion = criterion)
        forward_kwargs={'x': data.x.to(device),
                        'y': data.y.to(device),
                        'node_idx': int(node_idx),
                        'edge_index': data.edge_index.to(device)}
        method_exp = exp_method.get_explanation_node(**forward_kwargs)
        #acc = graph_exp_acc(gt_exp = gt_exp[0], generated_exp = method_exp)
        feedback = "Gradient"
    elif method=='cam':
        exp_method = CAM(model, activation = lambda x: torch.argmax(x, dim=1))
        forward_kwargs={'x': data.x,
                        'y': data.y,
                        'node_idx': int(node_idx),
                        'edge_index': data.edge_index}
        method_exp = exp_method.get_explanation_node(**forward_kwargs)
        #acc = graph_exp_acc(gt_exp = gt_exp[0], generated_exp = method_exp)
        feedback = "CAM"
    elif method=='gcam':
        exp_method = GradCAM(model, criterion = criterion)
        forward_kwargs={'x':data.x.to(device),
                        'y': data.y.to(device),
                        'node_idx': int(node_idx),
                        'edge_index': data.edge_index.to(device),
                        'average_variant': [True]}
        method_exp = exp_method.get_explanation_node(**forward_kwargs)
        #acc = graph_exp_acc(gt_exp = gt_exp[0], generated_exp = method_exp)
        feedback = "Grad-CAM"
    elif method=='gbp':
        exp_method = GuidedBP(model, criterion = criterion)
        forward_kwargs={'x': data.x.to(device),
                        'y': data.y.to(device),
                        'node_idx': int(node_idx),
                        'edge_index': data.edge_index.to(device)}
        method_exp = exp_method.get_explanation_node(**forward_kwargs)
        #acc = graph_exp_acc(gt_exp = gt_exp[0], generated_exp = method_exp)
        feedback = "Guided Backprop"
    elif method=='glime':
        exp_method = GraphLIME(model)
        forward_kwargs={'x': data.x.to(device),
                        'node_idx': int(node_idx),
                        'edge_index': data.edge_index.to(device)}
        method_exp = exp_method.get_explanation_node(**forward_kwargs)
        #acc = graph_exp_acc(gt_exp = gt_exp[0], generated_exp = method_exp)
        feedback = "GraphLIME"
    elif method=='ig':
        exp_method = IntegratedGradExplainer(model, criterion = criterion)
        forward_kwargs = {'x': data.x.to(device),
                        'edge_index': data.edge_index.to(device),
                        'node_idx': int(node_idx),
                        'label': pred_class}
        method_exp = exp_method.get_explanation_node(**forward_kwargs)
        #acc = graph_exp_acc(gt_exp = gt_exp[0], generated_exp = method_exp)
        feedback = "Integreated  Grad"
    elif method=='glrp':
        exp_method = GNN_LRP(model)
        forward_kwargs={'x': data.x.to(device),
                        'edge_index': data.edge_index.to(device),
                        'node_idx': node_idx,
                        'label': pred_class,
                        'edge_aggregator':torch.sum}
        method_exp = exp_method.get_explanation_node(**forward_kwargs)
        #acc = graph_exp_acc(gt_exp = gt_exp[0], generated_exp = method_exp)
        feedback = "LRP"
    elif method=='pgmex':
        exp_method=PGMExplainer(model, explain_graph=False, p_threshold=0.1)
        forward_kwargs={'node_idx': node_idx,
                        'x': data.x.to(device),
                        'edge_index': data.edge_index.to(device),
                        'top_k_nodes': 10}
        method_exp = exp_method.get_explanation_node(**forward_kwargs)
        #acc = graph_exp_acc(gt_exp = gt_exp[0], generated_exp = method_exp)
        feedback = "PGM Explainer"
    elif method=='pgex':
        exp_method=PGExplainer(model, emb_layer_name = 'gin'+str(layers_of_gnn), max_epochs = 10, lr = 0.1)
        forward_kwargs={'node_idx': node_idx,
                        'x': data.x.to(device),
                        'edge_index': data.edge_index.to(device),
                        'label': pred_class}
        method_exp = exp_method.get_explanation_node(**forward_kwargs)
        #acc = graph_exp_acc(gt_exp = gt_exp[0], generated_exp = method_exp)
        feedback = "PG Explainer"
    elif method=='rand':
        exp_method = RandomExplainer(model)
        forward_kwargs={'x': data.x.to(device),
                        'node_idx': int(node_idx),
                        'edge_index': data.edge_index.to(device)}
        method_exp = exp_method.get_explanation_node(**forward_kwargs)
        #acc = graph_exp_acc(gt_exp = gt_exp[0], generated_exp = method_exp)
        feedback = "Random Explainer"
    elif method=='subx':
        #start_time = time.time()
        exp_method = SubgraphX(model, reward_method = 'gnn_score', num_hops = layers_of_gnn, rollout=5, min_atoms = 3)
        forward_kwargs={'node_idx': node_idx,
                        'x': data.x,
                        'edge_index': data.edge_index,
                        'label': pred_class,
                        'max_nodes': 9}
        method_exp = exp_method.get_explanation_node(**forward_kwargs)
        #acc = graph_exp_acc(gt_exp = gt_exp[0], generated_exp = method_exp)
        #end_time = time.time()
        #execution_time = end_time - start_time
        feedback = "SubgraphX"
        #feedback += "\nTime taken: {:.4f} seconds".format(execution_time)
    else:
        OSError('Invalid argument!!')
    end_time  = time.time()
    execution_time = end_time-start_time
    feedback += "\nTime taken: {:.4f} seconds".format(execution_time)
    return method_exp, forward_kwargs, feedback





