import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx, from_networkx
from torch_geometric.datasets import ExplainerDataset
from torch_geometric.datasets.graph_generator import BAGraph
from torch_geometric.nn import BatchNorm
import networkx as nx
import matplotlib.pyplot as plt
import os
import random as rnd
from torch_geometric.nn import SAGEConv, Linear, GCNConv
import numpy as np


def count_unique_indices(edge_pairs):
    unique_indices = set()
    for edge in edge_pairs:
        unique_indices.add(edge[0])
        unique_indices.add(edge[1])
    unique_indices = list(set(unique_indices))
    return len(unique_indices)

def create_graph_from_pairs(edge_pairs, list_features):
    unique_indices = count_unique_indices(edge_pairs)
    edge_pairs = torch.tensor(edge_pairs, dtype=torch.long).t()
    edge_index = torch.stack([edge_pairs[0], edge_pairs[1]], dim=0)
    node_features = torch.ones(unique_indices, 2)
    features_list = node_features.tolist()
    for i in range(len(features_list)):
        features_list[i] = [list_features[i] for _ in range(2)]
    features = torch.tensor(features_list, dtype = torch.float)
    edges_tensor = edge_index
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
    edge_index = edges_tensor
    data = Data(x=features, edge_index=edge_index)
    data.num_classes = 2
    return data




def visualize_data(data):
    # Convert edge_index to a NetworkX graph
    G = to_networkx(data, to_undirected= True)
    pos = nx.spring_layout(G, seed=42)
    #nx.draw(G, pos, with_labels=True, node_color="skyblue", node_size=1000, font_size=12)
    nx.draw_networkx(G, pos, with_labels=True, node_color="skyblue", node_size=1000, font_size=12)
    #labels = {i: x[i].item() for i in range(len(x))}
    #nx.draw_networkx_labels(G, pos, labels=labels)
    folder_path = 'content_sds/own_graphs'
    #  TODO uniquify the name !!!! 
    save_path = os.path.join(folder_path, "plot_graph.pdf")  
    plt.show()
    plt.savefig(save_path, format = 'pdf')


def fill_all_edges(data):
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
    return data


def find_edge_index_i_to_label_j(data, index, j):
    r_index = []
    for pre_index, value in enumerate(data.edge_index[0]):
        #print(82, value, index)
        if value.item() == index:
            r_index.append([value.item(), []])
            #print(84, pre_index, data.edge_index[1][pre_index].item(), value, data.y[data.edge_index[1][pre_index]].item(), j)
            if data.y[data.edge_index[1][pre_index]].item() == j:
                r_index[-1][1].append(data.edge_index[1][pre_index].item())
                #print(r_index)
    return r_index
    
def add_edges_to_motifs(data):
    edges_to_add_to_data_list = []
    for index, value in enumerate(data.y):
        if value.item() == 3:
            m_index = find_edge_index_i_to_label_j(data, index, 1)
            for middle_edge in m_index:
                for middle_node_index in middle_edge[1]:
                    s_index = find_edge_index_i_to_label_j(data, middle_node_index, 2)
                    for end_edge in s_index:
                        for end_node_index in end_edge[1]:
                            edges_to_add_to_data_list.append((index, end_node_index))
    list1 = [pair[0] for pair in edges_to_add_to_data_list]
    list2 = [pair[1] for pair in edges_to_add_to_data_list]
    list1 = data.edge_index[0].tolist() + list1
    list2 = data.edge_index[1].tolist() + list2
    tensor1 = torch.tensor(list1)
    tensor2 = torch.tensor(list2)
    data.edge_index = torch.tensor([list1, list2])
    data = fill_all_edges(data)
    return data


def change_label_3_2_to_1(data):
    list_labels = data.y.tolist()
    list_labels = [1 if x == 3 or x == 2 else x for x in list_labels]
    data.y = torch.tensor(list_labels)
    
    return data


def build_dataset(num_non_motif_nodes, num_motifs):
    dataset = ExplainerDataset(
    graph_generator=BAGraph(num_nodes=num_non_motif_nodes, num_edges=2),
    motif_generator='house',
    num_motifs=num_motifs,
    )
    graph = dataset.get_graph()
    graph = fill_all_edges(graph)
    graph = add_edges_to_motifs(graph)
    #adjust features: same as the label
    old_labels = graph.y.tolist()
    graph = change_label_3_2_to_1(graph)
    #add train, validation, test dataset !
    N = len(graph.y.tolist())
    train_mask = torch.zeros(N, dtype=torch.bool)
    val_mask = torch.zeros(N, dtype=torch.bool)
    test_mask = torch.zeros(N, dtype=torch.bool)
    counter_train = 0
    counter_validation = 0
    for _ in range(N):
        rnd_var = rnd.random()
        if rnd_var < 0.5 and counter_train < 0.5*N:
            train_mask[_] = True
        elif rnd_var < 0.7 and counter_validation < 0.7*N:
            val_mask[_] = True
        else:
            test_mask[_] = True    
    graph.train_mask = train_mask
    graph.val_mask = val_mask
    graph.test_mask = test_mask
    graph.num_classes = 2
    '''    
    graph.x =  torch.rand(N, 1)
    list_nodes_in_motif = []
    
    for _ in range(len(old_labels)):
        if graph.y.tolist()[_] == True:
            list_nodes_in_motif.append(graph.y.tolist()[_])
    graph.x[list_nodes_in_motif] = 3.0
    '''
    number_features = 2
    features = np.ones((N, number_features))
    features_list = features.tolist()
    for _ in range(len(features_list)):
        #features_list[_] = [2.0*rnd.random()+1.0 for _ in range(number_features)]
        features_list[_] = [0 for _ in range(number_features)]
        if old_labels[_] == 1:
            features_list[_] = [1.0 for _ in range(number_features)]
        if old_labels[_] == 2:
            features_list[_] = [2.0 for _ in range(number_features)]
        if old_labels[_] == 3:
            features_list[_] = [3.0 for _ in range(number_features)]
    features = torch.tensor(features_list, dtype = torch.float)
    graph.x = features
    #graph.x = torch.tensor(features, dtype=torch.float)
    return graph


def build_motif_hetero(graph):
    
    return hgraph


class GNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels, dropout = 0.5)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels, dropout = 0.5)
        self.conv3 = SAGEConv(hidden_channels, hidden_channels, dropout = 0.3)
        self.conv4 = SAGEConv(hidden_channels, hidden_channels, dropout = 0.3)
        self.lin = torch.nn.Linear(hidden_channels, out_channels)
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        x = torch.relu(x)
        x = self.conv3(x, edge_index)
        x = torch.relu(x)
        x = self.lin(x)
        return x



def test_created_dataset(data, model):
    model.eval()
    pred = round(model(data.x, data.edge_index)[0,0].item(), 2)
    pred = model(data.x, data.edge_index)
    feedback_class_1 = [round(x[1].item(),2) for x in pred]
    sum_feedback_1 = sum(feedback_class_1)
    print('Score on the own Dataset for each node: ', feedback_class_1)
    print('Total Score of own Dataset: ', sum_feedback_1)






