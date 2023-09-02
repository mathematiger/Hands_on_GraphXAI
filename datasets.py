import torch
import os.path as osp
from torch_geometric.data import HeteroData
from torch_geometric.datasets import ExplainerDataset
from torch_geometric.datasets.graph_generator import BAGraph
from sklearn.model_selection import train_test_split
import random
from collections import Counter
from torch_geometric.datasets import OGB_MAG, DBLP
import torch_geometric.transforms as T

def count_ints_total(input_list, intput):
    count = 0
    for element in input_list:
        if element == intput:
            count += 1
    return count
def count_ints_until_entry(input_list, intput, entry):  #works
    return count_ints_total(input_list[:entry], intput)


#utils: retrieve the second argument of the list_current_to_new_indices:
def new_index(list_of_pairs, index):
    for pair in list_of_pairs:
        if pair[0] == index:
            return pair[1]

import random
def replace_random_zeros_with_one_or_three(input_list, prob_replace=0.07):
    output_list = []
    label_list = []
    counter = Counter(input_list)

    # Iterate over the counter and print the values and their frequencies
    for index_value in range(len(input_list)):
        if input_list[index_value] == 1:
            input_list[index_value] = 2
        elif input_list[index_value] == 2:
            input_list[index_value] = 1
    
    for value in input_list:
        if value == 0 and random.random() < 2*prob_replace:
            output_list.append(3)
            label_list.append(0)
        elif value == 0 and random.random() < prob_replace:
            output_list.append(1)
        elif value == 0 and random.random() < prob_replace:
            output_list.append(2)
        else:
            output_list.append(value)
            if value == 3:
                label_list.append(1)
    counter = Counter(input_list)
    # Iterate over the counter and print the values and their frequencies
    #for value, frequency in counter.items():
    #    print(f"Value: {value}, Frequency: {frequency}")
    return output_list, label_list



# it creates houses with labels 3-2-1 (top->bottom)
def create_hetero_ba_houses(not_house_nodes, houses):
    dataset = ExplainerDataset(
    graph_generator=BAGraph(num_nodes=not_house_nodes, num_edges=2),
    motif_generator='house',
    num_motifs=houses,
    )
    homgraph = dataset.get_graph()
    listnodetype = homgraph.y.tolist()
    listedgeindex = homgraph.edge_index.tolist()

    #randomly change some nodes of type 0 to type 3 or 1 and also retrieve a list of labels for nodes of type '3'
    listnodetype, label_list = replace_random_zeros_with_one_or_three(listnodetype, 0.1)

    number_of_each_type = []
    for i in range(4):
        number_of_each_type.append(count_ints_total(listnodetype, i))

    #[current index, new_index], where new_indes = count_ints_until_entry(... , label-of-current-index, current_index)
    list_current_to_new_indices = []
    for i in range(4):
        help_list_current_to_new_index = []
        for ind in range(len(listnodetype)):
            if listnodetype[ind] == i:
                help_list_current_to_new_index.append([ind, count_ints_until_entry(listnodetype, i, ind)])
        list_current_to_new_indices.append(help_list_current_to_new_index)


    hdata = HeteroData()
    #create nodes + feature 1 
    list_different_node_types = [str(i) for i in list(set(listnodetype))]
    for nodetype in list_different_node_types:
        hdata[nodetype].x = torch.ones(number_of_each_type[int(nodetype)], 1)
    #asign labels to node 3:
    hdata['3'].y = torch.tensor(label_list)


    #create edges
    for type_start_index in range(len(list_different_node_types)):
        for type_end_index in range(type_start_index, len(list_different_node_types)): 
            new_indices_start_list = []
            new_indices_end_list = []
            for start_node_index in range(len(listedgeindex[0])):
                # get nodetype of this label
                type_start = listnodetype[listedgeindex[0][start_node_index]]
                type_end = listnodetype[listedgeindex[1][start_node_index]]
                # check, if the labels are the wanted labels
                if type_start == type_start_index and type_end == type_end_index:
                    # get the new indizes:
                    look_up_list_start_node = list_current_to_new_indices[type_start]
                    look_up_list_end_node = list_current_to_new_indices[type_end]
                    new_start_index = new_index(look_up_list_start_node, listedgeindex[0][start_node_index])
                    new_end_index = new_index(look_up_list_end_node, listedgeindex[1][start_node_index])
                    new_indices_start_list.append(new_start_index)
                    new_indices_end_list.append(new_end_index)
            if new_indices_start_list and new_indices_end_list:
                #print(list_different_node_labels[label_start_index], [new_indices_start_list, new_indices_end_list])
                hdata[list_different_node_types[type_start_index], 'to', list_different_node_types[type_end_index]].edge_index = torch.tensor([new_indices_start_list, new_indices_end_list])
                if type_start_index != type_end_index:
                    hdata[list_different_node_types[type_end_index], 'to', list_different_node_types[type_start_index]].edge_index = torch.tensor([new_indices_end_list, new_indices_start_list])
    #only take nodes with labels
    idx = torch.arange(number_of_each_type[3])
    train_idx, valid_and_test_idx = train_test_split(
        idx,
        train_size=0.4,
    )
    valid_idx, test_idx = train_test_split(
        valid_and_test_idx,
        train_size=0.4,
    )
    hdata['3'].train_mask = torch.tensor(train_idx)
    hdata['3'].val_mask = torch.tensor(valid_idx)
    hdata['3'].test_mask = torch.tensor(test_idx)              
    return hdata


# ------------------- DBLP Dataset


def initialize_dblp():
    path = osp.join(osp.dirname(osp.realpath(__file__)), '../../data/DBLP')
    # We initialize conference node features with a single one-vector as feature:
    target_category_DBLP = 'conference'
    dataset = DBLP(path, transform=T.Constant(node_types=target_category_DBLP))
    target_category_DBLP = 'author'  # we want to predict classes of author
    # 4 different classes for author:
    #   database, data mining, machine learning, information retrieval.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    target = 'author'
    data = dataset[0]
    data =  data.to(device)
    return data, target