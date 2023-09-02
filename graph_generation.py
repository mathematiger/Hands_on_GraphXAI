import torch
import torch as th
import os.path as osp
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.datasets import OGB_MAG, DBLP
from torch_geometric.nn import HeteroConv, SAGEConv, Linear
import torch_geometric
from torch_geometric.data import HeteroData
from random import randint
import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
import dgl
import colorsys
import random
import sys
import copy
random_seed = 1
random.seed(random_seed)



# ------------------ utils + functions to randomly create graphs
def read_list_of_lists_from_file(filename):
    return torch.load(filename)


def save_results_to_file(list_of_lists_to_save, filename):
    torch.save(list_of_lists_to_save, filename)


def list_edge_feat(graph):
    list_edge_features_func = []
    for edge in graph.edge_types:
        if edge[1] not in list_edge_features_func:
            list_edge_features_func.append(edge[1])
    return list_edge_features_func


def list_node_feat(graph):
    return graph.node_types


def graphdict_and_features_to_heterodata(graph_dict, features_list):
    hdata = HeteroData()
    # create features and nodes
    for name_tuple in features_list:
        name = name_tuple[0]
        hdata[name].x = name_tuple[1]
    # create edges
    # read from dict
    for edge in graph_dict:
        hdata[edge[0], edge[1], edge[2]].edge_index = torch.tensor([graph_dict[edge][0].tolist(), 
                                            graph_dict[edge][1].tolist()], dtype=torch.long)
    return hdata


def heteroDatainfo(hetdata):
    list_n_types = list_node_feat(hetdata)
    node_types = [] # [[node_type, unique_values(int), #of features] for each nodetype]
    for nodet in list_n_types: # create list [nodetype, unique values(int), size]
        list_features = torch.empty(25, 0)
        list_int_unique = list()
        try:
            list_features = hetdata[nodet].x
            list_int_unique = list(set([int(i) for i in list(list_features.unique())])) # what does this do?
        except Exception as e:
            print(f"64 gg Here we skiped the error: {e}")
        node_types.append([nodet, list_int_unique, list_features.size(dim=1)])
    metapath_types = hetdata.edge_types # saving possible meta-paths
    return node_types, metapath_types


def search_triples(typesearched, data):
    list_type = []
    for triple in data.edge_types:
        if triple[0] == typesearched:
            list_cand = [triple[1], triple[2]]
            if(list_cand not in list_type):
                list_type.append(list_cand)
        if triple[2] == typesearched:  
            # we want to create also graphs with metapaths the other way around
            list_cand = [triple[1], triple[0]]
            if(list_cand not in list_type):
                list_type.append(list_cand)
    return(list_type)
#graph generation
def is_new_node_created(percent_new_node, random_seed = 88):
    random_seed +=1
    random.seed(random_seed)
    return random.random() < percent_new_node


def random_meta_without_startn(startn, available_meta, random_seed = 94):
    #find index for target_cat:
    nameind = 0
    for listind in range(0, len(available_meta) - 1):
        if available_meta[listind][0] == startn:
            nameind = listind
            break
    random_seed +=1
    random.seed(random_seed)
    rand_nr = randint(0, len(available_meta[nameind][1]) - 1)
    # create rest of metadata:
    return available_meta[nameind][1][rand_nr]


def create_random_start_graph(startn, list_available_meta):
    available_meta = list_available_meta
    meta_without_startn = random_meta_without_startn(startn, available_meta)  # chooses a random meta-path for completion of start-graph
    end=0
    if startn == meta_without_startn[1]:
        end = 1
    
    start_graph_dict =  {(startn, meta_without_startn[0], 
            meta_without_startn[1]):(torch.tensor([0], dtype = torch.long), 
            torch.tensor([end], dtype = torch.long))}
    start_graph_dict.update({(meta_without_startn[1], 
            meta_without_startn[0], startn):(torch.tensor([end], dtype = torch.long), 
            torch.tensor([0], dtype = torch.long))})
    return start_graph_dict


def create_new_edge(some_graph_dict, percent_new_node, list_available_meta):
    available_meta = list_available_meta
    # choose random start node
        # call current available node-types
    some_graph = dgl.heterograph(some_graph_dict)  # using dgl just as a work-around, it would also work otherwise
    rand_start_node_type = some_graph.ntypes[randint(0, len(some_graph.ntypes)-1)]  # create a uniform-random start-node for new edge
    rand_start_node_index = randint(0, some_graph.num_nodes(rand_start_node_type)-1)
    # choose metapath to add
    rand_edge_type, rand_end_node_type = random_meta_without_startn(rand_start_node_type, available_meta)
    
    # choose (by probability) to add a new node or not
    is_new_node_created_bool = is_new_node_created(percent_new_node)  # formula which chooses; not yet well-written
    if is_new_node_created_bool:
        # target_node_index is 1 higher now:
        # check, if node is in graph
        if rand_end_node_type in some_graph.ntypes:
            rand_end_node_index = some_graph.num_nodes(rand_end_node_type)
        else:
            rand_end_node_index = 0
        # update number of nodes and features
        # check, if key is already in dict and create start and end-tensors
    else:
        if rand_end_node_type in some_graph.ntypes:
            rand_end_node_index = randint(0, max(0,some_graph.num_nodes(rand_end_node_type)-1))
        else:
            rand_end_node_index = 0  
    if (rand_start_node_type, rand_edge_type, rand_end_node_type) in some_graph_dict.keys():
        # extract tensors
        tensor_start_end = some_graph_dict[(rand_start_node_type, rand_edge_type, rand_end_node_type)]
        # update tensors
        tensor_start = tensor_start_end[0]
        tensor_start = torch.cat((tensor_start, torch.tensor([rand_start_node_index], dtype = torch.long)),0)
        tensor_end = tensor_start_end[1]
        tensor_end = torch.cat((tensor_end, torch.tensor([rand_end_node_index], dtype = torch.long)),0)
    else:
        # if edge is new, create a tensor for start and end each, just containing the indices of the new edge
        tensor_start = torch.tensor([rand_start_node_index], dtype = torch.long)
        tensor_end = torch.tensor([rand_end_node_index], dtype = torch.long)

    # update:
    some_graph_dict.update({(rand_start_node_type, rand_edge_type, rand_end_node_type) : (tensor_start, tensor_end)})
    some_graph_dict.update({(rand_end_node_type, rand_edge_type, rand_start_node_type) : (tensor_end, tensor_start)})
    return some_graph_dict


def create_graph_with_n_edges(target_type, num, percent_new_node, list_available_meta):
    if num==1:
        return create_random_start_graph(target_type, list_available_meta)
    if num >1: 
        dict_current_graph =  create_random_start_graph(target_type, list_available_meta)
        for i in range(num-1):
            dict_current_graph = create_new_edge(dict_current_graph, percent_new_node, list_available_meta)
        return dict_current_graph
    else: return 'wrong input, need a number'


def one_node_features(nodetype_list, random_seed = 180):
    rnd_feat_list = []
    for j in range(nodetype_list[2]):
        random_seed +=1
        random.seed(random_seed)
        rnd_feat_list.append(int(random.choice(nodetype_list[1])))
    tensor_features = torch.tensor(rnd_feat_list, dtype=torch.float)
    return tensor_features


def all_node_features_one_type(nodename, dict_current_graph, hetdata, random_seed = 190):  
    # call with the list for one nodetype, receive an appended random feature-vector
    node_info = heteroDatainfo(hetdata)[0]
    # obtain node_info_triplet for nodename
    node_info_triplet = []  
    for triplets in node_info:
        if triplets[0] == nodename:
            node_info_triplet = triplets
            break
    # obtain number of nodes
    num_nodes = dgl.heterograph(dict_current_graph).num_nodes(nodename)
    # create node_features
    list_node_features = []
        # for each node: call one_node_features
        # save the obtained tensors to a list
    for _ in range(num_nodes):
        list_node_features.append(one_node_features(node_info_triplet, random_seed))
    # make a feature_matrix for this node_type
    feature_tensor_matrix = torch.stack(list_node_features)
    return feature_tensor_matrix


def create_features_to_dict(graph_dict, origdata, random_seed = 212):
    features_list = []
    # list of available nodetypes:
    listntypes = dgl.heterograph(graph_dict).ntypes
    for nodename in listntypes:
        features_list.append([nodename,all_node_features_one_type(nodename, graph_dict, origdata, random_seed)])
    # features_list = [nodetype, features]
    return graph_dict, features_list




     
def get_end_indices(dict_graph, start_type, edge_type, end_type, indices_start):
    start_values = dict_graph[(start_type, edge_type, end_type)][0].tolist()
    end_values = dict_graph[(start_type, edge_type, end_type)][1].tolist()
    indices_end = list()
    for index, value in enumerate(start_values):
        if value in indices_start:
            indices_end.append(end_values[index])
    #print('gg', 296, indices_end, start_values, start_type, end_type)
    return indices_end



def add_feat_one_to_dict(graph_dict):
    listntypes = dgl.heterograph(graph_dict).ntypes
    hd = HeteroData()
    #add features
    for nodetype in listntypes:
        num_nodes = dgl.heterograph(graph_dict).num_nodes(nodetype)
        hd[nodetype].x = torch.ones(num_nodes, 1)
    # add edges
    for edge in graph_dict:
        hd[edge[0], edge[1], edge[2]].edge_index = torch.tensor([graph_dict[edge][0].tolist(), 
                                            graph_dict[edge][1].tolist()], dtype=torch.long)
    return hd
        
    

        
        
    
    


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
# ------------------------  Evaluate graphs on BAHouse Dataset
        
def compute_confusion_house(dict_graph):
    house_graph = {('3', 'to', '2') :(torch.tensor([0,0], dtype = torch.long), 
            torch.tensor([0,1], dtype = torch.long)),
                     ('2', 'to', '3') :(torch.tensor([0,1], dtype = torch.long), 
            torch.tensor([0,0], dtype = torch.long)),
                      ('2', 'to', '1') :(torch.tensor([0,1], dtype = torch.long), 
            torch.tensor([0,1], dtype = torch.long)),
                     ('1', 'to', '2') :(torch.tensor([0,1], dtype = torch.long), 
            torch.tensor([0,1], dtype = torch.long)),
                      ('2', 'to', '2') :(torch.tensor([0,1], dtype = torch.long), 
            torch.tensor([1,0], dtype = torch.long)),
                       ('1', 'to', '1') :(torch.tensor([0,1], dtype = torch.long), 
            torch.tensor([1,0], dtype = torch.long))
                  }
    tp,fp,fn = 0,0,0   #tn is always negative 
    #fp is number of edges in dict_graph
    total_edgesfp = 0
    for edge, indices in dict_graph[0][0].items():
        total_edgesfp +=len(indices[0].tolist())
    fp = fp + float(total_edgesfp/2)
    total_edgesfn = 0
    for edge, indices in house_graph.items():
        total_edgesfn +=len(indices[0].tolist())
    fn = fn + float(total_edgesfn/2)
    #we go step by step through the graph and compute tp,fp,fn
    # start with tp
    checkpoint = 0 
    dict_graph = dict_graph[0][0]
    
    
    if ('3', 'to', '2') in dict_graph:
        thtw_indices = get_end_indices(dict_graph, '3', 'to', '2', [0])
        if len(thtw_indices)>=1:
            tp +=1
            fn -=1
            fp -=1
            checkpoint = 1
    if checkpoint == 1 and ('2', 'to', '1') in dict_graph:
        twon_indices = get_end_indices(dict_graph, '2', 'to', '1', thtw_indices)
        if len(thtw_indices)>=1:
            tp +=1
            fn -=1
            fp -=1
            checkpoint = 2
    if checkpoint == 2 and ('1', 'to', '1') in dict_graph:
        onon_indices = get_end_indices(dict_graph, '1', 'to', '1', twon_indices)
        if len(onon_indices)>=1:
            tp +=1
            fn -=1
            fp -=1
            checkpoint = 3
    if checkpoint == 3 and ('1', 'to', '2') in dict_graph:
        ontw_indices = get_end_indices(dict_graph, '1', 'to', '2', onon_indices)
        if len(ontw_indices)>=1:
            tp +=1
            fn -=1
            fp -=1
            checkpoint = 4
    if checkpoint == 4:
        if ('2', 'to', '3') in dict_graph:
            twth_indices = get_end_indices(dict_graph, '2', 'to', '3', ontw_indices)
            if len(twth_indices)>=1 and 0 in twth_indices:
                tp +=1
                fn -=1
                fp -=1
                checkpoint = 5
        if ('2', 'to', '2') in dict_graph:
            twtw_indices = get_end_indices(dict_graph, '2', 'to', '2', ontw_indices)
            if len(twtw_indices)>=1:  #does not matter, if we have taken exactly this path, we only land in this case, if there was a path 3-2-1-1-2-2
                for element in twtw_indices:
                    if element in thtw_indices:
                        tp +=1
                        fn -=1
                        fp -=1
                        checkpoint = 5
                        break
    if checkpoint == 1 and ('2', 'to', '2') in dict_graph:
        twtwcp1_indices = get_end_indices(dict_graph, '2', 'to', '2', thtw_indices)
        if len(twtwcp1_indices)>=1:
            tp +=1
            fn -=1
            fp -=1
            checkpoint = 11
        for element in twtwcp1_indices:
            if element in thtw_indices:
                tp +=1
                fn -=1
                fp -=1
                checkpoint = 12
                break
    if checkpoint == 0 and ('1', 'to', '1') in dict_graph:
        start_ones = dict_graph[('1', 'to', '1')][0].tolist()
        #end_ones = dict_graph[('1', 'to', '1')][1].tolist()
        checkpoint = 21
        for middleindex in ('0','1','3'):
            #compute 3-mi indices
            #compute 1-mi indices
            #look, if they match
            if ('3', 'to', middleindex) in dict_graph and ('1', 'to', middleindex) in dict_graph:
                thmi_indices = get_end_indices(dict_graph, '3', 'to', middleindex, [0])
                onmi_indices = get_end_indices(dict_graph, '1', 'to', middleindex, start_ones)
                for element in onmi_indices:
                    if element in thmi_indices:
                        tp +=1
                        fn -=1
                        fp -=1
                        checkpoint = 22
                        break
            if checkpoint == 22:
                break              
    return tp,fp,fn    
   
    
    
def compute_accu(tp=0, fp=0, fn=0, tn = 0):
    return float(tp+tn)/float(fp+fn+tp+tn)
def compute_f1(tp=0, fp=0, fn=0, tn = 0):
    return 2 * tp / (2 * tp + fp + fn)








# This function is called, to add features to a graph. The graph is passed as an dictionary for heterogeneous graphs.
# Here, random features are added to the graph, st. the GNN can be evaluated on top. The GNN-result for several times are averaged as the result.
def add_features_and_predict_outcome(examples_to_test,                   #How many different features should be added to the graph
                                             cat_to_explain,             # the category to explain
                                             model,                      # the model
                                             data,                       # the dataset
                                             list_results,               # saving the best result
                                             graph_dict,                 # the graph from the CE
                                             filename,                   # Where the results should be stored
                                             ce_str = None,              # The CE as string
                                             compute_acc = False,        # Whether to compute accuracy
                                             random_seed = 650
                                             ):
    for _ in range(examples_to_test):
        features_list = create_features_to_dict(graph_dict, data, random_seed)[1]
        hd = graphdict_and_features_to_heterodata(graph_dict, features_list)
        local_list = []
        mean_pred = 0.0
        try:
            out = model(hd.x_dict, hd.edge_index_dict)
            result = round(out[0][cat_to_explain].item(), 2)
            print(510, result)
            mean_pred += out[0][cat_to_explain].item()
            local_list.append([[graph_dict, features_list], result, ce_str])
            #list_results.append([[graph_dict, features_list] , result, ce_str])
        except Exception as e:
            print(f"244 Here we skiped the error: {e}")
            print(255, 'in error: ', cat_to_explain, graph_dict)
    mean_pred = mean_pred / float(examples_to_test)      
    closest_value = None
    closest_index = 0
    closest_difference = 2^24
    for _ in range(len(local_list)):            #Here, the closest_index is chosen, which is the graph, whose prediction is closest to the mean.
        value = local_list[_][1]
        difference = abs(value - mean_pred)
        if difference < closest_difference:
            closest_difference = difference
            closest_value = value
            closest_index = _
    best_index = 0
    value = -2^24
    for _ in range(len(local_list)):
        if value < local_list[_][1]:
            value = local_list[_][1]
            best_index = _
    tph,fph,fnh = 0,0,0
    mean_acc = -1
    max_acc = -1
    max_f1 = -1
    if compute_acc:
        # compute accuracy of the mean
        tph,fph,fnh = compute_confusion_house(local_list[closest_index])
        mean_acc =compute_accu(tp=tph, fp=fph, fn=fnh)
        # compute accuracy of graph with the best GNN result
        tph,fph,fnh = compute_confusion_house(local_list[best_index])
        max_acc =compute_accu(tp=tph, fp=fph, fn=fnh)
        max_f1 = compute_f1(tp=tph, fp=fph, fn=fnh)
    # add results to the graph with the best index
    print(546, best_index, local_list, examples_to_test)
    local_list[best_index].append([max_acc, mean_acc, max_f1, tph,fph,fnh]) #accuracy does not change as here only different features are created
    list_results.append(local_list[best_index])
    sorted_list = sorted(list_results, key=lambda x: x[1], reverse = True)
    sorted_list_results = [x[1] for x in sorted_list]
    # TODO: don't save results, as this would only cause to much unneccesary data
    save_results_to_file(sorted_list, filename)
    return read_list_of_lists_from_file(filename)
    


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

# -------------- old, probably redundant functions
def create_random_graphs_and_predict_outcome(examples_to_test, 
                                             number_edges_of_each_sample, 
                                             percent_new_node, cat_to_explain,
                                             model, 
                                             data,
                                             list_results,
                                             list_available_meta,
                                             target
                                             ):
    for _ in range(examples_to_test):
        graph_dict = create_graph_with_n_edges(target, number_edges_of_each_sample, percent_new_node, list_available_meta)
        features_list = create_features_to_dict(graph_dict, data)[1]
        hd = graphdict_and_features_to_heterodata(graph_dict, features_list)
        out = model(hd.x_dict, hd.edge_index_dict)
        result = round(out[0][cat_to_explain].item(), 2)
        list_results.append([[graph_dict, features_list] , result])




def create_graphs_and_save(list_number_of_edges, list_percent_new_node, examples_to_test, cat_to_explain, filename, 
                           model,  data, list_available_meta, target):
    list_results = []
    for number_edges_of_each_sample in list_number_of_edges:
        for percent_new_node in list_percent_new_node:
            create_random_graphs_and_predict_outcome(examples_to_test, 
                                                     number_edges_of_each_sample, percent_new_node, 
                                                     cat_to_explain,
                                                     model, data,
                                                     list_results,
                                                     list_available_meta,
                                                     target
                                                     )
    sorted_list = sorted(list_results, key=lambda x: x[1], reverse = True)
    sorted_list_results = [x[1] for x in sorted_list]
    save_results_to_file(sorted_list, filename)


    
    
    
    
    


def create_graphs_for_heterodata(hd, should_new_graphs_be_created,
                                 list_number_of_edges, list_percent_new_node, 
                                 examples_to_test, target_node_type_to_explain, cat_to_explain, filename,
                                 model
                                 ):
    target = target_node_type_to_explain
    data = hd # TODO: Check, if data is really hd
    list_available_meta = []  # creates a list, with: each entry is a list [node-type, list(available meta-paths-continuations from this node-type)]
    #list_edge_features = list_edge_feat(data)
    list_node_types = list_node_feat(data)
    for metatype in list_node_types:
        list_available_meta.append([metatype, search_triples(metatype, data)])
    data = hd # TODO: Check, if data is really hd
    if should_new_graphs_be_created:
        create_graphs_and_save(list_number_of_edges, list_percent_new_node, 
                        examples_to_test, cat_to_explain, filename, model,  data, list_available_meta, target)
        saved_list = read_list_of_lists_from_file(filename)
    else:
        saved_list = read_list_of_lists_from_file(filename)
    return saved_list
    

    
    
def compute_tp_ce(cd, motif_graph, edge_taken_dict, max_found, current_node):
    if len(cd) == 0:
        return max_found
    next_edge = cd[0]
    if next_edge in motif_graph:
        edge = motif_graph[next_edge]
        del cd[0]
        for ni in edge[0][0]:
            value_acc = compute_tp_ce(cd, motif_graph, edge_taken_dict, max_found, current_node)
    
    
def compute_confusion_for_ce_line(cd, motif = 'house'):
    motif_graph = {}
    if motif == 'house':
        motif_graph = {('3', 'to', '2') :(torch.tensor([0,0], dtype = torch.long), 
            torch.tensor([0,1], dtype = torch.long), [0,1]),
                     ('2', 'to', '3') :(torch.tensor([0,1], dtype = torch.long), 
            torch.tensor([0,0], dtype = torch.long), [1,0]),
                      ('2', 'to', '1') :(torch.tensor([0,1], dtype = torch.long), 
            torch.tensor([0,1], dtype = torch.long), [2,3]),
                     ('1', 'to', '2') :(torch.tensor([0,1], dtype = torch.long), 
            torch.tensor([0,1], dtype = torch.long), [3,2]),
                      ('2', 'to', '2') :(torch.tensor([0,1], dtype = torch.long), 
            torch.tensor([1,0], dtype = torch.long),[4]),
                       ('1', 'to', '1') :(torch.tensor([0,1], dtype = torch.long), 
            torch.tensor([1,0], dtype = torch.long),[5])
                  }
    #create dict: unique id :  bool(path already taken)
    edge_taken_dict = dict()
    for key,value in motif_graph.items():
        for index in value[2]:
            edge_taken_dict[index] = 0
    #print(edge_taken_dict)
    #works
    max_value_tp = compute_tp_ce(cd, motif_graph, edge_taken_dict, 0, ['3', 0])    
    
    
def predict_ce_inkursive(ce_dict, graph_dict, node_type_to_explain, index_to_explain):
    
    #print('gg',231, (ce_dict[0][0][0][0], ce_dict[0][0][0][1], ce_dict[0][0][0][2]), index_to_explain, ce_dict[0][0][1][0])
    if not ce_dict[0]:
        return True
    elif (ce_dict[0][0][0][0], ce_dict[0][0][0][1], ce_dict[0][0][0][2]) in graph_dict:# and index_to_explain in ce_dict[0][0][1][0]:
        dict_values = graph_dict[(ce_dict[0][0][0][0], ce_dict[0][0][0][1], ce_dict[0][0][0][2])]
        dict_indstart = dict_values['edge_index']
        start_values = graph_dict[(ce_dict[0][0][0][0], ce_dict[0][0][0][1], ce_dict[0][0][0][2])]['edge_index'][0].tolist()
        end_values = graph_dict[(ce_dict[0][0][0][0], ce_dict[0][0][0][1], ce_dict[0][0][0][2])]['edge_index'][1].tolist()
        value_to_transmit = ce_dict[0][0][0][2]
        local_end_values = list()
        for indstart, valstart in enumerate(start_values):
            if index_to_explain == valstart:
                local_end_values.append(end_values[indstart])    
        #print('gg',235,'Was True', ce_dict[0])
        if len(local_end_values) != 0:
            local_cedict= ce_dict
            del local_cedict[0][0]
            for ind in local_end_values:
                if predict_ce_inkursive(local_cedict, graph_dict, value_to_transmit, ind):
                    return True
    else:
        print(251, 'gg', 'Was False', ce_dict[0])
        return False  



def compute_prediction_ce(ce_dict, graph_dict, node_type_to_explain, index_to_explain):
    prediction_positive = 0
    if predict_ce_inkursive(ce_dict, graph_dict, node_type_to_explain, index_to_explain):
        prediction_positive +=1
    return prediction_positive