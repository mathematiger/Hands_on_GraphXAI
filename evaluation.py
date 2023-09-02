# Here, some scoring functions and other evaluation functions are implemented

# ----------- evaluating class expressions: Currently not in use. ------------
import random as random
import os.path as osp
import torch
import sys
from ce_generation import generate_cedict_from_ce
from graph_generation import compute_prediction_ce
import pandas as pd

from owlapy.model import OWLObjectProperty, OWLObjectSomeValuesFrom
from owlapy.model import OWLDataProperty
from owlapy.model import OWLClass, OWLClassExpression
from owlapy.model import OWLDeclarationAxiom, OWLDatatype, OWLDataSomeValuesFrom, OWLObjectIntersectionOf, OWLEquivalentClassesAxiom, OWLObjectUnionOf
from owlapy.model import OWLDataPropertyDomainAxiom
from owlapy.model import IRI
from owlapy.render import DLSyntaxObjectRenderer



def ce_fidelity(ce_for_fid, modelfid, datasetfid, node_type_expl, label_expl = -1, random_seed = 1): 
    fid_result = -1
    mask = datasetfid[node_type_expl].test_mask
    mask_tf = 0
    for value in mask.tolist():
        if str(value) == 'True' or str(value) == 'False':
            mask_tf = 1
            break
    metagraph = datasetfid.to_dict()
    ### falls node_type_expl == -1: Ändere dies auf das letzte aller möglichen labels
    if label_expl == -1:
        list_labels = datasetfid[node_type_expl].y
        label_expl = max(set(list_labels))
    modelfid.eval()
    pred = modelfid(datasetfid.x_dict, datasetfid.edge_index_dict).argmax(dim=-1)
    pred_list = pred.tolist()
    for index, value in enumerate(pred_list):
        if value != label_expl:
            pred_list[index] = 0
        else:
            pred_list[index] = 1
    pred = torch.tensor(pred_list)
    if mask_tf == 0:
        mask = datasetfid[node_type_expl]['test_mask']
        #print(205, label_expl, pred, len(pred.tolist()), pred.tolist().count(1), mask.tolist().count(True), mask)
        cedict = generate_cedict_from_ce(ce_for_fid)
        #mask = select_ones(mask, 100)
        # create new vector with samples only as true vector
        random_seed +=1
        random.seed(random_seed)
        smaller_mask = random.sample(mask.tolist(), k=min(200, len(mask.tolist())))
        mask = torch.tensor(smaller_mask)
    else:
        indices_of_ones = [i for i, value in enumerate(mask.tolist()) if value == True]
        random_seed +=1
        random.seed(random_seed)
        chosen_indices = random.sample(indices_of_ones, k=min(20, len(indices_of_ones)))
        mask = [i if i in chosen_indices else 0 for i in range(len(mask.tolist()))]
        mask = [x for x in mask if x != 0]
        mask = torch.tensor(mask)
        sys.exit()
    count_fids = 0
    count_zeros_test = 0
    count_zeros_gnn = 0
    for index in mask.tolist():
        cedict = generate_cedict_from_ce(ce_for_fid)
        result_ce_fid = compute_prediction_ce(cedict, metagraph, node_type_expl, index)
        if pred[index] ==result_ce_fid:
            count_fids +=1
        if result_ce_fid == 0:
            count_zeros_test +=1
        if pred[index] == 0:
            count_zeros_gnn +=1
    #print(226, 'zeros counted CE, GNN: ', count_zeros_test, count_zeros_gnn)
    fid_result = round(float(count_fids) / float(len(mask.tolist())),2)
    return fid_result




    

# TODO: Think of cases, where this could not work: How would we find the edge 1-1 in the house, if there are no 2-3 edges ?
def ce_confusion_iterative(ce, graph, current_graph_node): # current_graph_node is of form ['3',0], ['2',0]. ['2',1], etc. [nodetype, nodeid_of_nodetype]
    # TODO: Insert 'abstract current nodes', if a edge to a node not in the graph or without specified nodetype is called
    # save the number of abstract edges used (later, maybe instead of True / False as feedback ?
    
    
    result = set()
    if isinstance(ce, OWLClass):
        #print(282, 'is class')
        if current_graph_node[0] != remove_front(ce.to_string_id()):
            #print(283, 'class false', current_graph_node)
            return result, False, current_graph_node
        else:
            return result, True, current_graph_node
    elif isinstance(ce, OWLObjectProperty):
        edgdetype = remove_front(ce.to_string_id())
        available_edges = available_edges_with_nodeid(graph, current_graph_node[0], current_graph_node[1], edgetype) #form should be [edge, endnodetype, endnodeid]
        if len(available_edges) > 0:
                # TODO: Add this edge to result, as the edge has been found
            #result.update()
            # retrieve all available edges
            set_possible_edges = set()
            for aved in available_edges:
                set_possible_edges.update(aved[2]) 
            for edgeind in set_possible_edges:
                if edgeind not in result:
                    result.update(edgeind)
                    break
            return result, True, current_graph_node
        return result, False, current_graph_node
    elif isinstance(ce, OWLObjectSomeValuesFrom):
        #(305, 'is edge with prop')
        edgetype = remove_front(ce._property.to_string_id())
        available_edges = available_edges_with_nodeid(graph, current_graph_node[0], current_graph_node[1], edgetype) #form should be [edge, endnodetype, endnodeid]
        current_best_length = len(result)
        result_copy = copy.deepcopy(result)
        local_result = set()
        local_current_grnd = current_graph_node
        some_edgewas_true = False
        for aved in available_edges:
            local_result = set()
            for i in result_copy:
                local_result.update(set(i))
            local_result.add(aved[2])
            feed1, feed2, current_graph_node = ce_confusion_iterative(ce._filler, graph, [aved[0], aved[1]])
            #print(312, feed1, feed2, ce._filler)
            '''
            if feed2 and len(feed1)>=current_best_length:
                print(319, feed1)
                current_best_length = len(feed1)
                local_result_intern = feed1
                local_current_grnd = current_graph_node
            '''
            if feed2:
                some_edgewas_true = True
                #(319, feed1, current_graph_node)
                current_best_length = len(feed1)
                local_result_intern = feed1
                local_current_grnd = current_graph_node
                return set(list(local_result)+list(local_result_intern)), True, local_current_grnd
        if some_edgewas_true == False:
            current_graph_node = 'abstract'
            return result, True, current_graph_node
            
            
        #if current_best_length >0:
        #    return set(list(local_result)+list(local_result_intern)), True, local_current_grnd
        #else:
        #    return result, False, current_graph_node
    elif isinstance(ce, OWLObjectIntersectionOf):
        return_truth = True
        for op in ce.operands(): #TODO: First select class if available, then further edges
            if isinstance(op, OWLClass) == True: 
                feed1, feed2, current_graph_node = ce_confusion_iterative(op, graph, current_graph_node)
                if feed1 != None:
                    result.update(list(feed1))
                if feed2 == False:
                    return_truth = False
        for op in ce.operands():
            if isinstance(op, OWLClass) == False:
                feed1, feed2, current_graph_node = ce_confusion_iterative(op, graph, current_graph_node)
                if feed1 != None:
                    result.update(list(feed1))
                if feed2 == False:
                    return_truth = False
        return result, return_truth, current_graph_node
    else:
        return result, False, current_graph_node
    return result, False, current_graph_node




def ce_confusion(ce,  motif = 'house'):
    motifgraph = dict()
    if motif == 'house':
        motifgraph = {('3', 'to', '2') :(torch.tensor([0,0], dtype = torch.long), 
            torch.tensor([0,1], dtype = torch.long), [0,1]),
                     ('2', 'to', '3') :(torch.tensor([0,1], dtype = torch.long), 
            torch.tensor([0,0], dtype = torch.long), [1,0]),
                      ('2', 'to', '1') :(torch.tensor([0,1], dtype = torch.long), 
            torch.tensor([0,1], dtype = torch.long), [2,3]),
                     ('1', 'to', '2') :(torch.tensor([0,1], dtype = torch.long), 
            torch.tensor([0,1], dtype = torch.long), [3,2]),
                      ('2', 'to', '2') :(torch.tensor([0,1], dtype = torch.long), 
            torch.tensor([1,0], dtype = torch.long),[4,4]),
                       ('1', 'to', '1') :(torch.tensor([0,1], dtype = torch.long), 
            torch.tensor([1,0], dtype = torch.long),[5,5]),
                      # now the abstract class is included
                      ('0', 'to', 'abstract') :(torch.tensor([0], dtype = torch.long), 
            torch.tensor([0], dtype = torch.long), [-1]),
                     ('abstract', 'to', '0') :(torch.tensor([0], dtype = torch.long), 
            torch.tensor([0], dtype = torch.long), [-1]),
                      ('1', 'to', 'abstract') :(torch.tensor([0,1], dtype = torch.long), 
            torch.tensor([0,0], dtype = torch.long), [-1,-1]),
                     ('abstract', 'to', '1') :(torch.tensor([0,0], dtype = torch.long), 
            torch.tensor([0,1], dtype = torch.long), [-1,-1]),
                      ('2', 'to', 'abstract') :(torch.tensor([0,1], dtype = torch.long), 
            torch.tensor([0,0], dtype = torch.long), [-1,-1]),
                     ('abstract', 'to', '2') :(torch.tensor([0,0], dtype = torch.long), 
            torch.tensor([0,1], dtype = torch.long), [-1,-1]),
                      ('3', 'to', 'abstract') :(torch.tensor([0], dtype = torch.long), 
            torch.tensor([0], dtype = torch.long), [-1]),
                     ('abstract', 'to', '3') :(torch.tensor([0], dtype = torch.long), 
            torch.tensor([0], dtype = torch.long), [-1]),
                      ('abstract', 'to', 'abstract') :(torch.tensor([0], dtype = torch.long), 
            torch.tensor([0], dtype = torch.long), [-1])   
                  }
    test_bla = ce_confusion_iterative(ce, motifgraph, ['3',0])
    #print(test_bla)

    
def ce_score_fct(ce, list_gnn_outs, lambdaone, lambdatwo):
    #avg_gnn_outs-lambda len_ce - lambda_var
    length_of_ce = length_ce(ce)
    mean = sum(list_gnn_outs) / len(list_gnn_outs)
    squared_diffs = [(x - mean) ** 2 for x in list_gnn_outs]
    sum_squared_diffs = sum(squared_diffs)
    variance = sum_squared_diffs / (len(list_gnn_outs))
    return mean-lambdaone*length_of_ce-lambdatwo*variance