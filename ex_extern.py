import graphxai
from graphxai.explainers import GradCAM
from graphxai.metrics import graph_exp_faith
import torch
import matplotlib.pyplot as plt
from graphxai.datasets import ShapeGGen
from torch_geometric.data import Data
from graphxai.explainers import PGExplainer, IntegratedGradExplainer, GNNExplainer
from graphxai.metrics import graph_exp_acc
import random
import pickle
import time
random_seed = 42
random.seed(random_seed)
from torch_geometric.nn import GINConv
from graphxai.gnn_models.node_classification import train, test
import os
from graphxai.explainers import *

device = None











def save_dataset_to_file(data, path):
    filename, extension = os.path.splitext(path)
    count = 1
    while os.path.exists(path):
        path = f"{filename}_{count}{extension}"
        count += 1
    
    with open(path, "wb") as file:
        pickle.dump(data, file)
    path_end, extension = os.path.splitext(path)
    return path


def delete_all_files_in_folder(folder_path):
    try:
        # Get a list of all files in the folder
        file_list = os.listdir(folder_path)

        # Iterate through the list and delete each file
        for filename in file_list:
            file_path = os.path.join(folder_path, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
        print("All files in the folder", folder_path,"have been deleted.")
    except Exception as e:
        print(f"Error occurred: {e}")




class GNN(torch.nn.Module):
    def __init__(self,input_feat, hidden_channels, classes = 2):
        super(GNN, self).__init__()
        self.mlp_gin1 = torch.nn.Linear(input_feat, hidden_channels)
        self.gin1 = GINConv(self.mlp_gin1)
        self.mlp_gin2 = torch.nn.Linear(hidden_channels, hidden_channels)
        self.gin2 = GINConv(self.mlp_gin2)
        self.mlp_gin3 = torch.nn.Linear(hidden_channels, classes)
        self.gin3 = GINConv(self.mlp_gin3)

    def forward(self, x, edge_index):
        # NOTE: our provided testing function assumes no softmax
        #   output from the forward call.
        x = self.gin1(x, edge_index)
        x = x.relu()
        x = self.gin2(x, edge_index)
        x = x.relu()
        x = self.gin3(x, edge_index)
        return x



def get_exp_method(method, model, criterion, pred_class, dataset, node_idx, gt_exp):  
    #method = method.lower()
    #node_idx, gt_exp = dataset.choose_node(split = 'test')
    data = dataset.get_graph(use_fixed_split=True)
    #node_idx = data.test_mask.nonzero(as_tuple=True)[0][0]
    #gt_exp = dataset.explanations[node_idx]
    start_time = time.time()
    if method=='gnnex':
        exp_method = GNNExplainer(model)
        forward_kwargs={'x': data.x.to(device),
                        'node_idx': int(node_idx),
                        'edge_index': data.edge_index.to(device)}
        method_exp = exp_method.get_explanation_node(**forward_kwargs)
        acc = graph_exp_acc(gt_exp = gt_exp[0], generated_exp = method_exp)
        feedback = "GNNExplainer, \n acc: " + str(round(acc,2))
    elif method=='grad':
        exp_method = GradExplainer(model, criterion = criterion)
        forward_kwargs={'x': data.x.to(device),
                        'y': data.y.to(device),
                        'node_idx': int(node_idx),
                        'edge_index': data.edge_index.to(device)}
        method_exp = exp_method.get_explanation_node(**forward_kwargs)
        acc = graph_exp_acc(gt_exp = gt_exp[0], generated_exp = method_exp)
        feedback = "Gradient, \n acc: " + str(round(acc,2))
    elif method=='cam':
        exp_method = CAM(model, activation = lambda x: torch.argmax(x, dim=1))
        forward_kwargs={'x': data.x.to(device),
                        'y': data.y.to(device),
                        'node_idx': int(node_idx),
                        'edge_index': data.edge_index.to(device)}
        method_exp = exp_method.get_explanation_node(**forward_kwargs)
        acc = graph_exp_acc(gt_exp = gt_exp[0], generated_exp = method_exp)
        feedback = "CAM, \n acc: " + str(round(acc,2))
    elif method=='gcam':
        exp_method = GradCAM(model, criterion = criterion)
        forward_kwargs={'x':data.x.to(device),
                        'y': data.y.to(device),
                        'node_idx': int(node_idx),
                        'edge_index': data.edge_index.to(device),
                        'average_variant': [True]}
        method_exp = exp_method.get_explanation_node(**forward_kwargs)
        acc = graph_exp_acc(gt_exp = gt_exp[0], generated_exp = method_exp)
        feedback = "Grad-CAM, \n acc: " + str(round(acc,2))
    elif method=='gbp':
        exp_method = GuidedBP(model, criterion = criterion)
        forward_kwargs={'x': data.x.to(device),
                        'y': data.y.to(device),
                        'node_idx': int(node_idx),
                        'edge_index': data.edge_index.to(device)}
        method_exp = exp_method.get_explanation_node(**forward_kwargs)
        acc = graph_exp_acc(gt_exp = gt_exp[0], generated_exp = method_exp)
        feedback = "Guided Backprop, \n acc: " + str(round(acc,2))
    elif method=='glime':
        exp_method = GraphLIME(model)
        forward_kwargs={'x': data.x.to(device),
                        'node_idx': int(node_idx),
                        'edge_index': data.edge_index.to(device)}
        method_exp = exp_method.get_explanation_node(**forward_kwargs)
        acc = graph_exp_acc(gt_exp = gt_exp[0], generated_exp = method_exp)
        feedback = "GraphLIME, \n acc: " + str(round(acc,2))
    elif method=='ig':
        exp_method = IntegratedGradExplainer(model, criterion = criterion)
        forward_kwargs = {'x': data.x.to(device),
                        'edge_index': data.edge_index.to(device),
                        'node_idx': int(node_idx),
                        'label': pred_class}
        method_exp = exp_method.get_explanation_node(**forward_kwargs)
        acc = graph_exp_acc(gt_exp = gt_exp[0], generated_exp = method_exp)
        feedback = "Integreated  Grad, \n acc: " + str(round(acc,2))
    elif method=='glrp':
        exp_method = GNN_LRP(model)
        forward_kwargs={'x': data.x.to(device),
                        'edge_index': data.edge_index.to(device),
                        'node_idx': node_idx,
                        'label': pred_class,
                        'edge_aggregator':torch.sum}
        method_exp = exp_method.get_explanation_node(**forward_kwargs)
        acc = graph_exp_acc(gt_exp = gt_exp[0], generated_exp = method_exp)
        feedback = "LRP, \n acc: " + str(round(acc,2))
    elif method=='pgmex':
        exp_method=PGMExplainer(model, explain_graph=False, p_threshold=0.1)
        forward_kwargs={'node_idx': node_idx,
                        'x': data.x.to(device),
                        'edge_index': data.edge_index.to(device),
                        'top_k_nodes': 10}
        method_exp = exp_method.get_explanation_node(**forward_kwargs)
        acc = graph_exp_acc(gt_exp = gt_exp[0], generated_exp = method_exp)
        feedback = "PGM Explainer, \n acc: " + str(round(acc,2))
    elif method=='pgex':
        exp_method=PGExplainer(model, emb_layer_name = 'gin2', max_epochs = 10, lr = 0.1)
        forward_kwargs={'node_idx': node_idx,
                        'x': data.x.to(device),
                        'edge_index': data.edge_index.to(device),
                        'label': pred_class}
        method_exp = exp_method.get_explanation_node(**forward_kwargs)
        acc = graph_exp_acc(gt_exp = gt_exp[0], generated_exp = method_exp)
        feedback = "PG Explainer, \n acc: " + str(round(acc,2))
    elif method=='rand':
        exp_method = RandomExplainer(model)
        forward_kwargs={'x': data.x.to(device),
                        'node_idx': int(node_idx),
                        'edge_index': data.edge_index.to(device)}
        method_exp = exp_method.get_explanation_node(**forward_kwargs)
        acc = graph_exp_acc(gt_exp = gt_exp[0], generated_exp = method_exp)
        feedback = "Random Explainer, \n acc: " + str(round(acc,2))
    elif method=='subx':
        #start_time = time.time()
        exp_method = SubgraphX(model, reward_method = 'gnn_score', num_hops = dataset.model_layers, rollout=5, min_atoms = 3)
        forward_kwargs={'node_idx': node_idx,
                        'x': data.x,
                        'edge_index': data.edge_index,
                        'label': pred_class,
                        'max_nodes': 10}
        method_exp = exp_method.get_explanation_node(**forward_kwargs)
        acc = graph_exp_acc(gt_exp = gt_exp[0], generated_exp = method_exp)
        #end_time = time.time()
        #execution_time = end_time - start_time
        feedback = "SubgraphX, \n acc: " + str(round(acc,2))
        #feedback += "\nTime taken: {:.4f} seconds".format(execution_time)
    else:
        OSError('Invalid argument!!')
    end_time  = time.time()
    execution_time = end_time-start_time
    feedback += "\nTime taken: {:.4f} seconds".format(execution_time)
    return method_exp, forward_kwargs, feedback