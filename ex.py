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
from ex_extern import get_exp_method, GNN, save_dataset_to_file, delete_all_files_in_folder
import copy
import math
import sys
import ast


# --------- global variables
try:
    new_dataset = sys.argv[1]
    delete_old_visualizations = sys.argv[2]
    dataset_of_choice = int(sys.argv[3])
    explainers_to_test = ast.literal_eval(sys.argv[4]) 
    print('Running code with variables from shell')
except Exception as e:
    print('Importing variables from Shell has not worked')
    new_dataset = False
    delete_old_visualizations = True
    dataset_of_choice = 5 #some number between 1 and 10
    explainers_to_test = ['gnnex', 'gcam', 'subx']


# ------------ Code



# prepare: Erase old plots of explainers
import os
from graphxai.explainers import *
device = None
retrain_model = new_dataset
path_to_save = 'content/Users/plots_explainers/'






if delete_old_visualizations:
    delete_all_files_in_folder('content/plots_datasets')
    delete_all_files_in_folder('content/plots_explainers')


path = "content/datasets_ShapeGGen/datasets"+"_"+str(dataset_of_choice)+".pickle"
#new_dataset = True
if new_dataset == True:
    dataset = ShapeGGen(
        model_layers = 3,
        num_subgraphs = 110,
        subgraph_size = 13,
        prob_connection = 0.05,
        add_sensitive_feature = False,
        max_tries_verification = 3
    )

    path = save_dataset_to_file(dataset, "content/datasets_ShapeGGen/datasets.pickle")
    #with open("content/datasets_ShapeGGen/datasets.pickle", "wb") as file:
    #    pickle.dump(dataset, file)
else:
    with open("content/datasets_ShapeGGen/datasets_"+str(dataset_of_choice)+".pickle", "rb") as file:
        dataset = pickle.load(file)


# Train a model from scratch on the data:
path = path.replace("datasets_ShapeGGen", "models")
data = dataset.get_graph(use_fixed_split=True)
model = GNN(dataset.n_features, 64)
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001, weight_decay = 0.001)
criterion = torch.nn.CrossEntropyLoss()

# Train model:
#retrain_model = True
if retrain_model == True:
    for _ in range(1400):
        loss = train(model, optimizer, criterion, data)
    torch.save(model.state_dict(), path)
else:
    model.load_state_dict(torch.load(path))
    

# Final testing performance:
f1, acc, prec, rec, auprc, auroc = test(model, data, num_classes = 2, get_auc = True)
print('Test Accuracy score: {:.4f}'.format(acc))
print('Test F1 score: {:.4f}'.format(f1))
print('Test AUROC: {:.4f}'.format(auroc))



# model prediction for a random, but fixed, node
node_idx, gt_exp = dataset.choose_node(split = 'test')
pred = model(data.x, data.edge_index)[node_idx,:].argmax(dim=0)
label = data.y[node_idx]
pred_class = pred.clone()



# List of all methods:
exp_list = ['gnnex', 'gcam', 'subx', 'pgex', 
            'rand',  'pgmex', 'gbp', 'cam', 'grad']

exp_name_map = {
    'gnnex': 'GNNExplainer',
    'gcam': 'Grad-CAM',
    'subx': 'SubgraphX',
    'rand': 'Random',
    'pgmex': 'PGMExplainer',
    'pgex': 'PGExplainer',
    'gbp': 'Guided Backprop',
    'cam': 'CAM',
    'grad': 'Gradient' 
}
# global variable: sublist = ['gnnex', 'gcam', 'subx']
if len(explainers_to_test) >4:
    fig, ax = plt.subplots(2,math.ceil(len(explainers_to_test)*0.5+0.5), figsize = (20, 15))
else:
    fig, ax = plt.subplots(1,len(explainers_to_test)+1, figsize = (20, 15))
gt_exp[0].visualize_node(num_hops = 3, additional_hops = 0, graph_data = data, ax = ax[0], norm_imps = False,  show = True)
for index, method in enumerate(explainers_to_test):
    method_exp, forward_kwargs, feedback = get_exp_method(method, model, criterion, pred_class, dataset, node_idx, gt_exp)
    method_exp.visualize_node(num_hops = 3, additional_hops = 0, graph_data = data, ax = ax[index+1], norm_imps = False)
    ax[index+1].set_title(feedback)
ax[0].set_title("Ground Truth Explanation")
plt.show()
folder_path = 'content/plots_explainers'
save_path = os.path.join(folder_path, "plot_explainers_with_gt.pdf")  
plt.show()
plt.savefig(save_path, format = 'pdf')
#importance of colours: https://www.nature.com/articles/s41597-023-01974-x Fig. 8








