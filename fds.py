import random
from collections import defaultdict
import mlflow
import networkx as nx
import numpy as np
import torch
from torch_geometric.utils import from_networkx
from tqdm import tqdm as tq
from fds_data import create_dataset, GNN, choose_xnode, get_exp_method, GNN_GCN_4, GNN_GCN_3, GNN_GCN_2, GIN
from graphxai.gnn_models.node_classification import train, test, GCN_3layer_basic, GIN_3layer_basic
import matplotlib.pyplot as plt
import os
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import sys
import ast

# ------------ global variables
try: 
    new_dataset = ast.literal_eval(sys.argv[1])
    label_to_explain = int(sys.argv[2])
    explainers_to_test = ast.literal_eval(sys.argv[3])
    num_layers_gnn = ast.literal_eval(sys.argv[4])
except Exception as e:
    print(22, 'Running via Shell has not worked')
    new_dataset = False
    label_to_explain = 1
    explainers_to_test = ['gnnex', 'pgex']
    num_layers_gnn = 2

# -------------- rest of the code
path = 'content_fds/datasets/'
print(30, 'should new dataset be created: ', new_dataset)
if new_dataset:
    data = create_dataset(num_layers_gnn-1)
    torch.save(data, path+'fds'+str(num_layers_gnn))
else:
    data = torch.load(path+'fds'+str(num_layers_gnn))


#train GNN on data
path = 'content_fds/models/'


if num_layers_gnn == 4:
    model = GNN_GCN_4(in_channels=2, hidden_channels=64, out_channels=data.num_classes, num_layers = num_layers_gnn)
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.001, weight_decay = 0.001)
    criterion = torch.nn.CrossEntropyLoss()
    if new_dataset == True:
        for _ in range(500):
            loss = train(model, optimizer, criterion, data)
        torch.save(model.state_dict(), path+'model'+str(num_layers_gnn))
    else:
        model.load_state_dict(torch.load(path+'model'+str(num_layers_gnn)))
elif num_layers_gnn == 3:
    model = GNN_GCN_3(in_channels=2, hidden_channels=64, out_channels=data.num_classes, num_layers = num_layers_gnn)
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.001, weight_decay = 0.001)
    criterion = torch.nn.CrossEntropyLoss()
    if new_dataset == True:
        for _ in range(500):
            loss = train(model, optimizer, criterion, data)
        torch.save(model.state_dict(), path+'model'+str(num_layers_gnn))
    else:
        model.load_state_dict(torch.load(path+'model'+str(num_layers_gnn)))
elif num_layers_gnn == 2:
    model = GNN_GCN_2(in_channels=2, hidden_channels=64, out_channels=data.num_classes, num_layers = num_layers_gnn)
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.001, weight_decay = 0.001)
    criterion = torch.nn.CrossEntropyLoss()
    if new_dataset == True:
        for _ in range(900):
            loss = train(model, optimizer, criterion, data)
        torch.save(model.state_dict(), path+'model'+str(num_layers_gnn))
    else:
        model.load_state_dict(torch.load(path+'model'+str(num_layers_gnn)))




#  Testing
model.eval()
out = model(data.x, data.edge_index)
pred = out.argmax(dim=1)  # Use the class with highest probability.
true_Y = data.y[data.test_mask].tolist()
for _ in range(len(true_Y)):
    print(true_Y[_], pred[data.test_mask].tolist()[_])
acc = accuracy_score(true_Y, pred[data.test_mask].tolist())
#acc, f1, prec, rec, auprc, auroc = test(model, data, num_classes = data.num_classes, get_auc = False)
print('Test Accuracy score GCN: {:.4f}'.format(acc))
#print(acc, f1, prec, rec, auprc, auroc)


#explainers on top:
print('Here the Explainer starts!')
print('Different classes to explain: ', data.num_classes)
print('Layers of the GNN: ', num_layers_gnn)

# choose label to explain
node_idx = choose_xnode(data, label_to_explain, pred)
pred = model(data.x, data.edge_index)[node_idx,:].argmax(dim=0)

label = data.y[node_idx].item()
pred_class = pred.clone()
print('We explain the label', label)

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


fig, ax = plt.subplots(1,len(explainers_to_test), figsize = (20, 15))
for index, method in enumerate(explainers_to_test):
    method_exp, forward_kwargs, feedback = get_exp_method(method, model, criterion, pred_class, data, num_layers_gnn, node_idx)
    method_exp.visualize_node(num_hops = num_layers_gnn, additional_hops = 0, graph_data = data, ax = ax[index], norm_imps = False)
    ax[index].set_title(feedback)
plt.show()
folder_path = 'content_fds/plots_explainers'
save_path = os.path.join(folder_path, "plot_explainers" + "_GNNlayers_" + str(num_layers_gnn) + ".pdf")  
plt.show()
plt.savefig(save_path, format = 'pdf')