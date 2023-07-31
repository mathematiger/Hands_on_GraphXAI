import random
from collections import defaultdict

import mlflow
import networkx as nx
import numpy as np
import torch
from torch_geometric.utils import from_networkx
from tqdm import tqdm as tq
from fds_data import create_dataset, GNN, choose_xnode, get_exp_method
from graphxai.gnn_models.node_classification import train, test
import matplotlib.pyplot as plt
import os






path = 'content_fds/datasets/'

new_dataset = True


if new_dataset:
    data = create_dataset(2)
    torch.save(data, path+'fds')
else:
    data = torch.load(path+'fds')
    
#print(data)
#print(data.num_classes)



#add train, validation, test to data
#train_size = int(0.8 * len(data))
#test_size = len(data) - train_size
#train_dataset, test_dataset = torch.utils.data.random_split(data, [train_size, test_size])

#train GNN on data
path = 'content_fds/models/'
num_layers_gnn = 2
model = GNN(in_channels=2, hidden_channels=128, out_channels=data.num_classes, num_layers = num_layers_gnn)
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001, weight_decay = 0.0005)
criterion = torch.nn.CrossEntropyLoss()
if new_dataset:
    for _ in range(1000):
        loss = train(model, optimizer, criterion, data)
    torch.save(model.state_dict(), path+'model')
else:
    model.load_state_dict(torch.load(path+'model'))

acc, f1, prec, rec, auprc, auroc = test(model, data, num_classes = data.num_classes, get_auc = False)
print('Test Accuracy score: {:.4f}'.format(acc))





#explainers on top:
print('Here the Explainer starts!')
print('Different classes to explain: ', data.num_classes)
print('Layers of the GNN: ', num_layers_gnn)

#visualize !! 
node_idx = choose_xnode(data)
print(67, node_idx)
pred = model(data.x, data.edge_index)[node_idx,:].argmax(dim=0)
label = data.y[node_idx].item()
pred_class = pred.clone()
print('We explain the label', label)


#make function such that we can choose a label which we want to explain!
#TODO: Actually make num_layers_gnn the numbers of layers in the GNN





#torch.set_printoptions(threshold=torch.inf)
#print(data.edge_i  ndex)














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
sublist = ['gnnex', 'subx', 'pgex']

fig, ax = plt.subplots(1,len(sublist), figsize = (20, 15))
for index, method in enumerate(sublist):
    method_exp, forward_kwargs, feedback = get_exp_method(method, model, criterion, pred_class, data, num_layers_gnn, node_idx)
    print(105, method_exp, data.edge_index, data)
    method_exp.visualize_node(num_hops = num_layers_gnn, additional_hops = 0, graph_data = data, ax = ax[index], norm_imps = False)
    ax[index].set_title(feedback)
plt.show()
folder_path = 'content_fds/plots_explainers'
save_path = os.path.join(folder_path, "plot_explainers.pdf")  
plt.show()
plt.savefig(save_path, format = 'pdf')



