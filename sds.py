from torch_geometric.data import Data
import torch
from sds_utils import create_graph_from_pairs, visualize_data, build_dataset, GNN, test_created_dataset
from graphxai.gnn_models.node_classification import train, test, GCN_3layer_basic
import ast
import sys

# ------------- global Variables
try: 
    own_graph_list = ast.literal_eval(sys.argv[1])
    own_list_features = ast.literal_eval(sys.argv[2])
except Exception as e:
    print(22, 'Running via Shell has not worked')
    own_graph_list = [(0,1), (1,2), (2,3)]
    own_list_features = [0,1,2,3]

# ------------------ Rest of the code
new_dataset = False
new_model = False
new_model_sage = False
path = 'content_sds/datasets/'

# Check, if at most 5 nodes where created:
if len(own_list_features) > 6:
    print("please don't create more than 5 nodes!")
    sys.exit()

#create a set of pairs
own_graph = list(set(own_graph_list))
own_data = create_graph_from_pairs(own_graph, own_list_features)
if new_dataset:
    dataset = build_dataset(4000, 800)
    torch.save(dataset, path+'sds')
else:
    dataset = torch.load(path+'sds')
    
# visualize created graph
visualize_data(own_data)
path = 'content_sds/models/'
num_layers_gnn = 2
model = GCN_3layer_basic(hidden_channels = 64, input_feat = 2, classes = dataset.num_classes)
model_sage = GNN(in_channels=2, hidden_channels=32, out_channels=dataset.num_classes, num_layers = num_layers_gnn)
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001, weight_decay = 0.001)
criterion = torch.nn.CrossEntropyLoss()



if new_model == True:
    model.train()
    for _ in range(600):
        loss = train(model, optimizer, criterion, dataset)
    torch.save(model.state_dict(), path+'model')
else:
    model.load_state_dict(torch.load(path+'model'))

acc = test(model, dataset, num_classes = dataset.num_classes, get_auc = False)[0]
print('Test Accuracy score of GNN: {:.4f}'.format(acc))


# ------- Graphsage model
'''
if new_model_sage == True:
    model_sage.train()
    for _ in range(600):
        loss = train(model_sage, optimizer, criterion, dataset)
    torch.save(model.state_dict(), path+'model_sage')
else:
    model_sage.load_state_dict(torch.load(path+'model_sage'))

acc = test(model_sage, dataset, num_classes = dataset.num_classes, get_auc = False)[0]
print('Test Accuracy score of GNN: {:.4f}'.format(acc))
'''
# Test GNN on own Dataset:
test_created_dataset(own_data, model)
#print('Now we evaluate the Graph with the graphSAGE model: ' )
#test_created_dataset(own_data, model_sage)   




#get some motif dataset and a trained inductive GNN on it




#return the GNN-output on created graph