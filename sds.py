from torch_geometric.data import Data
import torch
from sds_utils import create_graph_from_pairs, visualize_data, build_dataset, GNN, test_created_dataset, GSAGE_2layer
from graphxai.gnn_models.node_classification import train, test, GCN_3layer_basic

#create a set of pairs
own_graph = list(set([(0,1), (1,2), (2,0)]))
own_data = create_graph_from_pairs(own_graph)
dataset = build_dataset(400, 80)
# visualize created graph
visualize_data(own_data)
path = 'content_sds/models/'
num_layers_gnn = 2
#model = GCN_3layer_basic(hidden_channels = 64, input_feat = len(dataset.y.tolist()), classes = dataset.num_classes)
model = GNN(in_channels=len(dataset.y.tolist()), hidden_channels=128, out_channels=dataset.num_classes, num_layers = num_layers_gnn)
optimizer = torch.optim.Adam(model.parameters(), lr = 0.0015, weight_decay = 0.001)
criterion = torch.nn.CrossEntropyLoss()


new_dataset = True
if new_dataset:
    model.train()
    for _ in range(3000):
        loss = train(model, optimizer, criterion, dataset)
    torch.save(model.state_dict(), path+'model')
else:
    model.load_state_dict(torch.load(path+'model'))

acc = test(model, dataset, num_classes = dataset.num_classes, get_auc = False)[0]
print('Test Accuracy score: {:.4f}'.format(acc))

# Test GNN on own Dataset:
test_created_dataset(own_data, model)




#get some motif dataset and a trained inductive GNN on it




#return the GNN-output on created graph