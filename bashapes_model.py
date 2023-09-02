#Model for BA-Shapes
#from BaShapes_Hetero import create_hetero_ba_houses
#import generatingXgraphs
import torch
# from dgl.data.rdf import AIFBDataset, AMDataset, BGSDataset, MUTAGDataset
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
# TODO: save bashapes and use it from saved 
#bashapes = create_hetero_ba_houses(700,120)
#print('Created BAShapes:', bashapes)

#generatingXgraphs.visualize_heterodata(bashapes)



# -----------------------------learn GNN on bashapes

class HeteroGNNBA(torch.nn.Module):
    def __init__(self, metadata, hidden_channels, out_channels, num_layers):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        
        for _ in range(num_layers):
            conv = HeteroConv({
                edge_type: SAGEConv((-1, -1), hidden_channels, dropout = 0.5)
                for edge_type in metadata[1]
            })
            self.convs.append(conv)
        self.lin = Linear(hidden_channels, out_channels)
    def forward(self, x_dict, edge_index_dict):
        for conv in self.convs:
            x_dict = {key: F.leaky_relu(x) 
            for key, x in conv(x_dict, edge_index_dict).items()}
        return self.lin(x_dict['3'])






def train_epoch(model, optimizer, bashapes):
    model.train()
    optimizer.zero_grad()
    out = model(bashapes.x_dict, bashapes.edge_index_dict)
    mask = bashapes['3'].train_mask
    loss = F.cross_entropy(out[mask], bashapes['3'].y[mask])
    loss.backward()
    optimizer.step()
    return float(loss)


@torch.no_grad()
def test(model, bashapes):
    model.eval()
    pred = model(bashapes.x_dict, bashapes.edge_index_dict).argmax(dim=-1) 
    accs = []
    for split in ['train_mask', 'val_mask', 'test_mask']:
        mask = bashapes['3'][split]
        acc = (pred[mask] == bashapes['3'].y[mask]).sum() / mask.size(dim=-1)
# here mask.size not mask.sum(), because the mask is saved as the indices and not as boolean values
        accs.append(float(acc))
    return accs


def train_model(model, optimizer, bashapes):
    model.train()
    for epoch in range(1, 200):
        loss = train_epoch(model, optimizer, bashapes)
        train_acc, val_acc, test_acc = test(model, bashapes)
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train: {train_acc:.4f}, '
              f'Val: {val_acc:.4f}, Test: {test_acc:.4f}')
              

# save model

def train_GNN(retrain, bashapes, layers):
    model = HeteroGNNBA(bashapes.metadata(), hidden_channels=64, out_channels=2,
                  num_layers=layers)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    bashapes, model = bashapes.to(device), model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=0.0005)
    print('started train_GNN')
    #retrain = True
    path_name_saved = "content/models/"+'HeteroBAShapes'
    is_file_there = osp.isfile(path_name_saved) 
    if(is_file_there == True and retrain == False):
        print("using saved model")
        model.load_state_dict(torch.load(path_name_saved))
    else:
        print('training new model')
        train_model(model, optimizer, bashapes)
        PATH = "content/models/" + 'HeteroBAShapes'
        print("File will be saved to: ", PATH)
        torch.save(model.state_dict(), PATH)
    # evaluate accuracy
    model.eval()            
    acc = test(model, bashapes)[2]
    print('Accuracy of GNN on BAShapes', acc)
    return model

#train_GNN(True)