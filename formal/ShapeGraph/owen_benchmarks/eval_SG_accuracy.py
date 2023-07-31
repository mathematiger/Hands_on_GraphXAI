import tqdm
import ipdb
import argparse, sys; sys.path.append('../..')
import random as rand
import torch
from metrics import *
from graphxai.explainers import *
from graphxai.datasets  import load_ShapeGGen
from graphxai.datasets.shape_graph import ShapeGGen
from graphxai.utils.performance.load_exp import exp_exists
from graphxai.gnn_models.node_classification.testing import GIN_3layer_basic, GCN_3layer_basic, GSAGE_3layer

my_base_graphxai = '/home/owq978/GraphXAI'
#my_base_graphxai = '/Users/owenqueen/Desktop/HMS_research/graphxai_project/GraphXAI'

def get_exp_method(method, model, criterion, bah, node_idx, pred_class):
    method = method.lower()
    if method=='gnnex':
        exp_method = GNNExplainer(model)
        forward_kwargs={'x': data.x.to(device),
                        'node_idx': int(node_idx),
                        'edge_index': data.edge_index.to(device)}
    elif method=='grad':
        exp_method = GradExplainer(model, criterion = criterion)
        forward_kwargs={'x': data.x.to(device),
                        'y': data.y.to(device),
                        'node_idx': int(node_idx),
                        'edge_index': data.edge_index.to(device)}
    elif method=='cam':
        exp_method = CAM(model, activation = lambda x: torch.argmax(x, dim=1))
        forward_kwargs={'x': data.x.to(device),
                        'y': data.y.to(device),
                        'node_idx': int(node_idx),
                        'edge_index': data.edge_index.to(device)}
    elif method=='gcam':
        exp_method = GradCAM(model, criterion = criterion)
        forward_kwargs={'x':data.x.to(device),
                        'y': data.y.to(device),
                        'node_idx': int(node_idx),
                        'edge_index': data.edge_index.to(device),
                        'average_variant': [True]}
    elif method=='gbp':
        exp_method = GuidedBP(model, criterion = criterion)
        forward_kwargs={'x': data.x.to(device),
                        'y': data.y.to(device),
                        'node_idx': int(node_idx),
                        'edge_index': data.edge_index.to(device)}
    elif method=='glime':
        exp_method = GraphLIME(model)
        forward_kwargs={'x': data.x.to(device),
                        'node_idx': int(node_idx),
                        'edge_index': data.edge_index.to(device)}
    elif method=='ig':
        exp_method = IntegratedGradExplainer(model, criterion = criterion)
        forward_kwargs = {'x': data.x.to(device),
                        'edge_index': data.edge_index.to(device),
                        'node_idx': int(node_idx),
                        'label': pred_class}
    elif method=='glrp':
        exp_method = GNN_LRP(model)
        forward_kwargs={'x': data.x.to(device),
                        'edge_index': data.edge_index.to(device),
                        'node_idx': node_idx,
                        'label': pred_class,
                        'edge_aggregator':torch.sum}
    elif method=='pgmex':
        exp_method=PGMExplainer(model, explain_graph=False, p_threshold=0.1)
        forward_kwargs={'node_idx': node_idx,
                        'x': data.x.to(device),
                        'edge_index': data.edge_index.to(device),
                        'top_k_nodes': 10}
    elif method=='pgex':
        exp_method=PGEX
        forward_kwargs={'node_idx': node_idx,
                        'x': data.x.to(device),
                        'edge_index': data.edge_index.to(device),
                        'label': pred_class}
    elif method=='rand':
        exp_method = RandomExplainer(model)
        forward_kwargs={'x': data.x.to(device),
                        'node_idx': int(node_idx),
                        'edge_index': data.edge_index.to(device)}
    elif method=='subx':
        exp_method = SubgraphX(model, reward_method = 'gnn_score', num_hops = bah.model_layers, rollout=5)
        forward_kwargs={'node_idx': node_idx,
                        'x': data.x.to(device),
                        'edge_index': data.edge_index.to(device),
                        'label': pred_class,
                        'max_nodes': 10}
    else:
        OSError('Invalid argument!!')
    return exp_method, forward_kwargs

def get_model(name):
    if name.lower() == 'gcn':
        model = GCN_3layer_basic(16, input_feat = 11, classes = 2)
    elif name.lower() == 'gin':
        model = GIN_3layer_basic(16, input_feat = 11, classes = 2)
    elif name.lower() == 'sage':
        # Get SAGE model
        model = GSAGE_3layer(16, input_feat = 11, classes = 2)
    else:
        OSError('Invalid model!')
    return model

parser = argparse.ArgumentParser()
parser.add_argument('--exp_method', required=True, help='name of the explanation method')
parser.add_argument('--model', required=True, help = 'Name of model to train (GIN, GCN, or SAGE)')
#parser.add_argument('--model_path', required=True, help = 'Location of pre-trained weights for the model')
parser.add_argument('--save_dir', default='./results/', help='folder for saving results')
parser.add_argument('--ignore_training', action='store_true', help='Ignores model training for PGEX')
parser.add_argument('--ignore_cache', action='store_true')
args = parser.parse_args()

seed_value=912
rand.seed(seed_value)
np.random.seed(seed_value)
torch.manual_seed(seed_value)

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load ShapeGGen dataset
# Smaller graph is shown to work well with model accuracy, graph properties
bah = torch.load(open(os.path.join(my_base_graphxai, 'data/ShapeGGen/unzipped/SG_homophilic.pickle'), 'rb'))

data = bah.get_graph(use_fixed_split=True)

#inhouse = (data.y[data.test_mask] == 1).nonzero(as_tuple=True)[0]
# test_set = (data.test_mask).nonzero(as_tuple=True)[0]
# np.random.shuffle(test_set.numpy())
# print(test_set)
test_set = torch.load(os.path.join(my_base_graphxai, 'formal/ShapeGGen', 'test_inds_SG_homophilic.pt'))

# Test on 3-layer basic GCN, 16 hidden dim:
model = get_model(name = args.model).to(device)

# Get prediction of a node in the 2-house class:
mpath = os.path.join(my_base_graphxai, 'formal/model_weights/model_homophily.pth')
model.load_state_dict(torch.load(mpath))

# Pre-train PGEX before running:
if args.exp_method.lower() == 'pgex':
    if not args.ignore_training:
        PGEX=PGExplainer(model, emb_layer_name = 'gin3' if isinstance(model, GIN_3layer_basic) else 'gcn3', max_epochs=10, lr=0.1)
        PGEX.train_explanation_model(data.to(device))
    else:
        # Assumes we already have all the explanations
        PGEX = None

gea_feat = []
gea_node = []
gea_edge = []

# Get predictions
pred = model(data.x.to(device), data.edge_index.to(device))

criterion = torch.nn.CrossEntropyLoss().to(device)

# Get delta for the model:
delta = np.load(os.path.join(my_base_graphxai, 'formal', 'model_weights', 'model_homophily_delta.npy'))[0]

# Cached graphs:
G = to_networkx_conv(data, to_undirected=True)

#save_exp_flag = args.exp_method.lower() in ['gnnex', 'pgex', 'pgmex', 'subx']
save_exp_flag = True
save_exp_dir = os.path.join(my_base_graphxai, 'formal/ShapeGGen', 'bigSG_explanations', args.exp_method.upper())

#for node_idx in tqdm.tqdm(inhouse[:1000]):
for node_idx in tqdm.tqdm(test_set):

    node_idx = node_idx.item()

    # Get predictions
    pred_class = pred[node_idx, :].reshape(-1, 1).argmax(dim=0)

    if pred_class != data.y[node_idx]:
        # Don't evaluate if the prediction is incorrect
        continue

    # Get explanation method
    explainer, forward_kwargs = get_exp_method(args.exp_method, model, criterion, bah, node_idx, pred_class)

    # Get explanations
    exp = exp_exists(node_idx, path = save_exp_dir, get_exp = True) # Retrieve the explanation, if it's there
    #exp = None
    #print(exp)

    if (exp is None) or args.exp_method.lower() == 'pgex'\
            or args.ignore_cache:
        exp = explainer.get_explanation_node(**forward_kwargs)

        if save_exp_flag and (args.exp_method.lower() != 'pgex'):
            # Only saving, no loading here
            torch.save(exp, open(os.path.join(save_exp_dir, 'exp_node{:0<5d}.pt'.format(node_idx)), 'wb'))

    # Calculate metrics
    #feat, node, edge = graph_exp_faith(exp, bah, model, sens_idx=[bah.sensitive_feature])
    feat, node, edge = graph_exp_acc(gt_exp = bah.explanations[node_idx], generated_exp=exp)
    gea_feat.append(feat)
    gea_node.append(node)
    gea_edge.append(edge)


############################
# Saving the metric values
# save_dir='./results_homophily/'
np.save(os.path.join(args.save_dir, f'{args.exp_method}_GEA_feat.npy'), gea_feat)
np.save(os.path.join(args.save_dir, f'{args.exp_method}_GEA_node.npy'), gea_node)
np.save(os.path.join(args.save_dir, f'{args.exp_method}_GEA_edge.npy'), gea_edge)
