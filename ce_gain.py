from owlapy.model import OWLObjectProperty, OWLObjectSomeValuesFrom
from owlapy.model import OWLDataProperty
from owlapy.model import OWLClass, OWLClassExpression
from owlapy.model import OWLDeclarationAxiom, OWLDatatype, OWLDataSomeValuesFrom, OWLObjectIntersectionOf
from owlapy.render import DLSyntaxObjectRenderer
from owlapy.model import OWLNamedIndividual, IRI

import pickle
import os
import sys
import ast

import bashapes_model as bsm
from datasets import create_hetero_ba_houses
from evaluation import ce_fidelity
from visualization import visualize_heterodata
from ce_generation import create_graphdict_from_ce
from graph_generation import add_feat_one_to_dict

dlsr = DLSyntaxObjectRenderer()
xmlns = "http://www.semanticweb.org/stefan/ontologies/2023/1/untitled-ontology-11#"
NS = xmlns
objprop = OWLObjectProperty(IRI(NS, 'to'))



# ------------- input variables
input_ce = ['1', ['11', ['111']], ['2'], ['3']]
new_dataset = False #not chooseable

try:
    input_ce = ast.literal_eval(sys.argv[1]) 
    print('Running code with variables from shell')
except Exception as e:
    print('Importing variables from Shell has not worked')
    input_ce = ['1', ['0', ['0']], ['2'], ['3']]





# ----------- utils
def delete_files_in_folder(folder_path):
    """Deletes all files in the specified folder."""
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(f"Error deleting file: {e}")

def create_ce_from_inputtree(tree):
    return_ce = OWLObjectIntersectionOf([])
    tce = tuple(return_ce._operands)
    lce = list(tce)
    current_class = OWLClass(IRI(NS, tree[0]))
    lce.append(current_class)
    return_ce._operands = tuple(lce)
    del tree[0]
    for i in tree: 
        child = OWLObjectIntersectionOf([])
        child = create_ce_from_inputtree(i)
        edge = OWLObjectSomeValuesFrom(property = objprop, filler = child)
        tcec = tuple(return_ce._operands)
        lcec = list(tcec)
        lcec.append(edge)
        return_ce._operands = tuple(lcec)
    return return_ce

#import BA-Hetero-dataset + model

if new_dataset == True:
    bashapes = create_hetero_ba_houses(500,100) # TODO: Save BAShapes to some file, code already somewhere
    with open('content_hetero/data_hetero/bashapes.pkl', 'wb') as f:
        pickle.dump(bashapes, f)
else:
    with open('content_hetero/data_hetero/bashapes.pkl', 'rb') as f:
        bashapes = pickle.load(f)


delete_files_in_folder('content_hetero/plots_hetero')
if new_dataset == True:
    modelHeteroBSM = bsm.train_GNN(True, bashapes, layers=4)
    with open('content_hetero/data_hetero/bashapes_model.pkl', 'wb') as f:
        pickle.dump(modelHeteroBSM, f)
else:
    with open('content_hetero/data_hetero/bashapes_model.pkl', 'rb') as f:
        modelHeteroBSM = pickle.load(f)
ce_to_test = create_ce_from_inputtree(input_ce)

#calculate fidelity of this CE
fidelity = ce_fidelity(ce_to_test, modelHeteroBSM, bashapes, '3')




#visualize graphs for this CE
def make_data_for_visualization(ce, origdata):
    node_types = origdata.node_types
    metagraph = origdata.edge_types #[str,str,str]
    edge_names = []
    list_classes_objprops = []
    for mp in metagraph:
        if mp[1] not in edge_names:
            edge_names.append(mp[1])
            list_classes_objprops.append([OWLClass(IRI(NS, mp[0])), [OWLObjectProperty(IRI(NS, mp[1]))]])
            list_classes_objprops.append([OWLClass(IRI(NS, mp[2])), [OWLObjectProperty(IRI(NS, mp[1]))]])
    graph_dict = create_graphdict_from_ce(ce_to_test, node_types, edge_names, metagraph)
    hd = add_feat_one_to_dict(graph_dict)
    return hd



hd = make_data_for_visualization(ce_to_test, bashapes)
modelHeteroBSM.eval()
pred = modelHeteroBSM(hd.x_dict, hd.edge_index_dict)
gnn_out_on_hd = round(pred[-1][1].item(), 2)
visualize_heterodata(hd, ce = dlsr.render(ce_to_test), gnnout = gnn_out_on_hd, list_all_nodetypes = hd.node_types, add_out = 'fidelity: '+str(fidelity), name_folder = 'content_hetero/plots_hetero')

print('Evaluated the Class Expression', dlsr.render(ce_to_test))
print('Graph has GNN-out of ', gnn_out_on_hd)
print('CE has a Fidelity of ', fidelity)
# call with hd, addname = '', ce = None, gnnout = None, mean_acc = None, add_out = None, list_all_nodetypes = None, label_to_explain = None, name_folder = ''
#visualize_heterodata(hd, ce = dlsr.render(ce_to_test), gnnout = gnn_out_on_hd, list_all_nodetypes = hd.node_types, add_out = 'fidelity: '+str(fidelity), name_folder = 'content_hetero/plots_hetero')























# ----------------- testing functions
testtree = ['1', ['11', ['111']], ['2'], ['3']]
#print(42, dlsr.render(create_ce_from_inputtree(testtree)))