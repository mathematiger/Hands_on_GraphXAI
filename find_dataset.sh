#!/bin/sh

new_dataset="True"  #creates a new dataset of the same form and trains a new GNN ontop
label_to_explain="1"  #some label from 0-3; label 4 summarizes all "other" nodes
explainers_to_test="['gnnex', 'gcam', 'subx']" #all explainers to test: 
layers_of_gnn="3"   # at least 2, at most 4.
    #'gnnex': 'GNNExplainer',
    #'gcam': 'Grad-CAM',
    #'subx': 'SubgraphX',    #can take a very long time
    #'rand': 'Random',
    #'pgmex': 'PGMExplainer',
    #'pgex': 'PGExplainer',    # may not work
    #'gbp': 'Guided Backprop',
    #'cam': 'CAM',
    #'grad': 'Gradient' 

python fds.py "$new_dataset" "$label_to_explain" "$explainers_to_test" "$layers_of_gnn"