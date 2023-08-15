#!/bin/sh

new_dataset="False"  #creates a new dataset of the same form and trains a new GNN ontop
label_to_explain="1"  #some label from 0-3; label 4 summarizes all "other" nodes
explainers_to_test="['gnnex', 'gbp']" #all explainers to test: 
layers_of_gnn="2"   # at least 2, at most 4.
    #'gnnex': 'GNNExplainer',    #works
    #'gcam': 'Grad-CAM',     #works
    #'subx': 'SubgraphX',    #works
    #'rand': 'Random',       #does not work, it finds edges not in the computation graph
    #'pgmex': 'PGMExplainer', #works
    #'pgex': 'PGExplainer',    # does not work, it finds edges not in the computation graph
    #'gbp': 'Guided Backprop',  #works
    #'cam': 'CAM',               #works
    #'grad': 'Gradient'           # works

python fds.py "$new_dataset" "$label_to_explain" "$explainers_to_test" "$layers_of_gnn"