#!/bin/sh

new_dataset="False"
datset_of_choice=6
explainers_to_test="['gnnex', 'gcam', 'subx', 'pgex']" #all explainers to test: 
    #'gnnex': 'GNNExplainer',
    #'gcam': 'Grad-CAM',
    #'subx': 'SubgraphX',
    #'rand': 'Random',
    #'pgmex': 'PGMExplainer',
    #'pgex': 'PGExplainer',
    #'gbp': 'Guided Backprop',
    #'cam': 'CAM',
    #'grad': 'Gradient' 

python ex.py "$new_dataset" "$datset_of_choice" "$explainers_to_test"