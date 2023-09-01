#!/bin/sh

new_dataset="False"
delete_old_visualizations="True"
dataset_of_choice="10"    #any number from 1 to 10
explainers_to_test="['gnnex','gcam']" #all explainers to test: 
    #'rand': 'Random', 
    #'gnnex': 'GNNExplainer',   # node and feature mask, trained by Mutual Information
    #'gcam': 'Grad-CAM',        # gradient-based
    #'subx': 'SubgraphX',       # Nodemask     
    #'pgmex': 'PGMExplainer',   # ???
    #'pgex': 'PGExplainer',     # edge mask, trained by Mutual Information
    #'gbp': 'Guided Backprop',  # gradient-based
    #'cam': 'CAM',              # gradient-based
    #'grad': 'Gradient'         # gradient-based

python ex.py "$new_dataset" "$delete_old_visualizations" "$dataset_of_choice" "$explainers_to_test"