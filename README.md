# Hands_on_GraphXAI
A hands-on tutorial on explainability of graph neural networks

## Overview

This tutorial consists of 3 parts:

1. Explore different explainers on different motif-datasets and compare them to the ground-truth motif

2. Challenge: Given a trained GNN on a graph and some explainers, find out, what nodes in the graph are in the classes 0,1,2 (Class 3 is the class of all other nodes).

3. Challenge: Given a trained inductive GNN on a motif-dataset, reconstruct the original motif.

## Initialization

1. Use the `requirements.txt` file, to create a virtual python 3.8 environment


## 1. Challenge

Run the ` ./run_explainers.sh` file, to see explanations for 1 out of 10 datasets for different explainers.

Settings:

`new_dataset="False"` Set for True, if a new dataset should be created, instead of visualizing one of the 10 pre-created datasets

`datset_of_choice=6` Choose any number between 1 and 10 here to access a pre-trained dataset.

`explainers_to_test="['gnnex', 'gcam', 'subx', 'pgex']"` Choose a list if explainers here, which subgraph-explanations you want to visualize.

You find the visualizations in the `content/plots_explainers` folder. Remark: This deletes itself each time the code is called.



## 2. Challenge

Goal: Find the rules, after which the classes 0-3 are made up. Class 4 summarizes all nodes, which don't follow the rules for the other classes.

Open by running the `./find_dataset.sh`. 

Settings:

1. `new_dataset="False"` Set for True, if a new dataset should be created, following the same rules. Then, also a new GNN is trained, which takes a bit of time.

2. `label_to_explain=1` Set to any integer from 0 to 3 to retain explanations for this labels

3. `explainers_to_test="['gnnex', 'gcam', 'subx', 'pgex']"` Choose a list if explainers here, which subgraph-explanations you want to visualize.

You find the visualizations in the `content_fds/plots_explainers` folder. Remark: This deletes itself each time the code is called.
