# Hands_on_GraphXAI
A hands-on tutorial on explainability of graph neural networks, for the Hands-on Workshop at GAIN in Kassel, 06.-08.09. The code is mostly based on `https://github.com/mims-harvard/GraphXAI`. Slides for this tutorial are `GAIN23.pptx`.

## Overview

This tutorial consists of 3 parts:

1. Explore different explainers on different motif-datasets and compare them to the ground-truth motif

2. Challenge: Given a trained inductive GNN on a motif-dataset, reconstruct the original motif.

3. Experiment: Here you can create simple Class Expressions and calculate fidelity of this CE, as well as the GNN-Output of the corresponding graphs.

## Initialization

1. Use the `requirements.txt` file, to create a virtual python 3.8 environment

2. For using the provided Jupyterhub: you may need to run `chmod +x find_motif.sh` for every shell to obtain permission using them.

## Using the Virtual Machine:
It is possible to use the virtual machine, set up at `http://131.234.28.100/jupyter` for this workshop. 
Username: `upb` Password: `UPBGAIN23`

Follow the following steps, in order that everybody can use the VM:

1. Navigate to the folder `/Documents/GraphXAI/`
2. Copy the folder `Hands_on_GraphXAI` and paste it with an unique name (e.g. count the neighbors to your right and in front of you and add these numbers respectively to the foldername)
3. Open a new terminal and activate the virtual environment `source hot/bin/activate`
4. Change directory to your created folder, e.g. to `cd Documents/GraphXAI/Hands_on_GraphXAI_34`
5. Continue with the steps beneath.

## 1. Exploring Explainers

Run the ` ./run_explainers.sh` file, to see explanations for 1 out of 10 datasets for different explainers.

Settings:

`new_dataset="False"` Set for True, if a new dataset should be created, instead of visualizing one of the 10 pre-created datasets

`datset_of_choice=6` Choose any number between 1 and 10 here to access a pre-trained dataset.

`explainers_to_test="['gnnex', 'gcam', 'subx', 'pgex']"` Choose a list if explainers here, which subgraph-explanations you want to visualize.

You find the visualizations in the `content/plots_explainers` folder. Remark: This deletes itself each time the code is called.


## 2. Challenge: Generate the Input-Data, which maximize the GNN

#### Main Goal: For a pre-trained GNN, create input-data, which maximize the GNN. 
#### Intermediate Goal: The GNN learned to classify nodes, whether they are in a particular motif or not. Find the motif would be a first step towards the main goal.

To start, run `./find_motif.sh`

How to create input-data:

1. Edges: in the list `own_graph`, add edges between node-ids to the list as pairs. E.g., the list `[(0,1), (1,2)]` created a graph with three nodes and two (undirected) edges. Please ensure, that you use node-ids consecutively, and not jump over an integer (e.g. `[(0,1), (1,3)]` would be wrong). Please do not create more than 6 nodes.

2. Node-Features: Each node has 1 feature, which can be set in the list `own_features`. In the motif, nodes have features in `{1,2,3}` (but you can use any features you want). The first element will set the node-feature for the node with id 0, the second element the node-feature for the node with id 1, etc. Please ensure, that you choose a feature for each created node.

Competition: The GNN gives feedback for a node being in the motif as a real number. This feedback is summed up and the person wins, who's graph gives the highest feedback.

## 3. Explore Class Expressions to find graphs of high fidelity for a heterogeneous BAShapes (Houses) Dataset

Caution: Here you need to activate a new virtual environment in the root directory, called `hot2` in the VM, using `source hot2/bin/activate` and again navigate to your copied folder. The install requirements for pip are found in the `requirementshot2.txt` file.

To start, run `run_ce_expl.sh`, which links to the code 'ce_gain.py'.

This code uses a dataset, which is a heterogeneous version of the BAShapes House dataset. Nodetypes are `'1'`, `'2'`, `'3'` for nodes in the house motifs from bottom to top and `0` for other nodes. All edgetypes are `'to'`. 

Here, the aim is to create a class expression, which uses logical AND, ClassExpression, and ObjectProperty. This means, you can create instances of a class and connect these instances by edges (a.k.a. object properties). This is done in a tree-like format as a list `[class, [child1], [child2]]`. The features of all nodes are set to `1`, just like in the original dataset. Keep attention, that you write the entries as strings, i.e. write `'1'` for class 1, instead of `1`.

If you want this to be a competition, try to find a CE which creates a graph which maximizes the fidelity on the dataset.

#### Research Question: Do CEs which lead to graphs with higher GNN-output also have a higher fidelity?
