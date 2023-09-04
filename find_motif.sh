#!/bin/sh

own_graph="[(0,1),(1,2),(0,2)]"  #create a graph by adding (undirected) edges between node-ids; Use at most 6 different nodes
own_features="[1,2,3]"    # set integer values between 1 and 3, value 0 means not in the motif
python sds.py "$own_graph" "$own_features"