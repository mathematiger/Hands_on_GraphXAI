#!/bin/sh

own_graph="[(0,1),(1,2),(0,2), (0,3), (0,4), (2,3), (3,4), (4,0)]"  #create a graph by adding (undirected) edges between node-ids; Use at most 6 different nodes
own_features="[3,2,2,1,1]"    # set integer values between 1 and 3, value 0 means not in the motif
python sds.py "$own_graph" "$own_features"