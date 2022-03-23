import networkx as nx
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
import scipy.io
from scipy.sparse import csc_matrix
import random
import json
import pickle

mat = scipy.io.loadmat('POS.mat')
labels = mat['group']
edges = mat['network']


def get_node_data(mat):
    nodedata = {}

    labels = mat['group']
    edges = mat['network']

    rows_labels, cols_labels = labels.nonzero()
    nodelabels = {}
    for r, c in zip(rows_labels, cols_labels): #  r = node number, c = label
        nodelabels.setdefault(str(r), "label=")
        nodelabels[str(r)] += str(c) + "="

    for r, label in nodelabels.items(): # r = node number, c = labels(multi)
        nodedata[str(r)] = (label, {r: 1})

    #     rows_edges, cols_edges = edges.nonzero()
    #     for r, c in zip(rows_edges, cols_edges):
    #         nodedata[str(r)][1][str(c)] = 1
    #         nodedata[str(c)][1][str(r)] = 1
    return nodedata

nodedata = get_node_data(mat)
print(len(nodedata))


edges_data = mat['network']
edges = []
rows_edges, cols_edges = edges_data.nonzero()
for r,c in zip(rows_edges, cols_edges):
    edges.append((str(r).strip(), str(c).strip()))

edges_path = '../graph/wikipedia_edges.txt'

with open(edges_path, 'w') as outfile:
    for edge in edges:
        outfile.write(edge[0]+' '+edge[1]+'\n')

new_label = []
for u, u_data in nodedata.items():
    labels_str = u_data[0]
    labels = labels_str.split('=')
    temp = []
    for i in range(1, len(labels)-1):
        temp.append(int(labels[i]))
    new_label.append(temp)
print(len(new_label))

with open('node_label.pickle', 'wb') as f:
    pickle.dump(new_label, f, pickle.HIGHEST_PROTOCOL)



