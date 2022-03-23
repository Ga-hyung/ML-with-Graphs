import numpy as np
import matplotlib.pyplot as plt
from umap import UMAP
import plotly.express as px
import seaborn as sns

layer_2 = np.load('../GCN2.npy')
layer_3 = np.load('../GCN3.npy')
layer_4 = np.load('../GCN4.npy')
layer_5 = np.load('../GCN5.npy')
layer_6 = np.load('../GCN6.npy')
layer_7 = np.load('../GCN7.npy')
labels = np.load('../labels.npy')

palette = {}
for n, y in enumerate(set(labels)):
    palette[y] = f'C{n}'

second = UMAP().fit_transform(layer_2)
third = UMAP().fit_transform(layer_3)
fourth = UMAP().fit_transform(layer_4)
fifth = UMAP().fit_transform(layer_5)
sixth = UMAP().fit_transform(layer_6)
seventh = UMAP().fit_transform(layer_7)

plt.figure(figsize = (10,10))
sns.scatterplot(x = second[:,0], y = second[:,1], hue = labels, palette = palette)
plt.legend(bbox_to_anchor = (1,1), loc = 'upper left')
plt.title('GCN with 2 hidden layers')
plt.savefig('그림/umap_2.png')
plt.show()