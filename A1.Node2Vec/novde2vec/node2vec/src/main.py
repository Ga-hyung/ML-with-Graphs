'''
Reference implementation of node2vec. 

Author: Aditya Grover

For more details, refer to the paper:
node2vec: Scalable Feature Learning for Networks
Aditya Grover and Jure Leskovec 
Knowledge Discovery and Data Mining (KDD), 2016
'''

import argparse
import numpy as np
import networkx as nx
import node2vec
from gensim.models import Word2Vec
from sklearn.exceptions import UndefinedMetricWarning
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

def parse_args():
	'''
	Parses the node2vec arguments.
	'''
	parser = argparse.ArgumentParser(description="Run node2vec.")

	parser.add_argument('--input', nargs='?', default='../graph/wikipedia_edges.txt',
	                    help='Input graph path')

	parser.add_argument('--output', nargs='?', default='embedding/500_word2vec_embedding',
	                    help='Embeddings path')

	parser.add_argument('--dimensions', type=int, default=128,
	                    help='Number of dimensions. Default is 128.')

	parser.add_argument('--walk-length', type=int, default=500,
	                    help='Length of walk per source. Default is 80.')

	parser.add_argument('--num-walks', type=int, default=10,
	                    help='Number of walks per source. Default is 10.')

	parser.add_argument('--window-size', type=int, default=10,
                    	help='Context size for optimization. Default is 10.')

	parser.add_argument('--iter', default=1, type=int,
                      help='Number of epochs in SGD')

	parser.add_argument('--workers', type=int, default=8,
	                    help='Number of parallel workers. Default is 8.')

	parser.add_argument('--p', type=float, default=1,
	                    help='Return hyperparameter. Default is 1.')

	parser.add_argument('--q', type=float, default=1,
	                    help='Inout hyperparameter. Default is 1.')

	parser.add_argument('--weighted', dest='weighted', action='store_true',
	                    help='Boolean specifying (un)weighted. Default is unweighted.')
	parser.add_argument('--unweighted', dest='unweighted', action='store_false')
	parser.set_defaults(weighted=False)

	parser.add_argument('--directed', dest='directed', action='store_true',
	                    help='Graph is (un)directed. Default is undirected.')
	parser.add_argument('--undirected', dest='undirected', action='store_false')
	parser.set_defaults(directed=False)

	return parser.parse_args()

def read_graph():
	'''
	Reads the input network in networkx.
	'''
	if args.weighted:
		G = nx.read_edgelist(args.input, nodetype=int, data=(('weight',float),), create_using=nx.DiGraph())
	else:
		G = nx.read_edgelist(args.input, nodetype=int, create_using=nx.DiGraph())
		for edge in G.edges():
			G[edge[0]][edge[1]]['weight'] = 1

	if not args.directed:
		G = G.to_undirected()

	return G

def learn_embeddings(walks, output):
	'''
	Learn embeddings by optimizing the Skipgram objective using SGD.
	'''
	walks = [[str(n) for n in walk] for walk in walks]
	model = Word2Vec(walks, vector_size=args.dimensions, window=args.window_size, min_count=0, sg=1, workers=args.workers, epochs=args.iter)
	node_embeddings = (model.wv.vectors)
	#model.save(f'embedding/{args.num_walks}_word2vec.model')
	np.save(output, node_embeddings, allow_pickle = False)
	return

def main(args):
	'''
	Pipeline for representational learning for all nodes in a graph.
	'''
	nx_G = read_graph()
	G = node2vec.Graph(nx_G, args.directed, args.p, args.q)
	G.preprocess_transition_probs()
	num_length = [900, 1000, 1100, 1200, 1300, 1400]
	for i in num_length:
		print(i)
		walks = G.simulate_walks(args.num_walks, i)
		output =f'embedding/{i}_word2vec_embedding'
		learn_embeddings(walks, output)
	# node_embeddings = (
	# 	model.wv.vectors
	# )
	# with open('../node_label.pickle', 'rb') as f:
	# 	node_label = pickle.load(f)
	#
	# x = node_embeddings
	# y = np.array(node_label, dtype=list)
	# mlb = MultiLabelBinarizer()
	# mlb.fit(y)
	# y = mlb.transform(y)
	# x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1)
	# clf = MLPClassifier(random_state=1, max_iter=1000).fit(x_train, y_train)
	# y_test_hat = clf.predict(x_test)
	# accuracy = accuracy_score(y_test, y_test_hat)
	# f1_score_mi = f1_score(y_test, y_test_hat, average='micro')
	# f1_score_ma = f1_score(y_test, y_test_hat, average='macro')
	# print("Accuracy = ",accuracy*100)
	# print("F1_score(micro) = ",f1_score_mi*100)
	# print("F1_score(macro) = ", f1_score_ma*100)

if __name__ == "__main__":
	args = parse_args()
	main(args)
