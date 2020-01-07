import numpy as np
import numpy.random as rand

def sample(k, N, P, C, c, a=2, initial_nodes_per_cluster=4, Herm=False, random_state=None):
	from tqdm import tqdm as tqdm
	if not (random_state == None):
		rand.seed(random_state)

	def flatten(nested_list):
		flat_list = []
		for l in nested_list:
			if type(l) == list:
				flat_list = flat_list + l
			else:
				flat_list = flat_list + [l]
	
	def select(v, P):
		index = -1
		while(v > 0):
			index +=1
			v -= P[index]
		return max(0, index)
	
	import networkx as nx
	
	initial_nodes = np.arange(0, k * initial_nodes_per_cluster)
	clusters = np.reshape(initial_nodes, (k, -1))
	communities = clusters.tolist()

	community_edges = [[]]*k
	
	G = nx.DiGraph()
	for i, com in enumerate(communities):
		G.add_nodes_from(com, community=i)
	
	for i, com in enumerate(communities):
		new_edges = [
		            	[(com[j], c) for c in com[j+1:]] for j in range(len(com))]
			    ]
		new_edges = flatten(new_edges)
		G.add_edges_from(new_edges)
		community_edges[i].extend(new_edges)
	
	for i, com in enumerate(communities[:-1]):
		G.add_edge(com[-1], communities[i+1][0])
		community_edges[i].append((com[-1],communities[i+1][0]))
	    
	for i in tqdm(range(initial_nodes_per_cluster*k,N)):
		r = rand.random()
		comm = select(r, P)
		communities[comm].append(i)
		G.add_node(i,community=comm)
		for _ in range(c):
			dest_node = i 
			r2 = rand.random()
			dest = select(r2, C[comm])
			while dest_node == i:
				if rand.random() < c / (c+a):
					edges = community_edges[comm]
					_, dest_node = edges[rand.randint(len(edges))]
				else:
					dest_node = rand.choice(communities[dest])
			G.add_edge(i, dest_node)
			community_edges[comm].append((i,dest_node))   
	
	res = None
	if Herm:
		res = nx.to_numpy_array(G, dtype=complex)
		res = (res - res.T) * 1j
	else:
		res = nx.to_numpy_array(G)
	
	#need to make matrix sparse to prevent memory errors:
	from scipy.sparse import csr_matrix
	res = csr_matrix(res)
	#Need to return both adjacency matrix and underlying communities.
	return (res, communities)
