import numpy as np
import numpy.random as rand

def sample(k, N, P, C, c, a=2, initial_nodes_per_cluster=4, Herm=False, random_state=None, **kwargs):
	print(f"c={c}, N={N}")
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
		return flat_list
	
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
		            	[(com[j], c) for c in com[j+1:]] for j in range(len(com))
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
#			print(C.shape, P.shape, dest)
			dbar = c * (C[:, dest] * P).sum() / P[dest]
			while dest_node == i:
				if rand.random() < dbar / (dbar + a):
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

def hard_cycle(k, η):
	C = np.zeros((k,k))
	for i in range(k):
		C[i, (i+1) % k] = 1
	noise = np.ones((k,k)) * η / (k-1)
	noise -= C * η / (k-1)
	C = C * (1 - η) + noise
	return C

def tree(h, η, σ):
	k = 2**h - 1 
	C = np.zeros((k,k))
	for i in range(1, k, 2):
		C[i, i+1] = σ
		C[i, i//2] = 1-σ
		C[i+1, i//2] = 1
	C[0,0] = 1
	nonzeros = (C > 0).astype(float)
	noise = (1 - nonzeros)
	for i in range(1, k, 2):
		noise[i , :] = noise[i, :] * η / (k-2)
		noise[i+1, :] = noise[i+1, :] * η / (k-1)
	noise[0, :] = noise[0, :] * η / (k-1)
	C = C * (1-η) + noise
	return C
