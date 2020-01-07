import numpy as np
import numpy.random as rand

def DSBM(k, n, p, q, F, random_state=None, Herm=True):
    if not (random_state == None):
        rand.seed(random_state)
    assert (F.shape == k, k) and (k >= 2) and (abs(p-0.5) <= 0.5) and (abs(q-0.5) <= 0.5)
    assert np.all(F + F.T == 1)
    N = k * n
    res = np.zeros((N,N), dtype=complex)
    for c in range(k):
        for d in range(c,k):
            prob = p if c == d else q
            cd_ns = rand.rand(n,n) <= prob
            dirs = 2 * (rand.rand(n,n) <= F[c,d]) - 1
            cd_ns = cd_ns * dirs
            res[c * n : (c+1) * n, d * n : (d+1) * n] = cd_ns
            
    res = (np.triu(res) - np.triu(res).T) * 1j
    
    if not Herm:
        res = 1 * (res/1j > 0)

    # Need to make matrix sparse format to prevent memory errors
    from scipy.sparse import csr_matrix
    res = csr_matrix(res)
    # Need to return both adjacency matrix and the underlying clusters.
    return res, np.reshape(np.arange(N), (k,n))

def DSBM_PA(k, N, P, C, c, a=2, initial_nodes_per_cluster=4, Herm=False, random_state=None):
    from tqdm import tqdm as tqdm
    if not (random_state == None):
        rand.seed(random_state)
    flatten = lambda edges : np.array([e for es in edges for e in es]) if len(edges) > 0 else []
    def select(v, P):
        index = -1
        while(v > 0):
            index +=1
            v -= P[index]
        return max(0, index)
    
    import networkx as nx
    
    communities = list(map(list,np.reshape(np.arange(0,initial_nodes_per_cluster*k), (k,initial_nodes_per_cluster))))
    community_edges = [[]]*k

    G = nx.DiGraph()
    for i, com in enumerate(communities):
        G.add_nodes_from(com, community=i)

    for i, com in enumerate(communities):
        edges = flatten([[(com[j], c) for c in com[j+1:]] for j in range(len(com))])
        G.add_edges_from(edges)
        community_edges[i].extend(edges)

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

    res = nx.to_numpy_array(G, dtype=complex) if Herm else nx.to_numpy_array(G)
    if Herm:
        res = (res - res.T) * 1j

    #need to make matrix sparse to prevent memory errors:
    from scipy.sparse import csr_matrix
    res = csr_matrix(res)
    #Need to return both adjacency matrix and underlying communities.
    return (res, communities)
