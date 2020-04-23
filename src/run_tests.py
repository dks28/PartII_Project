import testing.complete_meta_noise  as complete_meta_noise
import testing.cyclic_meta_noise    as cyclic_meta_noise
import testing.cleancycl_meta_noise as cleancycl_meta_noise
import testing.tree_meta_noise      as tree_meta_noise
import testing.intra_cluster_edges  as intra_cluster_edges
import numpy as np
import threading
import sys
import time
import json
import pandas as pd

"""
STEP 1:
Open all results files, we will have one result file for each family of experiments, that is
- noisy communication:
 -- complete metagraph (unnormalised)
    need parameters: p, q, η, model
    need results: ari
 -- cyclic metagraph (unnormalised)
    need parameters: p, q, η, model, k
    need results: algorithm, ari
 -- fixed cyclic metagraph (unnormalised)
    need parameters: c, k, η
    need results: algorithm, ari
 -- tree metagraph (unnormalised)
    need parameters: c, σ, η
    need results: algorithm, ari
- intra-cluster noise development
 -- complete metagraph, k=5 (unregularised)
    need parameters: p,q, model
    need results: algorthm, ari
 -- conduct the whole `interpolation' idea for Herm. fix q=0.5%.
    need parameters: p, model, r
    need result: ari
- self-loop normalisation
 -- complete metagraph for zero noise and some moderate noise
 -- fixed cyclic metagraph for zero noise and some moderate noise
 -- tree metagraph for zero noise and some moderate noise
- densification:
 -- as with self-loops

at this stage, we will open a blank csv file for each of these with the right columns. 
"""
# experiment tags:
nois_comm_comp = "noisy_communication_complete_metagraph"
nois_comm_cycl = "noisy_communication_cyclic_metagraph"
nois_comm_fcyc = "noisy_communication_cleaner_cycle"
nois_comm_tree = "noisy_communication_binary_tree"
intra_edges = "cluster_internal_edge_density"
intra_edges_interpol = "cluster_internal_edges_interpolation"
self_loop_comp = "self_loop_complete"
self_loop_fcyc = "self_loop_cleaner_cycle"
self_loop_tree = "self_loop_tree"
densify_comp = "densify_complete"
densify_fcyc = "densify_cleaner_cycle"
densify_tree = "densify_tree"

# columns to be recorded:
columns = {
	nois_comm_comp:       ['model', 'p', 'noise', 'algorithm', 'ari']
,	nois_comm_cycl:       ['model', 'p', 'k', 'noise', 'algorithm', 'ari']
,	nois_comm_fcyc:       ['c', 'k', 'noise', 'algorithm', 'ari']
,	nois_comm_tree:       ['c', 'sigma', 'noise', 'algorithm', 'ari']
,	intra_edges:          ['model', 'average_edge_probability', 'percentage_intra_edges', 'algorithm', 'ari']
,	intra_edges_interpol: ['model', 'average_edge_probability', 'percentage_intra_edges', 'r', 'ari']
,	self_loop_comp:       ['p', 'noise', 'tau', 'algorithm', 'ari']
,	self_loop_fcyc:       ['c', 'k', 'noise', 'tau', 'algorithm', 'ari']
,	self_loop_tree:       ['c', 'sigma', 'noise', 'tau', 'algorithm', 'ari']
,	densify_comp:         ['p', 'noise', 'omega', 'algorithm', 'ari']
,	densify_fcyc:         ['c', 'k', 'noise', 'omega', 'algorithm', 'ari']
,	densify_tree:         ['c', 'sigma', 'noise', 'omega', 'algorithm', 'ari']
}

experiments = [nois_comm_comp, nois_comm_cycl, nois_comm_fcyc, nois_comm_tree, intra_edges, intra_edges_interpol,self_loop_comp, self_loop_fcyc, self_loop_tree, densify_comp, densify_fcyc, densify_tree]

restart_exps = input("Do you want to overwrite all previous results? ")
while not(restart_exps.lower() in ['y', 'n']):
	restart_exps =input("Do you want to overwrite all previous results? (y/n)")
if restart_exps.lower() == 'y':
	for exp in experiments:
		df = pd.DataFrame(columns=columns[exp])
		df.to_csv(f"results/{exp}.csv")

"""
STEP 2: 
Conduct experiments. Iterate through the parameters for a series of seeds. 
"""

for seed in [6, 278, 42, 1111, 2222,3333]: # can extend later. should wait and see how long this runs before completion
	norm = None # At first, have no regularisation
	# Noisy Communication
	for η in np.linspace(0, 0.35, 5, endpoint=True):
	## Complete Metagraph
		complete_meta_noise.run_experiments(seed=seed, noise=η, tag=nois_comm_comp, norm=norm)
	## Cyclic Metagraph
		cyclic_meta_noise.run_experiments(seed=seed, noise=η, tag=nois_comm_cycl, norm=norm)
	## Cleaner Cycle
		cleancycl_meta_noise.run_experiments(seed=seed, noise=η, tag=nois_comm_fcyc, norm=norm)
	## Tree
		tree_meta_noise.run_experiments(seed=seed, noise=η, tag=nois_comm_tree, norm=norm)
	# Intra_Cluster Edges: η now specifies the percentage of edges remaining in one cluster.
	for η in np.linspace(0,0.9, 10, endpoint=True):
		intra_cluster_edges.run_experiments(seed=seed, noise=η, tag=intra_edges, norm=norm)
	norm = 'interpolation'
	for η in np.linspace(0,0.9, 5, endpoint=True):
		intra_cluster_edges.run_experiments(seed=seed, noise=η, tag=intra_edges_interpol, norm=norm)

	for η in  [0.15]: #[0.0, 0.15]:
		# Self-loops:
		norm = 'self-loops'
		## Complete Metagraph
		complete_meta_noise.run_experiments(seed=seed, noise=η, tag=self_loop_comp, norm=norm)
		## cleaner cycle:
		cleancycl_meta_noise.run_experiments(seed=seed, noise=η, tag=self_loop_fcyc, norm=norm)
		## tree:
		tree_meta_noise.run_experiments(seed=seed, noise=η, tag=self_loop_tree, norm=norm)
		
		# Densification
		norm = 'densify'
		## cleaner cycle:
		cleancycl_meta_noise.run_experiments(seed=seed, noise=η, tag=densify_fcyc, norm=norm)
		## Complete Metagraph
		complete_meta_noise.run_experiments(seed=seed, noise=η, tag=densify_comp, norm=norm)
		## tree:
		tree_meta_noise.run_experiments(seed=seed, noise=η, tag=densify_tree, norm=norm)
	
	print(f"Done with seed={seed}.")

		



print('done')
