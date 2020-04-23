import numpy as np
import pandas as pd
from ClusteringAlgs import herm, disim
from testing import evaluate
from graphmodels import DSBM, DSBM_PA
from graphmodels import convert, hermify
from itertools import product

def run_experiments(seed=0, noise=0, tag="", norm=None):
	print("Running ROCMG-experiments for seed", seed)
	n = 2000 if norm=='densify' else 4000
	if norm == None:
		for k, c in product([3,5], [20, 40, 60]):
			N = n*k
			C = DSBM_PA.hard_cycle(k=k, η=noise)
			P = np.ones(k)/k
			A, comms = DSBM_PA.sample(k=k, c=c, C=C, P=P, N=N, random_state=seed, a=c, Herm=False)
			cls = disim.cluster(A=A, kz=k, ky=k, norm=norm)
			ari = evaluate.ari(comms, cls)
			result3 = {
				'c': [c],
				'k': [k],
				'noise': [noise],
				'algorithm': ['DiSim'],
				'ari': [ari]
			}
			print("finished disim-pa, moving on to herm-pa")
			df = pd.DataFrame(data=result3)
			df.to_csv(f"results/{tag}.csv", mode='a', header=False)
			A = hermify.to_herm(A)
			cls = herm.cluster(A=A, k=k, norm=norm)
			ari = evaluate.ari(comms, cls)
			result4 = {
				'c': [c],
				'k': [k],
				'noise': [noise],
				'algorithm': ['Herm'],
				'ari': [ari]
			}
			df = pd.DataFrame(data=result4)
			df.to_csv(f"results/{tag}.csv", mode='a', header=False)			
			print(f"Concluded experiments for c={c}")
	elif norm == 'self-loops':
		for k, c in product([3,5], [20, 60]):
			N = n*k
			C = DSBM_PA.hard_cycle(k=k, η=noise)
			P = np.ones(k)/k
			A, comms = DSBM_PA.sample(k=k, c=c, C=C, P=P, N=N, random_state=seed, a=c, Herm=False)
			AH = hermify.to_herm(A)
			for τ in [0.1, 0.4, 0.75, 0.85, 1, 1.15, 1.25, 1.6, 2, 3]:
				print('clustering using disim')
				cls = disim.cluster(A = A, ky = k, kz = k, norm = norm, τ_self_loops = τ)
				ari = evaluate.ari(comms, cls)
				result1 = {
					'c' : [c],
					'k': [k],
					'noise': [noise],
					'tau': [τ],
					'algorithm': ['DiSim'],
					'ari': [ari]
				}
				df = pd.DataFrame(data=result1)
				df.to_csv(f"results/{tag}.csv", mode='a', header=False)
				print('finished with DiSim, moving to Herm')
				cls = herm.cluster(A = AH, k = k, norm = norm, τ_self_loops = τ)
				ari = evaluate.ari(comms, cls)
				result1['algorithm'] = ['Herm']
				result1['ari'] = [ari]
				df = pd.DataFrame(data=result1)
				df.to_csv(f"results/{tag}.csv", mode='a', header=False)
				print('done with herm')
	elif norm == 'densify':
		for k, c in product([5], [20, 60]):
			N = n*k
			C = DSBM_PA.hard_cycle(k=k, η=noise)
			P = np.ones(k)/k
			A, comms = DSBM_PA.sample(k=k, c=c, C=C, P=P, N=N, random_state=seed, a=c, Herm=False)
			AH = hermify.to_herm(A)
			for ω in np.linspace(0,0.002,9):
				print('clustering using disim')
				cls = disim.cluster(A = A, ky = k, kz = k, norm = norm, ω=ω)
				ari = evaluate.ari(comms, cls)
				result1 = {
					'c' : [c],
					'k': [k],
					'noise': [noise],
					'omega': [ω],
					'algorithm': ['DiSim'],
					'ari': [ari]
				}
				df = pd.DataFrame(data=result1)
				df.to_csv(f"results/{tag}.csv", mode='a', header=False)
				print('finished with DiSim, moving to Herm')
				cls = herm.cluster(A = AH, k = k, norm = norm, ω = ω)
				ari = evaluate.ari(comms, cls)
				result1['algorithm'] = ['Herm']
				result1['ari'] = [ari]
				df = pd.DataFrame(data=result1)
				df.to_csv(f"results/{tag}.csv", mode='a', header=False)
				print('done with herm')
