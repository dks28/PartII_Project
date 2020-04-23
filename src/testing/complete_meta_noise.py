import numpy as np
import pandas as pd
from ClusteringAlgs import herm, disim
from testing import evaluate
from graphmodels import DSBM, DSBM_PA
from graphmodels import convert, hermify

def run_experiments(seed=0, noise=0, tag="", norm=None):
	print("Running ROCMG-experiments for seed", seed)
	k = 5
	n = 2000 if norm == 'densify' else 4000
	F = DSBM.random_complete(k=k, η=noise, random_state=seed)
	if norm == None:
		for p in [0.002, 0.004, 0.006, 0.008]:
			q = p
			PA_kwargs = convert.DSBM_to_PA(k = k, n=n, p=p, q=q, F=F)
			A, comms = DSBM.sample(n=n, k=k, p=p, q=q, F=F, random_state=seed, Herm=False)
			print("finished sampling dsbm. moving to clustering.")
			cls = disim.cluster(A=A, kz=k, ky=k, norm=norm)
			ari_ds = evaluate.ari(comms, cls)
			result1 = {
				'model': ['DSBM'],
				'p': [p],
				'noise': [noise],
				'algorithm': ['DiSim'],
				'ari': [ari_ds]
			}
			df = pd.DataFrame(data=result1)
			df.to_csv(f'results/{tag}.csv', mode='a', header=False)
			print('finished disim-dsbm, moving on to herm-dsbm')
			A = hermify.to_herm(A)
			cls = herm.cluster(A=A, k=k, norm=norm)
			ari_herm = evaluate.ari(comms, cls)
			result2 = {
				'model': ['DSBM'],
				'p': [p],
				'noise': [noise],
				'algorithm': ['Herm'],
				'ari': [ari_herm]
			}
			df = pd.DataFrame(data=result2)
			df.to_csv(f"results/{tag}.csv", mode='a', header=False)
			print('finished herm-dsbm, movnig on to disim-pa')
			A, comms = DSBM_PA.sample(random_state=seed, a=PA_kwargs['c'], Herm=False, **PA_kwargs)
			cls = disim.cluster(A=A, kz=k, ky=k, norm=norm)
			ari = evaluate.ari(comms, cls)
			result3 = {
				'model': ['DSBM_PA'],
				'p': [p],
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
				'model': ['DSBM_PA'],
				'p': [p],
				'noise': [noise],
				'algorithm': ['Herm'],
				'ari': [ari]
			}
			df = pd.DataFrame(data=result4)
			df.to_csv(f"results/{tag}.csv", mode='a', header=False)			
			print(f"Concluded experiments for p={p}")
	elif norm == 'self-loops':
		for p in [0.003, 0.006]:
			q = p
			PA_kwargs = convert.DSBM_to_PA(k=k, n=n, p=p, q=q, F=F)
			A, comms = DSBM_PA.sample(a = PA_kwargs['c'], random_state=seed, Herm=False, **PA_kwargs)
			AH = hermify.to_herm(A)
			for τ in [0.1, 0.4, 0.75, 0.85, 1, 1.15, 1.25, 1.6, 2, 3]:
				print('clustering using disim')
				cls = disim.cluster(A = A, ky = k, kz = k, norm = norm, τ_self_loops = τ)
				ari = evaluate.ari(comms, cls)
				result1 = {
					'p' : [p],
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
		for p in [0.003, 0.006]:
			q = p
			PA_kwargs = convert.DSBM_to_PA(k=k, n=n, p=p, q=q, F=F)
			A, comms = DSBM_PA.sample(a = PA_kwargs['c'], random_state=seed, Herm=False, **PA_kwargs)
			AH = hermify.to_herm(A)
			for ω in np.linspace(0,0.002,9):
				print('clustering using disim')
				cls = disim.cluster(A = A, ky = k, kz = k, norm = norm, ω=ω)
				ari = evaluate.ari(comms, cls)
				result1 = {
					'p' : [p],
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


	pass
