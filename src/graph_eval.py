import graphmodels.DSBM as DSBM
import graphmodels.DSBM_PA as DSBM_PA
import graphmodels.convert as convert
import graphmodels.hermify as hermify
import itertools
import numpy as np
import scipy.sparse
"""
def run_test(seed,
             N,
	     k,
	     p,
	     q,
	     η):
	n = N // k
	F = DSBM.circle(k, η)
#	C = DSBM_PA.tree(k = k, η = 0.0, inner_edges=0.15)
#	print(F.shape)
	PA_kwargs = convert.DSBM_to_PA(n=n, k=k, p=p, q=q, F=F)
#	PA_kwargs['C'] = C
	
	aris_DSBM_disim = []
	aris_DSBM_herm = []
	
	
	GDSBM, comms_DSBM = DSBM.sample(n=n, k=k, p=p, q=q, F=F, random_state=seed, Herm=False)
	print([len(l) for l in comms_DSBM])
	clustersr = disim.cluster(GDSBM, k, k, mode='R')
	clustersl = disim.cluster(GDSBM, k, k, mode='L')
	print([len(l) for l in clustersr],
	      [len(l) for l in clustersl])
	ari_disimr = evaluate.ari(comms_DSBM, clustersr)
	ari_disiml = evaluate.ari(comms_DSBM, clustersl)
	print('Disim_R', ari_disimr)
	print()
	GDSBM = hermify.to_herm(GDSBM)
	clusters = herm.cluster(GDSBM, k, ϵ=-1)
	ari_herm = evaluate.ari(comms_DSBM, clusters)
	print('Herm', ari_herm)
	print()

	results = dict([])
	results['Herm'] = ari_herm
	results['DiSim_R'] = ari_disimr
	results['DiSim_L'] = ari_disiml
	return results

# circle metagraph
noises = np.array([0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3])
ks =  np.array([3,5,7])
ps =  np.array([ 0.0035, 0.004, 0.0045, 0.005])
seeds = np.array([6, 28, 496, 8128, 33550336])
ns =  np.array([1500, 2000, 2500, 5000])
for (η, k, n, p) in itertools.product(noises, ks, ns, ps):
	q = p
	F = DSBM.circle(k, η)
	print(F.shape)
	PA_kwargs = convert.DSBM_to_PA(n=n, k=k, p=p, q=q, F=F)
	spectra = []
	for seed in seeds:
	#	GPA, comms = DSBM.sample(n=n, k=k, p=p, q=q, F=F, random_state=seed, Herm=False)
		GPA, comms = DSBM_PA.sample(a=10, **PA_kwargs, Herm=False)
		print(GPA.shape)
		out_degrees = GPA.sum(axis=1)
		in_degrees = GPA.sum(axis=0)
		τ = out_degrees.mean()
		Pvals = in_degrees + τ
		_, m = GPA.shape
		Pτ = scipy.sparse.spdiags(Pvals, 0, m, m)
		print('computed P')
		Ovals = out_degrees + τ
		Oτ = scipy.sparse.spdiags(np.reshape(Ovals, (1,-1)), 0, m, m)
		print('computed O')
		tmp1 = Oτ.power(-0.5)
		print('generated tmp1')
		tmp2 = Pτ.power(-0.5)
		print('generated tmp2')
		print(tmp1.shape, GPA.shape, tmp2.shape)
		L =  tmp1 @ GPA @ tmp2
		from scipy.sparse import linalg as la
		U, Σ, V = la.svds(L, k=10)
		Σ.sort()
		spectra.append(Σ[::-1])
	import matplotlib.pyplot as plt
	fig, ax = plt.subplots()
	for Σ in spectra:
		ax.scatter(np.arange(1,11), Σ, s=1.5)
	plt.savefig(f'spectral_gap_k{k}_N{k*n}_noise{η}_p{p}.png')
#	plt.show()
#	input()
"""
def spectral_gap():
	# complete metagraph
	noises = np.array([0.0, 0.1, 0.2])
	ks =  np.array([3,5,7])
	ps =  np.array([ 0.0035, 0.0045, 0.005])
	seeds = np.array([6, 28, 496, 8128])
	ns =  np.array([1500, 2500, 5000])
	for (η, k, n, p) in itertools.product(noises, ks, ns, ps):
		q = p
		F = DSBM.random_complete(random_state=np.random.choice(seeds), k=k, η=η)
		print(F.shape)
		PA_kwargs = convert.DSBM_to_PA(n=n, k=k, p=p, q=q, F=F)
		spectra = []
		for seed in seeds:
		#	GPA, comms = DSBM.sample(n=n, k=k, p=p, q=q, F=F, random_state=seed, Herm=False)
			GPA, comms = DSBM_PA.sample(a=10, **PA_kwargs, Herm=False)
			print(GPA.shape)
			out_degrees = GPA.sum(axis=1)
			in_degrees = GPA.sum(axis=0)
			τ = out_degrees.mean()
			Pvals = in_degrees + τ
			_, m = GPA.shape
			Pτ = scipy.sparse.spdiags(Pvals, 0, m, m)
			print('computed P')
			Ovals = out_degrees + τ
			Oτ = scipy.sparse.spdiags(np.reshape(Ovals, (1,-1)), 0, m, m)
			print('computed O')
			tmp1 = Oτ.power(-0.5)
			print('generated tmp1')
			tmp2 = Pτ.power(-0.5)
			print('generated tmp2')
			print(tmp1.shape, GPA.shape, tmp2.shape)
			L =  tmp1 @ GPA @ tmp2
			from scipy.sparse import linalg as la
			U, Σ, V = la.svds(L, k=10)
			Σ.sort()
			spectra.append(Σ[::-1])
		import matplotlib.pyplot as plt
		fig, ax = plt.subplots()
		for Σ in spectra:
			ax.scatter(np.arange(1,11), Σ, s=1.5)
		plt.savefig(f'complete_meta_spectral_gap_k{k}_N{k*n}_noise{η}_p{p}.png')
	#	plt.show()
	#	input()

def eigenvectors():
	# complete metagraph
	noises = np.array([0.0, 0.1, 0.2])
	ks =  np.array([3,5,7])
	ps =  np.array([0.0045]); np.array([ 0.0035, 0.0045, 0.005])
	seeds = np.array([6]); np.array([6, 28, 496, 8128])
	ns =  np.array([2500]); np.array([1500, 2500, 5000])
	for (η, k, n, p) in itertools.product(noises, ks, ns, ps):
		q = p
		F = DSBM.random_complete(random_state=np.random.choice(seeds), k=k, η=η)
		print(F.shape)
		PA_kwargs = convert.DSBM_to_PA(n=n, k=k, p=p, q=q, F=F)
		spectra = []
		for seed in seeds:
		#	GPA, comms = DSBM.sample(n=n, k=k, p=p, q=q, F=F, random_state=seed, Herm=False)
			GPA, comms = DSBM_PA.sample(a=PA_kwargs['c'], **PA_kwargs, Herm=False)
			print(GPA.shape)
			out_degrees = GPA.sum(axis=1)
			in_degrees = GPA.sum(axis=0)
			τ = out_degrees.mean()
			Pvals = in_degrees + τ
			_, m = GPA.shape
			Pτ = scipy.sparse.spdiags(Pvals, 0, m, m)
			print('computed P')
			Ovals = out_degrees + τ
			Oτ = scipy.sparse.spdiags(np.reshape(Ovals, (1,-1)), 0, m, m)
			print('computed O')
			tmp1 = Oτ.power(-0.5)
			print('generated tmp1')
			tmp2 = Pτ.power(-0.5)
			print('generated tmp2')
			print(tmp1.shape, GPA.shape, tmp2.shape)
			L =  tmp1 @ GPA @ tmp2
			from scipy.sparse import linalg as la
			U, Σ, V = la.svds(L, k=10)
			inds = Σ.argsort()
			spectrum = {
				'L': U[:, inds[-3:]],
				'R': V[:, inds[-3:]],
				'comms': comms}
			print(spectrum['L'].shape)
			spectra.append(spectrum)

		import matplotlib.pyplot as plt
		figP, axesP = plt.subplots(1, 3, figsize=(12, 3), sharey='row')
		figP.suptitle('top left singular vectors')
		spectrum = spectra[np.random.choice(np.arange(len(spectra)))]
		for j in range(3,0, -1):
			num = 0
			for comm in spectrum['comms']:
				print(spectrum['L'][: , -j].shape)
				gL = np.array(spectrum['L'][:, -j]).reshape(-1)
				print(gL[comm].shape, gL[comm])
				axesP[j-1].bar(x = num + np.arange(len(comm)), height=gL[comm])
				num += len(comm)
			
		figP.savefig(f'DSBM_PA_thicker_tail_k{k}_N{k*n}_noise{η}.pdf')

eigenvectors()
