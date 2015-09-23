from core import *
from lowrank import *
import numpy as np
from sktensor import dtensor
from cp3 import CP

__all__ = ['Tucker', 'hosvd', 'hoid', 'tensor_to_id', 'plotindlst']

class Tucker:
	def __init__(self, eps = 1.e-7, G = None, U = None, I = None):

		self.G = G 
		self.U = U
		self.I = I
		
		if (G is not None) and (U is not None):
			self.rank  = G.shape 
			self.shape = tuple([u.shape[0] for u in U])
			self.ndim  = len(self.shape)
	
		else:
			self.ndim = 3
			self.rank = (0,0,0)
			self.shape = np.array([0,0,0])

	def lowrank_matricize(self):

		G, U = self.G, self.U
		
		ndim = self.ndim	
	
		ulst, vlst  = [], []
		modes = matricize(dtensor(G))
		for d, mode in zip(range(ndim),modes):
			lst = [U[i] for i in (range(d) + range(d+1, ndim))] 
			v = kron(lst, reverse = True).T
			ulst.append(np.dot(U[d], mode))
			vlst.append(v)

		return ulst, vlst	


	def matricize(self):	
		"""Use with caution"""

		ulst, vlst = self.lowrank_matricize()
		modes = [np.dot(u,v) for u,v in zip(ulst,vlst)]
	
		return modes

	
	def __rmul__(self, const): # only scalar by tensor product!
        	mult = copy.copy(self)
        	mult.G = const * self.G
        	return mult

    	def __neg__(self):
        	neg = copy.copy(self)
        	neg.G = (-1.) * neg.G
        	return neg



def hosvd(T, rank = None, eps = None, method = 'svd', compute_core = True):
	modes = matricize(T)
	dim = T.ndim
	
	assert (rank is not None) or (eps is not None)
	if eps is not None: eps_or_k = eps/float(dim)


	ulst = []
	for d, mode in zip(range(dim),modes):
		if rank is not None:	eps_or_k = rank[d]
		if method == 'svd':
			u, _, _ = svdtruncate(mode, eps_or_k)	
		elif method == 'randsvd':
			u, _, _ = randsvd(mode, eps_or_k)
		else:
			raise NotImplementedError
		ulst.append(u)

	#Compute core tensor
	if compute_core:
		G = T.ttm(ulst, transp = True)
		return Tucker(G = G, U = ulst)
	else:
		return ulst

#def hooi(T, rank, maxiter):
	#for k in range(maxiter):


def hoid(T, rank = None, eps = None, compute_core = True):
	modes = matricize(T)
	dim   = T.ndim

	assert (rank is not None) or (eps is not None)
	if eps is not None: eps_or_k = eps/float(dim)
	
	clst, indlst = [], []
	for d, mode in zip(range(dim), modes):
		if rank is not None:	eps_or_k = rank[d]
		p, _ = id_(mode, eps_or_k)
		c = mode[:, p]
		clst.append(c)
		indlst.append(p)

	#Compute core tensor
	if compute_core:
		cinv = [np.linalg.pinv(c) for c in clst]
		G = T.ttm(cinv)
		return Tucker(G = G, U = clst, I = indlst)
	else:
		return clst



def tensor_to_id(full, T, rank, method = 'deim', compute_core = True):

	dim = T.ndim
 	
	if not (isinstance(T, Tucker) or isinstance(T, CP)):
		raise NotImplementedError, "Only implemented for Tucker and CP"

	#Unfold
	modes = matricize(full)	
	alst, blst = T.lowrank_matricize()

	clst, indlst = [], []

	modeerr = []
	for a, b, mode, r  in zip(alst, blst, modes, rank):
		_, _, vh = lowrank_to_svd(a, b.conj().T)

		if method == 'deim':
			q, fact = deim(vh[:,:r])
		elif method == 'dime':
			q, fact = dime(vh[:,:r])
		else:
			raise NotImplementedError

		modeerr.append(fact)

		c = mode[:,q]


		clst.append(c)
		indlst.append(q)

	#Compute core tensor
	if compute_core:
		cinv = [np.linalg.pinv(c) for c in clst]
		G = full.ttm(cinv)
		return Tucker(G = G, U = clst, I = indlst), modeerr
	else:
		return clst, modeerr


def plotindlst(T,indlst):
	ndim = T.ndim
	modes = matricize(T)

	import  matplotlib.pyplot as plt
	f, axarray = plt.subplots(nrows = ndim, sharex = True)

	dim = 0
	for ax, mode, ind in zip(axarray,modes,indlst):
		I = np.zeros_like(mode)
		I[:,np.array(ind)] = 2.
		im = ax.pcolormesh(I)
		im.set_cmap('binary')

		dim += 1
		ax.set_title('Mode number %i'%(dim))
	
	return

if __name__ == '__main__':
	A = np.random.randn(5,5,5)
	
	rank = A.shape 

	A = dtensor(A)
	T = hosvd(A, rank = rank, method = 'svd')
	Tmodes = T.matricize()
	Amodes = matricize(A)

	isclose = [np.allclose(a,t) for a, t in zip(Amodes,Tmodes)]
	print isclose


	T = hoid(A, rank = rank)
	Tmodes = T.matricize()
	Amodes = matricize(A)

	isclose = [np.allclose(a,t) for a, t in zip(Amodes,Tmodes)]
	print isclose


	Tf, _ = tensor_to_id(A, T, method = 'dime')
	Tmodes = Tf.matricize()
	Amodes = matricize(A)

	isclose = [np.allclose(a,t) for a, t in zip(Amodes,Tmodes)]
	print isclose
