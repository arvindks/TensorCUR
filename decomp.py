from core import *
from lowrank import *
from rrqr import *
import numpy as np
from numpy import dot
from numpy.linalg import norm, inv, pinv
from sktensor import dtensor, khatrirao, ktensor
from scipy.linalg.interpolative import interp_decomp

__all__ = ['CP',\
	'Tucker',\
	'hosvd', \
	'sthosvd',\
	'hoid',\
	'sthoid',\
	'sthoid_svd',\
	'sthoid_svd2',\
	'tensor_to_id',\
	'plotindlst']



class CP(ktensor):
	"""
	Tensor decomposition in Kruskal format
	
	See also:
	---------
		scikit.ktensor

	"""
	def __init__(self, U, lmbda = None):
		ktensor.__init__(self, U, lmbda = lmbda)

	def lowrank_matricize(self):
		"""
		Returns a list of low-rank matrices along each unfolding 
		A_i = U_iV_i^T

		Returns:
		--------
		ulst:	list of ndarray
				
	
		vlst:	list of ndarray

		"""

		U, l = self.U, self.lmbda
		dim  = self.ndim
		
		ulst, vlst = [], []
		L = np.diag(l)
		
		for n in range(dim):
			lst = list(range(n)) + list(range(n + 1, dim))

			utemp = [U[l] for l in lst]
			mat = khatrirao(tuple(utemp), reverse = True).conj().T

			
			ulst.append(U[n])
			vlst.append(dot(L,mat))

		return  ulst, vlst

	def matricize(self):	
		"""
		Returns a list of d-mode unfoldings Use with caution. 
		Can result in large matrices

		Returns:
		--------
		modelst:  list
			List of mode unfoldings. 
		
		"""
		ulst, vlst = self.lowrank_matricize()
		modelst = [dot(u,v) for u,v in zip(ulst,vlst)]

		
		return modelst


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

	def lowrank_matricize(self, method = 'modemult'):

		G, U = self.G, self.U
		ndim = self.ndim	
		vlst = []
		if method == 'kron':
			modes = matricize(dtensor(G))
			for d, mode in zip(range(ndim),modes):
				lst = [U[i] for i in (range(d) + range(d+1, ndim))] 
				v = dot(mode, kron(lst, reverse = True).conj().T)
				vlst.append(v)
		elif method == 'modemult':
			
			for d in range(ndim):
				lst = [U[i] for i in (range(d) + range(d+1, ndim))] 
				v = G.ttm(lst, mode = list(range(d) + range(d+1, ndim))).unfold(d)
				vlst.append(v)
		else: 
			raise NotImplementedError	


		return U, vlst	


	def matricize(self, method = 'modemult'):	

		ulst, vlst = self.lowrank_matricize(method)
		modes = [dot(u,v) for u,v in zip(ulst,vlst)]
	
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
	"""
	Parameters:
	-----------
	T:	(I_1,...,I_d) dtensor
		object of dtensor class. See scikit.dtensor
		
	rank:	(r_1,...,r_d) int array, optional
		Ranks of the individual modes	

	eps:	float, optional	
		Relative error of the representation

	method:	{'svd','randsvd'} string 
		SVD uses numpy SVD, RandSVD uses scipy.linalg.interpolative
	
	compute_core: bool, optional
		Compute core tensor by projection. True by default

	Returns:
	--------
	T:	object of Tucker class
		G - core, U - list of mode vectors. Returned if compute_core = True

	ulst:	list
		List of column vectors. Returned if compute_core = False


	"""
	modes = matricize(T)
	dim = T.ndim
	
	assert (rank is not None) or (eps is not None)
	if eps is not None: eps_or_k = eps/np.sqrt(dim)


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

def sthosvd(T, rank = None, eps = None, method = 'svd'):
	"""
	Parameters:
	-----------
	T:	(I_1,...,I_d) dtensor
		object of dtensor class. See scikit.dtensor
		
	rank:	(r_1,...,r_d) int array, optional
		Ranks of the individual modes	

	eps:	float, optional	
		Relative error of the representation

	method:	{'svd','randsvd'} string 
		SVD uses numpy SVD, RandSVD uses scipy.linalg.interpolative
	
	Returns:
	--------
	T:	object of Tucker class
		G - core, U - list of mode vectors. Returned if compute_core = True


	"""
	
	dim   = T.ndim

	#Core tensor and orthonormal factors
	ulst = []
	G = dtensor(T)
	for d in range(dim):
		mode = G.unfold(d)
		if rank is not None:	eps_or_k = rank[d]
		if method == 'svd':
			u, _, _ = svdtruncate(mode, eps_or_k)	
		elif method == 'randsvd':
			u, _, _ = randsvd(mode, eps_or_k)
		else:
			raise NotImplementedError
		ulst.append(u)
		
		#Recompute the core tensor (slightly more expensive)
		G = G.ttm(u, d, transp = True) 
		
	return Tucker(G = G, U = ulst) 
	

def hoid(T, rank = None, eps = None, method = 'qr', compute_core = True):
	"""
	Parameters:
	-----------
	T:	(I_1,...,I_d) dtensor
		object of dtensor class. See scikit.dtensor
	
	rank:	(r_1,...,r_d) int array, optional
		Ranks of the individual modes	

	eps:	float, optional	
		Relative error of the representation

	method:	{'qr','randsvd'} string, optional
		qr uses pivoted QR, 'interpolative' uses scipy.linalg.interpolative
		Uses 'qr' by default
	
	compute_core: bool, optional
		Compute core tensor by projection. True by default

	Returns:
	--------
	T:	object of Tucker class
		G - core, U - list of mode vectors, I - index list. 
		Returned if compute_core = True

	ulst:	list
		List of column vectors. Returned if compute_core = False

	"""

	modes = matricize(T)
	dim   = T.ndim

	assert (rank is not None) or (eps is not None)
	if eps is not None: eps_or_k = eps/np.sqrt(dim)
	
	clst, indlst = [], []
	for d, mode in zip(range(dim), modes):
		if rank is not None:	eps_or_k = rank[d]
		
		if method == 'qr':
			p, _ = id_(mode, eps_or_k)
		elif method == 'rrqr':
			assert eps_or_k >= 1.
			_,_,p = srrqr(mode, eps_or_k)
		elif method == 'rid':
			assert eps_or_k >= 1.
			rpp = mode.shape[0] + 10	#Oversampling factor
			Omega = np.random.randn(rpp,mode.shape[0])
			ind, _ = interp_decomp(np.dot(Omega,mode), eps_or_k, rand = False)
			p = ind[:eps_or_k]
		elif method == 'randsvd':
			assert eps_or_k >= 1.
			k = eps_or_k
			ind, _ = interp_decomp(mode, eps_or_k)
			p = ind[:k]
		else:
			raise NotImplementedError
			
		c = mode[:, p]
		clst.append(c)
		indlst.append(p)

	#Compute core tensor
	if compute_core:
		cinv = [pinv(c, 1.e-8) for c in clst]
		G = T.ttm(cinv)
		return Tucker(G = G, U = clst, I = indlst)
	else:
		return clst


def sthoid(T, rank = None,  method = 'rrqr', compute_core = True):
	"""
	Parameters:
	-----------
	T:	(I_1,...,I_d) dtensor
		object of dtensor class. See scikit.dtensor
	
	rank:	(r_1,...,r_d) int array, optional
		Ranks of the individual modes	

	method:	{'rrqr'(default),'dime'} string, optional
		deim uses DEIM method, dime uses pivoted QR, rrqr uses RRQR, 'lev' uses leverage scores approach
		Uses 'rrqr' by default
	
	compute_core: bool, optional
		Compute core tensor by projection. True by default

	Returns:
	--------
	T:	object of Tucker class
		G - core, U - list of mode vectors, I - index list. 
	
	"""

	dim   = T.ndim
	assert (rank is not None) or (eps is not None)

	#Core tensor and orthonormal factors
	ulst, indlst, modeerr = [], [], []
	
	G = dtensor(T)
	for d in range(dim):
		mode = G.unfold(d)
		r = rank[d]
		u, _, vh = svdtruncate(mode, r); v = vh.conj().T;
		
		if method == 'rrqr':
			_, _, p = srrqr(v.conj().T, r)
			fact = norm(inv(v[p,:]))
		elif method == 'deim':
			p, fact = deim(v)
		elif method == 'dime':
			p, fact = dime(v)
		else:
			raise NotImplementedError

		c = T.unfold(d)[:, p]

		G = G.ttm(pinv(c, rcond = 1.e-8), d)		
		if d > 0:
			Tk = G.ttm(ulst, mode = list(range(d))).unfold(d)
			_, _, v = lowrank_to_svd(c, Tk.conj().T)
		ulst.append(c)
		indlst.append(p)
		modeerr.append(fact)
	
	return Tucker(G = G, U = ulst, I = indlst), modeerr




def sthoid_svd(T, rank = None,  method = 'rrqr', compute_core = True):
	"""
	Compute STHOID but uses ST_HOSVD and then uses RRQR/DIME 
	Parameters:
	-----------
	T:	(I_1,...,I_d) dtensor
		object of dtensor class. See scikit.dtensor
	
	rank:	(r_1,...,r_d) int array, optional
		Ranks of the individual modes	

	method:	{'rrqr'(default),'dime'} string, optional
		deim uses DEIM method, dime uses pivoted QR, rrqr uses RRQR, 'lev' uses leverage scores approach
		Uses 'rrqr' by default
	
	compute_core: bool, optional
		Compute core tensor by projection. True by default

	Returns:
	--------
	T:	object of Tucker class
		G - core, U - list of mode vectors, I - index list. 
	
	"""

	dim   = T.ndim
	assert (rank is not None) or (eps is not None)

	#Core tensor and orthonormal factors
	ulst, clst, indlst, modeerr = [], [], [], []
	
	G = dtensor(T)
	for d in range(dim):
		mode = G.unfold(d)
		r = rank[d]
		u, _, vh = svdtruncate(mode, r); v = vh.conj().T;


		G = G.ttm(u, d, transp = True)		
		if d > 0:
			Tk = G.ttm(ulst, mode = list(range(d))).unfold(d)
			_, _, v = lowrank_to_svd(u, Tk.conj().T)
		ulst.append(u)
	
		if method == 'rrqr':
			_, _, p = srrqr(v.conj().T, r)
			fact = norm(inv(v[p,:]))
		elif method == 'deim':
			p, fact = deim(v)
		elif method == 'dime':
			p, fact = dime(v)
		else:
			raise NotImplementedError

		c = T.unfold(d)[:, p]
		clst.append(c)
		indlst.append(p)
		modeerr.append(fact)
	
	#Compute core tensor
	if compute_core:
		cinv = [pinv(c, rcond = 1.e-8) for c in clst]
		G = T.ttm(cinv)
		return Tucker(G = G, U = clst, I = indlst), modeerr
	else:
		return clst, modeerr


def sthoid_svd2(T, rank = None,  method = 'rrqr', compute_core = True):
	"""
	Compute STHOID but uses ST_HOSVD and then uses RRQR/DIME 
	Parameters:
	-----------
	T:	(I_1,...,I_d) dtensor
		object of dtensor class. See scikit.dtensor
	
	rank:	(r_1,...,r_d) int array, optional
		Ranks of the individual modes	

	method:	{'rrqr','r'} string, optional
		qr uses pivoted QR, 'interpolative' uses scipy.linalg.interpolative
		Uses 'qr' by default
	
	compute_core: bool, optional
		Compute core tensor by projection. True by default

	Returns:
	--------
	T:	object of Tucker class
		G - core, U - list of mode vectors, I - index list. 
	
	"""

	dim   = T.ndim
	
	Tt = sthosvd(T, rank)
	return tensor_to_id(T, Tt, rank, method = method, compute_core = compute_core)


def tensor_to_id(full, T, rank, method = 'deim', compute_core = True, **kwargs):
	"""
	Parameters:
	-----------
	full:	(I_1,...,I_d) dtensor
		object of dtensor class. See scikit.dtensor

	T:	(I_1,...,I_d) 
		object of CP/Tucker class. Low-rank representation
	
	rank:	(r_1,...,r_d) int array, optional
		Ranks of the individual modes	

	method:	{'deim','dime','rrqr','lev'} string, optional
		'deim' uses the DEIM approach, 'dime' uses Pivoted QR approach, 'rrqr' uses strong rank-revealing QR, \
			'lev' uses leverage score sampling
		See lowrank.py for details	
		
	compute_core: bool, optional
		Compute core tensor by projection. True by default

	lev_n: 	int, optional
		Number of vectors used in the leverage score 
	

	Returns:
	--------
	T:	object of Tucker class
		G - core, U - list of mode vectors, I - index list. 
		Returned if compute_core = True

	clst:	list
		List of column vectors. Returned if compute_core = False

	modeerr: tuple of size d  
		Multiplicative factor from converting tensor into HOID format

	"""

	dim = T.ndim
 	
	if not (isinstance(T, Tucker) or isinstance(T, CP)):
		raise NotImplementedError, "Only implemented for Tucker and CP"

	#Unfold
	modes = matricize(full)	
	alst, blst = T.lowrank_matricize()

	clst, indlst = [], []

	modeerr = []
	for a, b, mode, r  in zip(alst, blst, modes, rank):
		_, _, v = lowrank_to_svd(a, b.conj().T)
		v = v[:,:r]
		if method == 'deim':
			q, fact = deim(v)
		elif method == 'dime':
			q, fact = dime(v)
		elif method == 'rrqr':
			_, _, q = srrqr(v.conj().T, r)
			fact = norm(inv(v[q,:]))
		elif method == 'lev':
			q, fact = leverage_sampling(v, r)
		else:
			raise NotImplementedError

		modeerr.append(fact)
		c = mode[:,q]

		clst.append(c)
		indlst.append(q)

	modeerr = tuple(modeerr)


	#Compute core tensor
	if compute_core:
		cinv = [pinv(c, rcond = 1.e-8) for c in clst]
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


	Tf, _ = tensor_to_id(A, T, rank, method = 'dime')
	Tmodes = Tf.matricize()
	Amodes = matricize(A)

	isclose = [np.allclose(a,t) for a, t in zip(Amodes,Tmodes)]
	print isclose
