import numpy as np
from scipy.linalg import qr, svd

from numpy import dot
from numpy.linalg import solve, norm, lstsq, inv, pinv
from rrqr import srrqr

__all__ = ['deim', \
	   'dime', \
	   'svdtruncate',\
	   'randsvd',\
	   'id_',\
	   'leverage_sampling',\
	   'lowrank_to_id',\
	   'lowrank_to_qr',\
	   'lowrank_to_svd']


def deim(V, verbose = False):
	"""
	Parameters:
	-----------
	V:	(m,k) ndarray
		Input matrix with orthonormal columns
	verbose: bool, optional
		Display the DEIM error. False by default	

	Returns:
	--------
	p:	(k,) array of int
		Index set of 
	fact:	float
		Error in the DEIM computation ||(P^*V)^{-1}||
	
	References:
	-----------

	.. [1] Chaturantabut, Saifon, and Danny C. Sorensen. 
		"Nonlinear model reduction via discrete empirical interpolation." 
		SIAM Journal on Scientific Computing 32.5 (2010): 2737-2764.

	"""
	k  = V.shape[1]
	p1 = np.argmax(np.abs(V[:,0]))
	p  = [p1]

	for j in np.arange(1,k):
		Vj = V[:,:j] 
		vj = V[:,j]
	

		pj = np.array(p)
		r  =  vj - dot(Vj, solve(Vj[pj,:], vj[pj]))
		
		pj = np.argmax(np.abs(r))
		p.append(pj)

	p = np.array(p)
	
	mat = V[p,:]
	fact = norm(inv(mat))
	if verbose:	print "DEIM Error is %g" %(fact )

	return p, fact
	

def dime(V, verbose = False):
	"""
	Parameters:
	-----------
	V:	(m,k) ndarray
		Input matrix with orthonormal columns
	verbose: bool, optional
		Display the DEIM error. False by default	

	Returns:
	--------
	p:	(k,) array of int
		Index set of 
	fact:	float
		Error in the DIME computation ||(P^*V)^{-1}||
	
	References:
	-----------

	.. [1] Drmac, Zlatko, and Serkan Gugercin. 
		"A New Selection Operator for the Discrete Empirical Interpolation 
		Method -- improved a priori error bound and extensions." 
		arXiv preprint arXiv:1505.00370 (2015).	
	"""

	k = V.shape[1]
	_, _, p = qr(V.conj().T, mode = 'economic', pivoting = True)
	p = p[:k]	

	mat  = V[p,:]
	fact = norm(inv(mat))
	if verbose:	print "DIME Error is %g" %(fact )

	return p, fact



def leverage_sampling(V, k, verbose = False):
	"""
	Parameters:
	-----------
	V:	(m,k) ndarray
		Input matrix with orthonormal columns
	verbose: bool, optional
		Display the error using leverage scores. False by default	

	Returns:
	--------
	p:	(k,) array of int
		Index set of 
	fact:	float
		Error in the DIME computation ||(P^*V)^{-1}||
	
	References:
	-----------
	.. [1] 


	"""

	score = np.sum(V**2.,1)
	pk = score/np.sum(score)

	
	from scipy import stats
	xk = np.arange(V.shape[0])
	lev =  stats.rv_discrete(name='lev', values=(xk, pk))
	
	#The factor 4 was found by experimentation
	ns = np.max([4*k, int(k*np.log(k))])
	
	
	#Randomized then deterministic strategy
	found = False
	count = 0	
	while not found:
		count += 1

		rand_sample = lev.rvs(size = ns)
		_, _, p = srrqr(V[rand_sample,:].conj().T, k)
		p = rand_sample[p]

		fact = norm(pinv(V[p,:k]))

		if fact < 1.e10:	break

	if verbose:	print "Number of rounds are %d" %(count)




	return p, fact





def svdtruncate(A, eps_or_k):
	"""
	Parameters:
	----------
	eps_or_k : float or int
        	Relative error (if `eps_or_k < 1`) or rank (if `eps_or_k >= 1`) of
        	approximation.

	Returns:
	--------
       	u : (m, k) ndarray
        	Unitary matrices. 
    	s : (k,) array
        	The singular values for every matrix, sorted in descending order.
    	vh : (n, k) array
        	Unitary matrices.
	Raises
    	------
    	LinAlgError
        	If SVD computation does not converge.	

	"""	
	u, s, vh = svd(A, full_matrices = False)
	ind = np.flatnonzero(s/s[0] > eps_or_k) if eps_or_k < 1 else range(eps_or_k)
	u, s, vh = u[:,ind], s[ind], vh[ind,:]

	return u, s, vh


def randsvd(A, eps_or_k):
	"""
	Interpolative decomposition computed using randomized SVD approach. It is a 
	wrapper for scipy.linalg.interpolative
	Parameters:
	----------
	eps_or_k : float or int
        	Relative error (if `eps_or_k < 1`) or rank (if `eps_or_k >= 1`) of
        	approximation.

	Returns:
	--------
       	u : {(m, k) } array
        	Unitary matrices. The actual shape depends on the value of
        	``full_matrices``. Only returned when ``compute_uv`` is True.
    	s : (k,) array
        	The singular values for every matrix, sorted in descending order.
    	v : {(n, k) } array
        	Unitary matrices. The actual shape depends on the value of
        	``full_matrices``. Only returned when ``compute_uv`` is True.

	Notes:
	------
		See scipy.linalg.interpolative function interp_decomp, id_to_svd

	"""	

	from scipy.linalg.interpolative import interp_decomp, id_to_svd
	if eps_or_k < 1:
		k, idx, proj = interp_decomp(A, eps_or_k)
		B = A[:,idx[:k]]
	else:
		idx, proj= interp_decomp(A, eps_or_k)
		k = eps_or_k

	B = A[:,idx[:k]]
	u, s, vh = id_to_svd(B, idx, proj)

	return u, s, vh


def id_(A, eps_or_k):
	"""
	Interpolative decomposition computed using Pivoted QR factorization

	Parameters:
	----------
	eps_or_k : float or int
        	Relative error (if `eps_or_k < 1`) or rank (if `eps_or_k >= 1`) of
        	approximation.

	Returns:
	--------
       	u : {(m, k) } array
        	Unitary matrices. The actual shape depends on the value of
        	``full_matrices``. Only returned when ``compute_uv`` is True.
    	s : (k,) array
        	The singular values for every matrix, sorted in descending order.
    	v : {(n, k) } array
        	Unitary matrices. The actual shape depends on the value of
        	``full_matrices``. Only returned when ``compute_uv`` is True.

	References:
	----------
	.. [1] Cheng, Hongwei, Zydrunas Gimbutas, Per-Gunnar Martinsson, 
		and Vladimir Rokhlin. "On the compression of low rank matrices." 
		SIAM Journal on Scientific Computing 26, no. 4 (2005): 1389-1404.

	"""	


	#Compute Pivoted QR factorization
	_, R, p = qr(A, mode = 'economic', pivoting = True)

	s = np.abs(np.diag(R))
	if eps_or_k < 1.:
		ind = np.argwhere(s/s[0] > eps_or_k)
		rank = ind.size
	else:
		rank = eps_or_k 
	
	R11 = R[:rank,:rank]
	R12 = R[:rank,rank:]	

	#Columns of A that are relevant
	q = p[:rank]
		
	T,_,_,_    = lstsq(R11,R12)
	X    	   = np.hstack([np.eye(T.shape[0]), T])[:,np.argsort(p)]

	return q, X 


def lowrank_to_id(A,B):
	"""
	Convert a low-rank factorization C appr A*B into an interpolative decomposition
	C approx C[:,ind]* X


	Parameters:
	-----------
	A:	(m,k) ndarray
		First of the two low-rank matrices
	B:	(k,n) ndarray
		Second of the two low-rank matrices
		
	Returns:
	--------
	
	ind:	(k,) ndarray
		Indices of columns to be sampled from C

	X:	(k, n) ndarray
		Interpolation coefficients	
	
	Notes:
	------
	
	See also scipy.linalg.interpolative


	"""

	k = B.shape[0]
	p, X = id_(B.T, k)

	return p, X
	


def lowrank_to_qr(A, B):
	"""
	Convert a low-rank factorization C appr A*B into an equivalent QR 
	C approx QR 


	Parameters:
	-----------
	A:	(m,k) ndarray
		First of the two low-rank matrices
	B:	(k,n) ndarray
		Second of the two low-rank matrices
		
	Returns:
	--------
	Q:	(m,k) ndarray
		
		
		
	
	Notes:
	------
	
	"""

	qa, ra = qr(A, mode = 'economic')
	d = dot(ra, B.conj().T)
	qd, r = qr(d, mode = 'economic')

	q = dot(qa,qd)

	return q,r 



def lowrank_to_svd(A, B):
	"""M = AB^T"""

	qa, ra = qr(A, mode = 'economic')
	qb, rb = qr(B, mode = 'economic')

	mat = dot(ra,rb.conj().T)
	u, s, vh = svd(mat)

	return dot(qa, u), s, dot(qb,vh.conj().T)


	
