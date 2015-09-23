import numpy as np
from scipy.linalg import qr, svd

__all__ = ['deim', \
	   'dime', \
	   'svdtruncate',\
	   'randsvd',\
	   'id_',\
	   'lowrank_to_id',\
	   'lowrank_to_qr',\
	   'lowrank_to_svd']


def deim(V, verbose = False):


	k  = V.shape[1]
	p1 = np.argmax(np.abs(V[:,0]))
	p  = [p1]



	for j in np.arange(1,k):
		Vj = V[:,:j] 
		vj = V[:,j]
	

		pj = np.array(p)
		r  =  vj - np.dot(Vj, np.linalg.solve(Vj[pj,:], vj[pj]))
		
		pj = np.argmax(np.abs(r))
		p.append(pj)

	p = np.array(p)
	
	mat = V[p,:]
	fact = np.linalg.norm(np.linalg.inv(mat))
	if verbose:	print "DEIM Error is %g" %(fact )

	return p, fact
	

def dime(V, verbose = False):
	
	k = V.shape[1]
	_, _, p = qr(V.conj().T, mode = 'economic', pivoting = True)

	p = p[:k]

	mat  = V[p,:]
	fact = np.linalg.norm(np.linalg.inv(mat))
	if verbose:	print "DIME Error is %g" %(fact )

	return p, fact


def svdtruncate(A, eps_or_k):
	
	u, s, vh = svd(A, full_matrices = False)
	ind = np.flatnonzero(s/s[0] > eps_or_k) if eps_or_k < 1 else range(eps_or_k)
	u, s, vh = u[:,ind], s[ind], vh[ind,:]


	return u, s, vh


def randsvd(A, eps_or_k):

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


	#Compute rank-revealing QR factorization
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
		
	T,_,_,_    = np.linalg.lstsq(R11,R12)
	X    	   = np.hstack([np.eye(T.shape[0]), T])[:,np.argsort(p)]
	

	return q, X 


def lowrank_to_id(A,B):

	k = B.shape[0]
	p, X = id_(B.T, k)

	return p, X
	


def lowrank_to_qr(A, B):
	""" AB^T"""  

	qa, ra = qr(A, mode = 'economic')
	d = np.dot(ra, B.conj().T)
	qd, r = qr(d, mode = 'economic')

	q = np.dot(qa,qd)

	return q,r 



def lowrank_to_svd(A, B):
	"""M = AB^T"""

	qa, ra = qr(A, mode = 'economic')
	qb, rb = qr(B, mode = 'economic')

	mat = np.dot(ra,rb.conj().T)
	u, s, vh = svd(mat)

	return np.dot(qa, u), s, np.dot(qb,vh.conj().T)

