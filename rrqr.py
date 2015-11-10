import numpy as np
from scipy.linalg import qr, svd
from numpy.linalg import norm, inv, slogdet, cond

__all__ = ['srrqr']


def srrqr(M, k, f = 1., verbose = False):
	
	m, n = M.shape
	minmn = np.min([m,n])
	assert k <= minmn, "k must be less than min{m,n} k = %d,m = %d, n = %d" % (k,m,n) 



	#QR with column pivoting
	Q, R, p = qr(M, mode = 'economic', pivoting = True)

	if k == n:	return Q, R, p 
	
	increase_found 	= True
	counter_perm   	= 0
	iterc		= 0
	


	while (increase_found) and iterc <= 100:
		iterc += 1		

		A = R[:k,:k]
		AinvB = np.linalg.solve(A,R[:k,k:])	#Form A^{-1}B

		C = R[k:minmn,k:] 
		
		#Compute column norms of C
		if k < m:	gamma = np.apply_along_axis(norm, 0, C[:,:n-k])
		
		#Find row norms of A^{-1}
		omega = np.apply_along_axis(norm, 0, inv(A).T)


		F = AinvB**2. 
		if k < m:	F += (np.outer(omega,gamma))**2.
		ind = np.argwhere(F > f**2.)
		if ind.size == 0:	#finished
			increase_found = False
		else:	#we can increase |det(A)|
			i, j = ind[0,:]
			counter_perm += 1
			#permute columns i and j
			R[:,[i,j+k]] = R[:,[j+k,i]]
			p[[i,j+k]]   = p[[j+k, i]]

			#retriangularize R
			q,R = qr(R, mode = 'economic')
			Q = np.dot(Q,q)	
		#print p

	Rkk = R[:k,:k]
	inv_norm = norm(inv(Rkk),2)
	res_norm = norm(R[k:minmn,k:],2) if k < minmn else 0.

	

	if verbose:	
		print "Norm of inverse is %g" % (inv_norm)
		print "Norm of residual is %g" % (res_norm)
		sgn, det = slogdet(Rkk)
		print "Log-determinant of selected columns is %g with sign %g" %\
				(det, sgn)
		print "Conditioning of selected columns is %g" %(cond(Rkk))

		
	p = p[:k]
	return Q, R, p	
if __name__ == '__main__':
	from scipy.linalg import hadamard
	
	A = np.random.randn(100,30)
	k = 30 
	Q, R, p = srrqr(A, k, verbose = True)
	print p
	print np.allclose(A[:,p],np.dot(Q,R[:,:k])) 

	A = hadamard(32, dtype = 'd')
	k = 5 
	Q, R, p = srrqr(A, k, verbose = True)
	print p
	print np.allclose(A[:,p],np.dot(Q,R[:,:k])) 

	A = np.random.randn(30,100)
	k = 30 
	Q, R, p = srrqr(A, k, verbose = True)
	print p
	print np.allclose(A[:,p],np.dot(Q,R[:,:k])) 
