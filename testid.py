from core import *
from lowrank import *
import numpy as np
from sktensor import dtensor, cp_als, khatrirao
from decomp import *
from numpy.linalg import norm
import matplotlib
matplotlib.rcParams['xtick.labelsize'] = 15.
matplotlib.rcParams['ytick.labelsize'] = 15.

np.random.seed(223)

def ten_id(A, r, method = 'svd'):

	rank = (r,r,r)

	A = dtensor(A)
	Amodes = matricize(A)
	
	if method == 'svd':
		T = hosvd(A, rank, method = 'svd')

	elif method == 'randsvd':
		T = hosvd(A, rank, method = 'randsvd')
	elif method == 'rid':
		#Tensor ID
		T = hoid(A, rank, method = 'rid') 
	elif method == 'pqr':
		#Tensor ID
		T = hoid(A, rank, method = 'qr') 
	elif method == 'id':
		#Tensor ID
		T = hoid(A, rank, method = 'rrqr') 

	elif method == 'stid':
		T = sthoid(A, rank, method = 'rrqr') 

	else:
		raise NotImplementedError

	#Tmodes = T.matricize()
	#err = [np.linalg.norm(tm-am)/np.linalg.norm(am) for am,tm in zip(Amodes,Tmodes)]

	am = A.unfold(0)
	tm = T.G.ttm(T.U).unfold(0)	
	err = norm(tm-am)/norm(am)



	return (err, T.I) if method == 'id' else err

def lowrank_to_id(A, H, rank, method = 'deim'):

	Amodes = matricize(A)
		
	T, modeerr = tensor_to_id(A, H, rank, method = method)
	#Tmodes = T.matricize()
	#err = [np.linalg.norm(tm-am)/np.linalg.norm(am) for am,tm in zip(Amodes,Tmodes)]
	
	am = A.unfold(0)
	tm = T.G.ttm(T.U).unfold(0)	
	err = norm(tm-am)/norm(am)


	return err, modeerr
	


def leverage_accuracy(A, H, maxr, ns = 10):

	#Compute right singular vectors
	v = []
	Amodes = matricize(A)
	for d in range(A.ndim):
		_, _, vh = np.linalg.svd(Amodes[d], full_matrices = False)
		v.append(vh[:ns,:].T)

	lev_mode_err = np.zeros((maxr,3), dtype = 'd')
	err_lev = np.zeros((maxr,), dtype = 'd')
	#Compute accuracy 
	for r in np.arange(maxr):
		clst = []
		for d, mode in zip(range(A.ndim), Amodes):
			q, fact = leverage_sampling(v[d], r+1)	
			clst.append(mode[:,q])
			lev_mode_err[r,d] = fact
		cinv = [np.linalg.pinv(c) for c in clst]
		T = Tucker(G = A.ttm(cinv), U = clst); 
		am = A.unfold(0)
		tm = T.G.ttm(T.U).unfold(0)	
		err_lev[r] = norm(tm-am)/norm(am)

	
		#Tmodes = T.matricize();
		#err = [np.linalg.norm(tm-am)/np.linalg.norm(am) for am,tm in zip(Amodes,Tmodes)]
		#err_lev[r] = np.max(np.array(err))

	return err_lev, lev_mode_err 


def test1(A, maxr = 10, fname = 'hilbert'):
	err_s = np.zeros((maxr,), dtype = 'd')
	err_r = np.zeros((maxr,), dtype = 'd')
	err_p = np.zeros((maxr,), dtype = 'd')
	err_i = np.zeros((maxr,), dtype = 'd')
	err_st = np.zeros((maxr,), dtype = 'd')
	
	for r in np.arange(maxr):
		err_s[r] = ten_id(A, r = r+1, method = 'svd')
		err_r[r] = ten_id(A, r = r+1, method = 'randsvd')
		err_r[r] = ten_id(A, r = r+1, method = 'pqr')
		err_i[r], _ = ten_id(A, r = r+1, method = 'id')
		err_st[r] = ten_id(A, r = r+1, method = 'stid')

	plt.figure()
	plt.semilogy(np.arange(maxr)+1, err_s, 'k-', linewidth = 2.)
	plt.semilogy(np.arange(maxr)+1, err_r, 'g-', linewidth = 2.)
	plt.semilogy(np.arange(maxr)+1, err_i, 'r--', linewidth = 2.)
	plt.semilogy(np.arange(maxr)+1, err_p, 'c-', linewidth = 2.)
	plt.semilogy(np.arange(maxr)+1, err_st, 'm-', linewidth = 2.)
	plt.legend(('HOSVD', 'HORID', 'HOID - RRQR', 'HOID - PQR', 'STHOID'))
	plt.xlabel('Rank [r]', fontsize = 18.)
	plt.ylabel('rel. err.' , fontsize = 18.)
	plt.title('Relative Error', fontsize = 24)
	
	plt.savefig('figs/hoid_' + fname + '.png')	
	


	return

def test2(A, maxr = 10, fname = 'hilbert', ns = 10):
	
	A = dtensor(A)
	Amodes = matricize(A)	
	
	err_deim = np.zeros((maxr,), dtype = 'd')	
	err_dime = np.zeros((maxr,), dtype = 'd')	
	err_rrqr = np.zeros((maxr,), dtype = 'd')	

	deim_mode_err = np.zeros((maxr,3), dtype = 'd')
	dime_mode_err = np.zeros((maxr,3), dtype = 'd')
	rrqr_mode_err = np.zeros((maxr,3), dtype = 'd')
		
	for r in np.arange(maxr):
		#Approximate the matrix using HOSVD
		rank = (r+1,r+1,r+1)
		H = hosvd(A, rank, method = 'randsvd')
		err_deim[r], mode = lowrank_to_id(A, H, rank, method = 'deim')
		deim_mode_err[r,:] = mode
		err_dime[r], mode = lowrank_to_id(A, H, rank, method = 'dime')
		dime_mode_err[r,:] = mode
		err_rrqr[r], mode = lowrank_to_id(A, H, rank, method = 'rrqr')
		rrqr_mode_err[r,:] = mode

	err_lev, lev_mode_err = leverage_accuracy(A, H, maxr, ns = ns)	

	plt.figure()
	plt.semilogy(np.arange(maxr) + 1, np.max(deim_mode_err,axis=1) , \
			'k-', linewidth = 2., label = r'DEIM') 
	plt.semilogy(np.arange(maxr) + 1, np.max(dime_mode_err,axis=1) , \
			'c-', linewidth = 2., label = r'PQR') 
	plt.semilogy(np.arange(maxr) + 1, np.max(rrqr_mode_err,axis=1) , \
			'r--', linewidth = 2., label = r'RRQR') 
	plt.xlabel('Rank [r]', fontsize = 18.)
	plt.ylabel(r'max. $||(P^*V)^{-1}||_F$' , fontsize = 18.)

	plt.title('Error constants', fontsize = 24)
	plt.legend(loc = 4)		
	plt.savefig('figs/deim_err_' + fname + '.png')


	plt.figure()
	plt.semilogy(np.arange(maxr) + 1, lev_mode_err, linewidth = 2.) 
	plt.xlabel('Rank [r]', fontsize = 18.)
	plt.ylabel(r'$||(P^*V)^{-1}||_F$' , fontsize = 18.)
	plt.legend(('Mode 1', 'Mode 2', 'Mode 3'))
	plt.title('Error constants - Leverage scores ', fontsize = 20)
	plt.savefig('figs/lev_err_' + fname + '.png')

	

	plt.figure()
	
	plt.semilogy(np.arange(maxr)+1, err_deim, 'k-', linewidth = 3.)
	plt.semilogy(np.arange(maxr)+1, err_dime, 'c-', linewidth = 2.)
	plt.semilogy(np.arange(maxr)+1, err_rrqr, 'r--', linewidth = 2.)
	plt.semilogy(np.arange(maxr)+1, err_lev, 'b-', linewidth = 2.)
	plt.legend(('DEIM', 'PQR', 'RRQR', 'Lev'), loc = 3)
	plt.xlabel('Rank [r]', fontsize = 18.)
	plt.ylabel('rel. err.' , fontsize = 18.)
	plt.title('Relative Error', fontsize = 24)
	plt.savefig('figs/deim_' + fname + '.png')	

	return	
def test3(A, maxr = 10, fname = 'hilbert', ns = 10):
	
	A = dtensor(A)
	Amodes = matricize(A)	
	
	H = hosvd(A, rank = A.shape, method = 'svd')
	err_lev, lev_mode_err = leverage_accuracy(A, H, maxr, ns = ns)	

	plt.figure()
	plt.semilogy(np.arange(maxr) + 1, lev_mode_err, linewidth = 2.) 
	plt.xlabel('Rank [r]', fontsize = 18.)
	plt.ylabel(r'$||(P^*V)^{-1}||_F$' , fontsize = 18.)
	plt.legend(('Mode 1', 'Mode 2', 'Mode 3'))
	plt.title('Error constants - Leverage scores ', fontsize = 20)
	plt.savefig('figs/lev_err_' + fname + '.png')


	return	



def generate_tensor(n = 20, opt = 0):
	mfact = 2. if opt == 0 else 1000.
	e1 = np.arange(10) + 1;	e2 = np.arange(10,n) + 1;	
	fact = np.concatenate((mfact/e1,1./e2))

	from scipy.sparse import rand
	
	u = rand(n, n, density = 0.1).todense()
	v = rand(n, n, density = 0.1).todense()
	w = rand(n, n, density = 0.1).todense()
	
	U = [u, v, w] 

	G = np.zeros((n,n,n), dtype = 'd')
	for k in range(n):	G[k,k,k] = fact[k]

	T = Tucker(G = G, U = U)

	return T


def vis_ind(A, fname = 'hilbert'):
	
	#Visualize indices
	_, ind = ten_id(A, r = 10, method = 'id')
	plotindlst(dtensor(A), ind)
	plt.savefig('figs/indlst' + fname + '.png')

	return



def example1(n = 20, visualize = False):
	#Generate tensor
	arr = np.arange(n)+1
	x, y, z = np.meshgrid(arr, arr, arr)
	A = 1./np.sqrt(x**2. + y**2. + z**2.)

	import matplotlib.pyplot as plt
	plt.close('all')
	

	#Plot error using HOSVD, HORSVD, HOID
	test1(A, maxr = 8, fname = 'hilbert')
	#Plot Error using DEIM and DIME and leverage score
	test2(A,  maxr = 8, fname = 'hilbert', ns = -1)

	if visualize: vis_ind(A, fname = 'hilbert')

	return

	
def example2(n = 20, opt = 1, ns = 10, visualize = False):

	#Generate Tucker factorization
	T = generate_tensor(n = n, opt = 1)
	modes = T.matricize()
	
	#Generate full tensor
	A = dtensor(np.reshape(np.array(modes[0]),(n,n,n)))
	
	#Plot error using HOSVD, HORSVD, HOID
	test1(A, maxr = 15, fname = 'sparse')
	#Plot Error using DEIM and DIME and leverage score
	test2(A, maxr = 15, fname = 'sparse', ns = ns) 

	if visualize: vis_ind(A, fname = 'sparse')

	return

def example2b(n = 20, opt = 1):

	#Generate Tucker factorization
	T = generate_tensor(n = n, opt = 1)
	modes = T.matricize()
	
	#Generate full tensor
	A = dtensor(np.reshape(np.array(modes[0]),(n,n,n)))
	
	#Plot Error using DEIM and DIME and leverage score
	test3(A, maxr = 15, fname = 'sparse_10', ns = 15) 
	test3(A, maxr = 15, fname = 'sparse_all', ns = -1) 

	return



if __name__ == '__main__':
	import matplotlib.pyplot as plt
	plt.close('all')
	
	
	example1(n = 50)
	example2(n = 50, opt = 1, ns = -1)	
	example2b(n = 50, opt = 1)	
	plt.show()
