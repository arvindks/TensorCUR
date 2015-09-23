from core import *
from lowrank import *
import numpy as np
from sktensor import dtensor, cp_als, khatrirao
from cp3 import CP 
from tucker3 import *

import matplotlib
matplotlib.rcParams['xtick.labelsize'] = 15.
matplotlib.rcParams['ytick.labelsize'] = 15.


def ten_id(A, r, method = 'svd'):

	rank = (r,r,r)

	A = dtensor(A)
	Amodes = matricize(A)
	

	if method == 'svd':
		T = hosvd(A, rank, method = 'svd')

	elif method == 'randsvd':
		T = hosvd(A, rank, method = 'randsvd')

	elif method == 'id':
		#Tensor ID
		T = hoid(A, rank) 

	else:
		raise NotImplementedError

	Tmodes = T.matricize()
	err = [np.linalg.norm(tm-am)/np.linalg.norm(am) for am,tm in zip(Amodes,Tmodes)]

	return (np.max(err), T.I) if method == 'id' else np.max(err)
def lowrank_to_id(A, H, rank, method = 'deim'):

	Amodes = matricize(A)
		
	T, modeerr = tensor_to_id(A, H, rank, method = method)
	Tmodes = T.matricize()
	err = [np.linalg.norm(tm-am)/np.linalg.norm(am) for am,tm in zip(Amodes,Tmodes)]

	return np.max(err), modeerr
	


def test1(A, maxr = 10):
	err_s = np.zeros((maxr,), dtype = 'd')
	err_r = np.zeros((maxr,), dtype = 'd')
	err_i = np.zeros((maxr,), dtype = 'd')
	
	for r in np.arange(maxr):
		err_s[r] = ten_id(A, r = r+1, method = 'svd')
		err_r[r] = ten_id(A, r = r+1, method = 'randsvd')
		err_i[r], _ = ten_id(A, r = r+1, method = 'id')

	plt.figure()
	plt.semilogy(np.arange(maxr)+1, err_s, 'k-', linewidth = 2.)
	plt.semilogy(np.arange(maxr)+1, err_r, 'g-', linewidth = 2.)
	plt.semilogy(np.arange(maxr)+1, err_i, 'r-', linewidth = 2.)
	plt.legend(('SVD', 'RandSVD', 'ID'))
	plt.xlabel('Rank [r]', fontsize = 18.)
	plt.ylabel('rel. err.' , fontsize = 18.)
	plt.title('Relative Error', fontsize = 24)
	
	plt.savefig('figs/hoid_hilbert.png')	
	


	return

def test2(A, maxr = 10):
	
	A = dtensor(A)
	Amodes = matricize(A)	
	
	err_deim = np.zeros((maxr,), dtype = 'd')	
	err_dime = np.zeros((maxr,), dtype = 'd')	

	deim_mode_err = np.zeros((maxr,3), dtype = 'd')
	dime_mode_err = np.zeros((maxr,3), dtype = 'd')
	
	for r in np.arange(maxr):
		#Approximate the matrix using HOSVD
		rank = (r+1,r+1,r+1)
		H = hosvd(A, rank, method = 'svd')

		err_deim[r], mode = lowrank_to_id(A, H, rank, method = 'deim')
		deim_mode_err[r,:] = mode
		err_dime[r], mode = lowrank_to_id(A, H, rank, method = 'dime')
		dime_mode_err[r,:] = mode

	

	plt.figure()
	plt.semilogy(np.arange(maxr) + 1, np.max(deim_mode_err,axis=1) , 'k-', linewidth = 2., label = r'DEIM') 
	plt.semilogy(np.arange(maxr) + 1, np.max(dime_mode_err,axis=1) , 'r-', linewidth = 2., label = r'DIME') 
	plt.xlabel('Rank [r]', fontsize = 18.)
	plt.ylabel(r'max. $||(P^*V)^{-1}||_F$' , fontsize = 18.)

	plt.legend(loc = 4)		
	plt.savefig('figs/deim_err.png')


	plt.figure()
	
	plt.semilogy(np.arange(maxr)+1, err_deim, 'k-', linewidth = 2.)
	plt.semilogy(np.arange(maxr)+1, err_dime, 'g-', linewidth = 2.)
	plt.legend(('DEIM', 'DIME'))
	plt.xlabel('Rank [r]', fontsize = 18.)
	plt.ylabel('rel. err.' , fontsize = 18.)
	plt.title('Relative Error', fontsize = 24)
	plt.savefig('figs/deim_hilbert.png')	

	return	



def testcp(A, maxr = 10):

	Ad = dtensor(A)

	err_cp = np.zeros((maxr,), dtype = 'd')	
	err_id = np.zeros((maxr,), dtype = 'd')	

	for r in np.arange(maxr):

		P, fit, itr, _ = cp_als(Ad, r+1)
		T = P.toarray()
		err_cp[r] = np.linalg.norm(A-T)/np.linalg.norm(T)
		rank = (r+1,r+1,r+1)
		err_id[r] = lowrank_to_id(Ad, CP(P.U, P.lmbda), rank, method = 'dime')

	plt.figure()
	
	plt.semilogy(np.arange(maxr)+1, err_cp, 'r-', linewidth = 2.)
	plt.semilogy(np.arange(maxr)+1, err_id, 'g-', linewidth = 2.)


	return err_cp


def vis_ind(A):
	
	#Visualize indices
	_, ind = ten_id(A, r = 10, method = 'id')
	plotindlst(dtensor(A), ind)
	plt.savefig('figs/indlst.png')

	return



if __name__ == '__main__':

	#Generate tensor
	n = 20
	arr = np.arange(n)+1
	x, y, z = np.meshgrid(arr, arr, arr)
	A = 1./np.sqrt(x**2. + y**2. + z**2.)
	#A = 1./(x+y+z+3.) 

	import matplotlib.pyplot as plt
	plt.close('all')
	

	#Plot error using HOSVD, HORSVD, HOID
	#test1(A)
	#Plot Error using DEIM and DIME
	#test2(A)

	#Visualize indices from HOID
	vis_ind(A)
		
	#testcp(A)
		
	plt.show()
