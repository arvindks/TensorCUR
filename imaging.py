from core import *
from lowrank import *
import numpy as np
from sktensor import dtensor, cp_als, khatrirao
from decomp import *

from numpy.linalg import norm
import matplotlib.pyplot as plt


import matplotlib
matplotlib.rcParams['xtick.labelsize'] = 15.
matplotlib.rcParams['ytick.labelsize'] = 15.



def ten_id(A, r, method = 'svd'):
	d = A.ndim
	rank = tuple(r*np.ones((d,), dtype = 'i'))

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

	T, modeerr = tensor_to_id(A, H, rank, method = method)
	#Tmodes = T.matricize()


	am = A.unfold(0)
	tm = T.G.ttm(T.U).unfold(0)	
	err = norm(tm-am)/norm(am)


	return err, modeerr
	


def imaging_tensor_5d(ns = 20, nr = 20, nt = 20):
	
	xs = np.linspace(-1.,1.,ns)
	ys = np.linspace(-1.,1.,ns)
	
	xr = np.linspace(-1.,1.,nr)
	yr = np.linspace(-1.,1.,nr)

	t = np.linspace(0.1, 1.1, nt)

	Xs, Ys, Xr, Yr, T = np.meshgrid(xs, ys, xr, yr, t)
	
	K = np.power(4*np.pi*T,-1.5)*np.exp(-((Xs-Ys)**2.+(Xr-Yr)**2. + 2.**2.)/(4.*T))

	return K

def imaging_tensor_3d(ns = 20, nr = 20, nt = 20):
	xs = np.linspace(-1.,1.,ns)
	ys = np.linspace(-1.,1.,ns)
	
	Xs, Ys = np.meshgrid(xs, ys)
	Xs, Ys = np.meshgrid(Xs.ravel(), Ys.ravel())	

	xr = np.linspace(-1.,1.,nr)
	yr = np.linspace(-1.,1.,nr)

	Xr, Yr = np.meshgrid(xr, yr)
	Xr, Yr = np.meshgrid(Xr.ravel(), Yr.ravel())	
	t = np.linspace(0.1, 1.1, nt)

	K = np.zeros((ns**2.,nr**2.,nt), dtype = 'd') 
	for k in np.arange(nt):
		K[:,:,k] = np.power(4*np.pi*t[k],-1.5)*np.exp(-((Xs-Ys)**2.+(Xr-Yr)**2. + 2.**2.)/(4.*t[k]))

	return K


def test1(A, maxr = 10, fname = 'imaging'):

	err_s = np.zeros((maxr,), dtype = 'd')
	err_r = np.zeros((maxr,), dtype = 'd')
	err_p = np.zeros((maxr,), dtype = 'd')
	err_i = np.zeros((maxr,), dtype = 'd')
	err_st = np.zeros((maxr,), dtype = 'd')

	
	for r in np.arange(maxr):
		err_s[r] = ten_id(A, r = r+1, method = 'svd')
		err_r[r] = ten_id(A, r = r+1, method = 'rid')
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
	
	d = A.ndim

	err_deim = np.zeros((maxr,), dtype = 'd')	
	err_dime = np.zeros((maxr,), dtype = 'd')	
	err_rrqr = np.zeros((maxr,), dtype = 'd')	

	deim_mode_err = np.zeros((maxr,d), dtype = 'd')
	dime_mode_err = np.zeros((maxr,d), dtype = 'd')
	rrqr_mode_err = np.zeros((maxr,d), dtype = 'd')
		
	for r in np.arange(maxr):
		#Approximate the matrix using HOSVD
		rank = tuple((r+1)*np.ones((d,), dtype = 'i'))
		H = hosvd(A, rank, method = 'svd')
		err_deim[r], mode = lowrank_to_id(A, H, rank, method = 'deim')
		deim_mode_err[r,:] = mode
		err_dime[r], mode = lowrank_to_id(A, H, rank, method = 'dime')
		dime_mode_err[r,:] = mode
		err_rrqr[r], mode = lowrank_to_id(A, H, rank, method = 'rrqr')
		rrqr_mode_err[r,:] = mode


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
	
	plt.semilogy(np.arange(maxr)+1, err_deim, 'k-', linewidth = 3.)
	plt.semilogy(np.arange(maxr)+1, err_dime, 'c-', linewidth = 2.)
	plt.semilogy(np.arange(maxr)+1, err_rrqr, 'r--', linewidth = 2.)
	plt.legend(('DEIM', 'PQR', 'RRQR'), loc = 3)
	plt.xlabel('Rank [r]', fontsize = 18.)
	plt.ylabel('rel. err.' , fontsize = 18.)
	plt.title('Relative Error', fontsize = 24)
	plt.savefig('figs/deim_' + fname + '.png')	

	return	




if __name__ == '__main__':
	K = imaging_tensor_3d()
	test1(K, fname = 'imaging_3d')
	#test2(K, fname = 'imaging_3d')
	K = imaging_tensor_5d()
	test1(K, maxr = 7, fname = 'imaging_5d')
	#test2(K, maxr = 7, fname = 'imaging_5d')
	plt.show()
