import numpy as np
from numpy.linalg import norm, pinv
from scipy.linalg import qr, svd
from lowrank import *
from sktensor import dtensor
from decomp import *
from core import *
from time import time
from scipy.linalg.interpolative import estimate_spectral_norm, estimate_spectral_norm_diff


def create_training_tensor(m = 256, fname = 'training/train.'):
	matlst = []
	for i in range(10):
		mat = np.loadtxt(fname + str(i), delimiter = ',')
		matlst.append(mat)
	
	n = np.min([mat.shape[0] for mat in matlst])
	T = np.zeros((m,n,10), dtype = 'd')
	for i in range(10):
		T[:,:,i] = np.copy(matlst[i][:n,:].T)
	
	return T 

def read_training_tensor(fname = 'training/mnist_all.mat'):
	from scipy.io import loadmat
	mat = loadmat(fname)
	matlst = []
	for i in range(10):
		matt = mat['train'+str(i)]
		matlst.append(matt)

	m = matt.shape[1]
	n = np.min([mat.shape[0] for mat in matlst])
	T = np.zeros((m,n,10), dtype = 'd')
	for i in range(10):
		T[:,:,i] = np.copy(matlst[i][:n,:].T)

	return T

def read_test_tensor(fname = 'training/mnist_all.mat'):
	from scipy.io import loadmat
	mat = loadmat(fname)
	matlst = []
	for i in range(10):
		matt = mat['test'+str(i)]
		matlst.append(matt)

	m = matt.shape[1]
	nsh= [mat.shape[0] for mat in matlst]
	T = np.zeros((m,np.sum(nsh)), dtype = 'd')
	truth = np.zeros((np.sum(nsh),), dtype = 'd')


	count = 0
	for i in range(10):
		T[:,count:count+nsh[i]] = np.copy(matlst[i].T)
		truth[count:count+nsh[i]] = i
		count += nsh[i]
	return T, truth


def hosvd_classifier(A, test, truth, err= False, m = 32, n = 32, k = 15):
	start = time()
	T = hosvd(A, rank = (m,n,10), method = 'svd') 
	print "Time for computing HOSVD is %g" %(time()-start)		

	if err:
		am = A.unfold(0)
	 	tm = T.G.ttm(T.U).unfold(0)	
		print norm(tm-am)/norm(am)
	

	U, V = T.U[0], T.U[1]
	F = A.ttm([U,V], mode = [0,1], transp = True)
	
		
	#transformed test data
	ttd = np.dot(U.conj().T,test)	

	#Create basis for each class
	Qlst = []
	for c in range(10):
		q, _, _ = svd(F[:,:,c])		
		Qlst.append(q[:,:k])
	
	from numpy import dot
	#pass through all the data
	cl  = np.zeros_like(truth)
	start = time()
	for p in np.arange(test.shape[1]):
		dp = ttd[:,p]
		res = [norm(dp-dot(q,dot(q.T,dp))) for q in Qlst]
		cl[p] = np.argmin(np.array(res))
	print "Time for computing classification is %g" %(time()-start)		

	
	#Classification error
	acc = np.extract(truth == cl, truth).size/np.float(truth.size)
	
	return acc


def cur_classifier(A, test, truth, method = 'hoid', qr = 'randsvd', err = False, \
			m = 32, n = 32, k = 15):
	start = time()
	if method == 'hoid':
		T = hoid(A, rank = (m,n,10), method = qr) 
	elif method == 'sthoid':
		T = sthoid(A, rank = (m,n,10), method = qr) 
	print "Time for computing HOID is %g" %(time()-start)		

	
	if err:
		am = A.unfold(0)
	 	tm = T.G.ttm(T.U).unfold(0)	
		print norm(tm-am)/norm(am)
	
	
	U, V = T.U[0], T.U[1]
	F = A.ttm([pinv(U, 1.e-8),pinv(V, 1.e-8)], mode = [0,1])
		
	

	#Create basis for each class
	Qlst = []
	for c in range(10):
		q, _, _ = svd(F[:,:,c])		
		Qlst.append(q[:,:k])
	
	from numpy import dot
	#pass through all the data
	cl  = np.zeros_like(truth)
	start = time()
	#transformed test data
	ttd = np.dot(pinv(U, 1.e-8),test)	
	for p in np.arange(test.shape[1]):
		dp = ttd[:,p]
		res = [norm(dp-dot(q,dot(q.T,dp))) for q in Qlst]
		cl[p] = np.argmin(np.array(res))
	print "Time for computing classification is %g" %(time()-start)		

	
	#Classification error
	acc = np.extract(truth == cl, truth).size/np.float(truth.size)
	
	return acc


if __name__ == '__main__':
	#Create training tensor
	A = read_training_tensor()
	A = dtensor(A);	

	#get test data	
	test, truth = read_test_tensor()
	
	#acc = hosvd_classifier(A, test, truth, err = True, m = 64, n = 142, k = 15)
	#print acc	

	#acc = cur_classifier(A, test, truth, err = True, m = 64, n = 142, k = 30)
	#print acc
	acc = cur_classifier(A, test, truth, err = True, m = 64, n = 142, k = 30, qr = 'rid')
	print acc

	#acc = cur_classifier(A, test, truth, err = True, \
	#		m = 64, n = 142, k = 30, method = 'sthoid')
	#print acc	

