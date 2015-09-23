import numpy as np
from sktensor import dtensor

__all__ = ['matricize', 'kron']

def matricize(T):
	assert isinstance(T, dtensor)
	return [T.unfold(i) for i in range(T.ndim)]

def kron(mat, reverse = False):
	matlst = mat 
	if isinstance(mat, tuple):
		matlst = list(mat)

	matorder = np.arange(len(mat)) if not reverse else np.arange(len(mat))[::-1]

	prod = matlst[matorder[0]]
	for ind in matorder[1:]:
		prod = np.kron(prod,matlst[ind])

	return prod


if __name__ == '__main__':

	lst = [np.random.randn(i,i) for i in range(1,4)]

	A = kron(lst)
	B = np.kron(lst[0],np.kron(lst[1],lst[2]))
	print np.allclose(A,B)

	A = kron(lst, reverse = True)
	B = np.kron(lst[2],np.kron(lst[1],lst[0]))
	print np.allclose(A,B)
