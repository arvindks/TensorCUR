from core import *
import numpy as np
from scipy.linalg import pinvh

from sktensor import khatrirao, ktensor

__all__ = ['CP']

class CP(ktensor):
	def __init__(self, U, lmbda = None):
		ktensor.__init__(self, U, lmbda = lmbda)

	def lowrank_matricize(self):

		U, l = self.U, self.lmbda
		dim  = self.ndim
		
		ulst, vlst = [], []
		L = np.diag(l)
		
		for n in range(dim):

			lst = list(range(n)) + list(range(n + 1, dim))

			utemp = [U[l] for l in lst]
			mat = khatrirao(tuple(utemp), reverse = True).conj().T

			
			ulst.append(np.dot(U[n],L))
			vlst.append(mat)

		return  ulst, vlst

	def matricize(self):	
		"""Use with caution. Can result in large matrices"""
		ulst, vlst = self.lowrank_matricize()
		modelst = [np.dot(u,v) for u,v in zip(ulst,vlst)]

		
		return modelst

