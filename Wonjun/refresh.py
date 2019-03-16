import sys
import time
import math
import numpy as np
from numpy import linalg as LA
import sympy as sp
from scipy.linalg import expm, inv
import multiprocessing as mp
from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool 
import pickle

class CD:
	def __init__(self, small_t, T):
		self.T = T
		self.small_t = small_t

		self.M = np.matrix([[0, 1], 
							[-2,-3]])

		self.N_C = np.matrix([[0.5,  0], 
							  [  0,0.5]])

		self.N_D = np.matrix([[  0,  0], 
							  [  0,  0]])

		self.Nt = 10
		self.ds = 1.0 * self.small_t / self.Nt

		self.sigma = 0.01
		self.count = 0

		self.maxit = 500
		self.dim = 2
		self.eps = 0.5e-7

		self.AiT_array = np.zeros((2*self.Nt,2))
		for j in range(self.Nt):
			self.AiT_array[j*2:j*2+2,:] = (-np.matmul(expm(-(self.T - j * self.ds) * self.M) , self.N_C)).T

		self.sum_matrix = np.zeros((int(self.Nt),self.Nt*2))
		for j in range(int(self.Nt)):
			self.sum_matrix[j,2*j] = 1
			self.sum_matrix[j,2*j+1] = 1

	def dFi(self, p, z, i):

		assert((i==0)|(i==1))

		if i == 0:
			piplus = np.matrix([[p[0,0]+self.sigma],
								[p[1,0]]])
		else:
			piplus = np.matrix([[p[0,0]],
								[p[1,0]+self.sigma]])

		Jpart = self.Jstar(piplus) - self.Jstar(p)

		Hpart1 = np.matmul(self.AiT_array, piplus)
		Hpart2 = np.matmul(self.AiT_array, p)

		# self.count = 1

		if self.count == 0:
			print(Hpart1.shape)

		Hpart1 = np.matmul(self.sum_matrix, np.power(Hpart1,2))
		Hpart2 = np.matmul(self.sum_matrix, np.power(Hpart2,2))

		if self.count == 0:
			print(Hpart1.shape)

		Hpart1 = np.power(Hpart1, 0.5)
		Hpart2 = np.power(Hpart2, 0.5)


		if self.count == 0:
			print(Hpart1.shape)

		self.count += 1



		Hpart = np.sum(Hpart1 - Hpart2)

		if self.count < 100:
			print(i,Hpart)

		# if self.count < 5:
		# 	print("Jpart")
		# 	print(Jpart)
		# 	print("Hpart")
		# 	print(Hpart)
		# 	print("p")
		# 	print(p)
		# 	print(piplus)
		# 	print((Jpart + Hpart)/self.sigma - z[i,0])
		# 	self.count += 1

		return (Jpart + Hpart * self.ds)/self.sigma - z[i,0]

	def Jstar(self, p):
		return 0.5 + 0.5 * (p[0,0]*p[0,0] + 4.0/25.0 * p[1,0] * p[1,0])

	def calculate(self, pinit, z):
		count = 0
		L = 5.0
		alpha = 1.0/L

		stop = False
		p = pinit

		for k in range(self.maxit):
			for i in range(self.dim): # dimension is always 2 for this problem
				dp = - alpha * self.dFi(p,z,i)
				p[i,0] += dp

				# print(dp)

				if abs(dp) > self.eps:
					count = 0
					if k == self.maxit-1:
						k = 0
						alpha /= 2
						L *= 2
				else:
					count += 1

				if count == self.dim:
					stop = True
					break
			if stop:
				break

print("HERE WE GO")
T = 0.7
small_t = 0.1
N = 60

cd = CD(small_t, T)

pool = Pool(4)

args = []
for i in range(N):
	print("i=",i)
	for j in range(N):

		ii = -3 + i*0.1
		jj = -3 + j*0.1

		
		z = np.matrix([[ii],[jj]])
		pinit = np.matrix([[2],[-0.2]])
		args.append([pinit, z])

		# t = time.time()
		# cd.calculate(pinit, z)
		# print(time.time() - t)
	# 	break
	# break

t = time.time()
r = pool.starmap(cd.calculate, args)
print(time.time() - t)
