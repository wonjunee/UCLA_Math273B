import pickle
import matplotlib.pyplot as plt
import matplotlib.mlab as ml
from matplotlib import animation
import pandas as pd
import sys
import numpy as np
import csv

XX = []
YY = []
ZZ = []

N = 60

frames = 14

ex = "3"

for i in range(1,frames+1):
	data = []
	with open("{}file{}.dat".format(i, ex)) as F:
		csv_reader = csv.reader(F, delimiter=',')
		for row in csv_reader:
			data.append(list(map(lambda x: float(x), row)))

	# print(data)


	data = np.array(data)

	xx = data[:,0]
	yy = data[:,1]
	pp = data[:,2]


	X = np.zeros((N+1,N+1))
	Y = np.zeros((N+1,N+1))
	Z = np.zeros((N+1,N+1))

	for i in range(N+1):
		for j in range(N+1):
			X[i,j] = xx[(N+1)*i+j]
			Y[i,j] = yy[(N+1)*i+j]
			Z[i,j] = pp[(N+1)*i+j]

	XX.append(X)
	YY.append(Y)
	ZZ.append(Z)
	# fig = plt.figure(figsize=(10,4),facecolor='w')
	# ax1 = fig.add_subplot(1,2,1)
	# ax2 = fig.add_subplot(1,2,2)
	# plt.set_colorbar()

	# ims = []
	# for i in range(7):
	# 	im = ax1.contourf(XX[i],YY[i],ZZ[i])
	# 	im.figure = fig
	# 	ims.append([im])

	# ax1.contour(X,Y,Z)
	# ax2.pcolormesh(X,Y,Z, cmap='RdBu')

	# plt.show()



fig = plt.figure(figsize=(10,4),facecolor='w')
ax1 = fig.add_subplot(1,2,1)
ax2 = fig.add_subplot(1,2,2)
# plt.set_colorbar()

# ims = []
# for i in range(7):
# 	im = ax1.contourf(XX[i],YY[i],ZZ[i])
# 	im.figure = fig
# 	ims.append([im])

def animate(i):
	ax1.clear()
	# ax1.set_title("small t = 0.{}".format(i+1))

	c1 = ax1.contour(XX[i],YY[i],ZZ[i])
	c2 = ax2.pcolormesh(XX[i],YY[i],ZZ[i], cmap='RdBu')
	return c1,c2,

# call the animator.  blit=True means only re-draw the parts that have changed.
# anim = animation.ArtistAnimation(fig, ims,interval=7, blit=False)
anim = animation.FuncAnimation(fig, animate,
                               frames=frames, interval=frames, blit=False)

anim.save('example{}.mp4'.format(ex),fps=5, writer="ffmpeg")
