'''
Generates data with n clusters, where
every each cluster is labelled with 1. 
The labelled clusters are linearly
separable with the line y = x, and each point
is spread from a randomly chosen centroid with 
a random Gaussian spread of mu = 0, sigma = 0.3
'''

import numpy as np
import random as rand

rand.seed()

def gen(n = 4,x_lim=20,y_lim=20, n_gen=20):
	mu, sigma = 0, 0.3
	centroids = [(rand.uniform(0,x_lim),y_lim/2) for i in range(n)]
	for i in range(len(centroids)):
		p = centroids[i]
		centroids[i] = (p,1)
	points = []
	labels = []
	for c in centroids:
		x_add = [0 for i in range(n_gen)]
		y_add = np.random.normal(mu,sigma,n_gen)
		p = [ [c[0][0]+x_add[i], c[0][1]+y_add[i]] for i in range(n_gen)]
		l = [ c[1] for i in range(n_gen)]
		points += p
		labels += l
	return points,labels