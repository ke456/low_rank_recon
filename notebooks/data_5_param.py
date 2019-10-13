import numpy as np
import random as rand

rand.seed()


'''
Creates some data points using a 'fictitious' generating story about
students getting into university from high school with parameters
(grade, # hours spent in academic extra-curriculars, # hours spent in other ec).
We have the following clusters for those that are accepted:
1) very high grades (with mu = 95, var=1) (100%)
2) high grades (~90), with high hours in both ec (90%)
3) high grades (~90), with high academic ec, low other ec (85%)
4) high grades , with low academic ec, high other ec (70%)
5) okay grades (~85),
'''
def gen(n=100):
    data = []
    labels = []
	# generate (with label 1) data centered around 95 in arg1
    # with normal spreads: 1) [95,1]; 2) [2, 6]; 3) [4,4]
    n_gen = n
    s1 = np.random.normal(97,0.5,n_gen)
    s2 = np.random.normal(2,6,n_gen)
    s3 = np.random.normal(4,4,n_gen)
    # modify data points that cannot exist (like > 100 in arg1)
    for i in range(n_gen):
        labels.append(1)
        d = [s1[i],s2[i],s3[i]]
        if d[0] > 100:
            d[0] = 100
        if d[1] < 0:
            d[1] = 0
        if d[2] < 0:
            d[2] = 0
        data.append(d)
    # cluster 2
    s1 = np.random.normal(90,0.5,n_gen)
    s2 = np.random.normal(10,1,n_gen)
    s3 = np.random.normal(12,1,n_gen)
    # modify data points that cannot exist (like > 100 in arg1)
    for i in range(n_gen):
        labels.append(1)
        d = [s1[i],s2[i],s3[i]]
        if d[0] > 100:
            d[0] = 100
        if d[1] < 0:
            d[1] = 0
        if d[2] < 0:
            d[2] = 0
        data.append(d)
        
    # cluster 3
    s1 = np.random.normal(90,0.5,n_gen)
    s2 = np.random.normal(10,1,n_gen)
    s3 = np.random.normal(2,1,n_gen)
    # modify data points that cannot exist (like > 100 in arg1)
    for i in range(n_gen):
        labels.append(1)
        d = [s1[i],s2[i],s3[i]]
        if d[0] > 100:
            d[0] = 100
        if d[1] < 0:
            d[1] = 0
        if d[2] < 0:
            d[2] = 0
        data.append(d)
        
    # generate data centered around 90 in 
    return data,labels