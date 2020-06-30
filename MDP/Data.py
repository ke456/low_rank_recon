import copy
from sklearn.cluster import KMeans
import numpy as np
import random as rand

# assumes that the tuple given is (id, val)
def bool_feature(t):
    val = t[1]
    res = [1 for i in range(len(val))]
    for i in range(len(val)):
        if (val[i] == None):
            res[i] = 0
    return res

# wrapper class for the data. Can load the data by
# build an instance with file fname
class Data:
    KMEANS = 0
    
    def __init__(self, tuples = []):
        self.data = copy.copy(tuples)
        self.normalizing_vals = ([], []) # use this to normalize new points
        self.clustering = None
        self.c_num = None
    
    def __get_item__(self, key):
        return self.data(key)
    
    def loadfile(self, fname):
        self.data = [] # list of Tuples
        f = open(fname, 'r')
        lines = f.readlines()
        ID = 0
        for l in lines:
            s = l.split(',')
            val = []
            for sv in s:
                val.append(float(sv))
            self.data.append((ID, val))
            ID = ID+1
            
    def get(self, key):
        return (copy.copy(self.data[key][0]), copy.copy(self.data[key][1]))
            
    def normalize(self):
        num_features = len(self.data[0][1])
        if (len(self.data) == 0):
            return
        for f in range(num_features):
            vals = [ self.data[i][1][f] for i in range(len(self.data)) ]
            max_v = max(vals)
            min_v = min(vals)
            self.normalizing_vals[0].append(max_v)
            self.normalizing_vals[1].append(min_v)
            for d in self.data:
                d[1][f] = (d[1][f]-min_v)/(max_v - min_v)
                
    # updats p with the value of p[f] filled in
    def update(self, p, f):
        p[1][f] = self.data[p[0]][1][f]
        
    def cluster_K_means(self, k):
        d = []
        for p in self.data:
            d.append(p[1])
        self.clustering = KMeans(n_clusters=k).fit(np.array(d))
        self.c_num = self.KMEANS
  
    def split(self, r):
        test_n = int(r*len(self.data))
        rand.shuffle(self.data)
        temp = Data(self.data[:test_n])
        self.data = self.data[test_n:]
        return temp
        
            
            
            
            
            
            
            
            
            
            
        