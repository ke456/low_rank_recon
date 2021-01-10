import copy
from sklearn.cluster import KMeans
import numpy as np
import random as rand
import statistics

nan = -0.0000001

def isnan(v):
    return v == nan

# assumes that the tuple given is (id, val)
def bool_feature(p):
    val = p
    res = [1 for i in range(len(val))]
    for i in range(len(val)):
        if (isnan(val[i])):
            res[i] = 0
    return res

def worst_error(n):
    v1 = np.array([i+1 for i in range(n)])
    v2 = np.array([n-i for i in range(n)])
    return np.linalg.norm(v1-v2)/n

# projects p to the known features f. For example, if only
# x and y are known in (x,y,z), then f = (1,1,0) and 
# proj( (a,b,c) , f) = (a,b)
def proj(p, f):
    res = []
    for i in range(len(f)):
        if f[i] == 1:
            res.append(p[i])
    return np.array(res)

# wrapper class for the data. Can load the data by
# build an instance with file fname
class Data:
    
    # Constants for clustering 'mode'
    KMEANS = 0
    
    def __init__(self, unknown_rate=0.3, tuples = []):
        self.data = copy.copy(tuples)
        self.normalizing_vals = ([], []) # use this to normalize new points
        self.clustering = None
        self.c_num = self.KMEANS
        self.prob_cluster = [] # probably of cluster c appearing
        self.costs = []
        self.action_space = None
        self.observation_space = None
        self.unknown_rate = unknown_rate
        self.tau = 0
        self.max_cost = 0
        self.alpha = 0.3
        self.beta = 1
        self.it = 0
        self.max_error = 0
        self.groups = []
        
    
    def loadfile(self, fname):
        self.data = [] # list of Tuples
        temp_data = [] # just the data portion
        f = open(fname, 'r')
        lines = f.readlines()
        ID = 0
        for l in lines:
            s = l.split(',')
            val = []
            for sv in s:
                val.append(float(sv))
            temp_data.append(val)
        rand.shuffle(temp_data)
        for d in temp_data:
            self.data.append((ID,d))
            ID += 1
        self.costs = [ 0 for i in range(len(self.data[0][1]))]
        self.action_space = len(self.data[0][1])+1
        self.observation_space = (self.action_space,)
            
    def get(self, key):
        return (copy.copy(self.data[key][0]), copy.copy(self.data[key][1])+[0])
    
    # masks the value of t[1][i] to nan
    def mask(self, t, i):
        t[1][i] = nan
    
    def next_element(self):
        if self.it == len(self.data):
            self.it = 0
        datapoint = self.get(self.it)
        self.it += 1
        mask = [rand.uniform(0,1) for i in range(self.action_space)]
        
        # Mask values with unknown
        for index, prob in enumerate(mask):
            #if prob < self.unknown_rate:
            if True:
                datapoint[1][index] = nan
        datapoint[1][-1] = 0
        return datapoint
        
    def reset(self):
        # Returns a random datapoint as state with some unknowns
        r=rand.randint(0, len(self.data)-1)
        datapoint = self.get(r)
        #if self.it == len(self.data):
        #    self.it = 0
        #datapoint = self.get(self.it)
        #self.it += 1
        mask = [rand.uniform(0,1) for i in range(self.action_space)]
        # Mask values with unknown
        for index, prob in enumerate(mask):
            if prob < self.unknown_rate:
                datapoint[1][index] = nan
        datapoint[1][-1] = 0
        return datapoint
            
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
                
    def set_costs(self,costs=[]):
        self.costs = copy.copy(costs)
        n = len(self.data[0][1])
        if len(self.costs) == 0:
            self.costs = [1/n for i in range(n)]
    
    def set_groups(self,groups=[]):
        for g in groups:
            self.groups.append(copy.copy(g))
                
    # updats p with the value of p[f] filled in
    def update(self, p, f):
        p[1][f] = self.data[p[0]][1][f]
        
    def next_state(self, state, action):
        c = 0
        if action != -1 and action < len(state[1])-1:
            c = self.costs[action]
            
        ns = (copy.copy(state[0]), copy.copy(state[1]))
        ns[1][-1] = ns[1][-1] +  c
        
        if action != -1 and action < len(state[1])-1:
            ns[1][action] = self.data[state[0]][1][action]
            group = []
            for g in self.groups:
                if action in g:
                    group = g
            for a in group:
                ns[1][a] = self.data[state[0]][1][a]
                
        return ns
    
    def step(self, state, action):
        next_state = self.next_state(state, action)
        score = self.score(next_state)
        reward = 0
        if action != -1 and action < len(state[1])-1:
            reward = self.alpha * -self.costs[action]
        
        #done = ((score[0] <= self.tau) and score[1]) or (next_state[2] > self.max_cost)
        done = (state[1][-1] > self.max_cost) or (nan not in state[1]) or (action >= len(state[1])-1)
        #done = (nan not in state[1]) or (action >= len(state[1])-1)
        r = 0
        if done:
            r += self.beta*(1-score[0]/self.max_error)
            if score[1]:
                r += (1-self.beta)*1
        reward += (1-self.alpha) * r
        # Last param is info
        return next_state, reward, done, self.score(state)
    
    def get_mask(self, t):
        return bool_feature(t[1])
    
    def bool_feature(self, p):
        val = p
        res = [1 for i in range(len(val))]
        for i in range(len(val)):
            if (isnan(val[i])):
                res[i] = 0
        return res
    
    def actions(self, t):
        res = []
        for i in range(len(t[1])-1):
            if isnan(t[1][i]) and (self.costs[i] + t[1][-1] <= self.max_cost):
                #print("action:",i, "utdc:", self.costs[i] + t[1][-1])
                res.append(i)
        res.append(len(t[1])-1)
        return res
    
    def sample_action(self, t):
        act = self.actions(t)
        if (len(act) == 1):
            return len(t[1])-1
        r = rand.choice(self.actions(t)[:len(act)-1])
        #c = rand.choice([r,act[-1]])
        return r
        
    def cluster_K_means(self, k):
        d = []
        for p in self.data:
            d.append(p[1])
        self.clustering = KMeans(n_clusters=k).fit(np.array(d))
        self.c_num = self.KMEANS
        self.prob_cluster = [0 for i in range(k)]
        for label in self.clustering.labels_:
            self.prob_cluster[label] += 1
        for i in range(k):
            self.prob_cluster[i] /= len(self.data)
        self.max_error = worst_error(k)
        print(self.prob_cluster)
         
    # retains r%
    def split(self, r):
        test_n = int((1-r)*len(self.data))
        temp = Data(tuples=self.data[:test_n])
        self.data = self.data[test_n:]
        for i in range(len(self.data)):
            self.data[i] = ( self.data[i][0] - test_n , self.data[i][1] )
        temp.action_space = len(temp.data[0][1])
        temp.observation_space = (temp.action_space,)
        temp.costs = copy.copy(self.costs)
        temp.tau = copy.copy(self.tau)
        temp.clustering = copy.copy(self.clustering)
        temp.unknown_rate = self.unknown_rate
        temp.tau = self.tau
        temp.max_cost = self.max_cost
        temp.alpha = self.alpha
        temp.beta = self.beta
        temp.max_error = self.max_error
        temp.groups = self.groups
        return temp
        
    # REWARD FUNCTIONS
    def reward(self, state, action):
        if self.c_num == self.KMEANS:
            ns = self.next_state(state, action)
            return (1-self.costs[action])*(self.score(state)-self.score(ns))
    
    '''
    This is the top level score function, only use this one as it chooses
    the correct one based on the settings. Score differs from the reward
    of an action as the score is a function of the current state t.
    '''
    def score(self, t):
        
        if self.c_num == self.KMEANS:
            return self.rank(t)
        
    def empty(self, p):
        for i in p:
            if not isnan(i):
                return False
        return True
    
    def rank(self, t):
        
        L = len(self.clustering.cluster_centers_)
        p = t[1][:len(t[1])-1]
        # project the centroids by the known features of p
        f = bool_feature(p)
        proj_p = proj(p,f)
        proj_cent = [ proj(self.clustering.cluster_centers_[i],f) for i in range(L) ]
        
        dists = [ (np.linalg.norm( proj_cent[i] - proj_p, 2),i) for i in range(L) ]
        dists.sort()
        ranks = [ dists[i][1] for i in range(L) ]
        #print("pred:",[dists[i][1] for i in range(len(dists))] )
        
        p_true = self.get(t[0])[1][:len(t[1])-1]
        cent = self.clustering.cluster_centers_
        dists_true = [ (np.linalg.norm( cent[i] - p_true, 2),i) for i in range(L) ]
        dists_true.sort()
        ranks_true = [dists_true[i][1] for i in range(L) ] 
        #print("true:", [dists_true[i][1] for i in range(len(dists))])
        
        num_wrong = 0
        for i in range(len(dists)):
            if not(dists[i][1] == dists_true[i][1]):
                num_wrong += 1
        
        ind = np.array([ ranks.index(i)+1 for i in range(L) ])
        ind_true = np.array([ ranks_true.index(i)+1 for i in range(L)])
        
        if (self.empty(p)):
            return  (self.max_error, False)
        
        # using MSE
        MSE = np.linalg.norm(ind - ind_true,2) / L
        
        return MSE, (ranks_true[0] == ranks[0])
            
            
            
            
            
            
            
            
            
        