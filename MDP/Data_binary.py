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
        self.ranks = {}
        self.true_ranks = []
        self.validation = []
    
    def loadfile(self, fname):
        self.data = [] # list of Tuples
        temp_data = [] # just the data portion
        f = open(fname, 'r')
        lines = f.readlines()
        ID = 0
        for l in lines:
            s = l.split(',')
            val = []
            to_append = True
            for sv in s:
                val.append(float(sv))
                if -1 in val:
                    to_append = False
                    break
            if to_append:
                temp_data.append(val)
        rand.shuffle(temp_data)
        for d in temp_data:
            self.data.append((ID,d))
            ID += 1
        self.costs = [ 0 for i in range(len(self.data[0][1]))]
        self.action_space = len(self.data[0][1])+1
        self.observation_space = (self.action_space,)
        self.validation = [i for i in range(len(self.data))]
            
    def get(self, key):
        t = (copy.copy(self.data[key][0]), copy.copy(self.data[key][1])+[nan])
        return (t[0], bool_feature(t[1]), t[1])
    
    def set_validation(self, n):
        self.validation = [i for i in range(n)]
    
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
            if prob < self.unknown_rate:
            #if True:
                datapoint[2][index] = nan
        datapoint[2][-1] = nan
        return (datapoint[0], bool_feature(datapoint[2]), datapoint[2])
        
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
                datapoint[2][index] = nan
        datapoint[2][-1] = nan
        return (datapoint[0], bool_feature(datapoint[2]), datapoint[2])
            
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
        p[2][f] = self.data[p[0]][2][f]
        
    def next_state(self, state, action):
        c = 0
        if action != -1 and action < len(state[1])-1:
            c = self.costs[action]
            
        ns = (copy.copy(state[0]), copy.copy(state[1]), copy.copy(state[2]))
        ns[1][-1] = ns[1][-1] +  c
        
        if action != -1 and action < len(state[2])-1:
            ns[2][action] = self.data[state[0]][1][action]
            ns[1][action] = 1
            group = []
            for g in self.groups:
                if action in g:
                    group = g
            for a in group:
                ns[2][a] = self.data[state[0]][1][a]
                ns[1][a] = 1
                
        return ns
    
    def step(self, state, action):
        next_state = self.next_state(state, action)
        
        reward = 0
        if action != -1 and action < len(state[1])-1:
            reward = self.alpha * -self.costs[action]
        
        #done = ((score[0] <= self.tau) and score[1]) or (next_state[2] > self.max_cost)
        done = (state[1][-1] > self.max_cost) or (nan not in state[2]) or (action >= len(state[2])-1)
        #done = (nan not in state[1]) or (action >= len(state[1])-1)
        r = 0
        
        if done:
            score = self.score(next_state)/self.max_error
            r += (1-score)*100
        reward += (1-self.alpha) * r
        # Last param is info
        return next_state, reward, done, 0
    
    def bool_feature(self, p):
        val = p
        res = [1 for i in range(len(val))]
        for i in range(len(val)):
            if (isnan(val[i])):
                res[i] = 0
        return res
    
    def actions(self, t):
        res = []
        for i in range(len(t[2])-1):
            if isnan(t[2][i]) and (self.costs[i] + t[1][-1] <= self.max_cost):
                #print("action:",i, "utdc:", self.costs[i] + t[1][-1])
                res.append(i)
        res.append(len(t[2])-1)
        return res
    
    def get_mask(self, t):
        return copy.copy(t[1])
    
    def sample_action(self, t):
        act = self.actions(t)
        if (len(act) == 1):
            return len(t[2])-1
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
        self.compute_true_ranks()
        
    def compute_true_ranks(self):
        cent = self.clustering.cluster_centers_
        L = len(cent)
        for d in self.data:
            p = d[1]
            dists = [ (np.linalg.norm( cent[i] - p, 2),i) for i in range(L) ]
            dists.sort()
            ranks = [dists[i][1] for i in range(L) ] 
            self.true_ranks.append(ranks)
        
         
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
        if len(self.validation) > len(self.data):
            self.validation = [i for i in range(len(self.data))]
        temp.validation = [i for i in range(len(temp.data))]
        temp.compute_true_ranks()
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
            return self.rank_all(t)
    
    def rank_all(self,t):
        rank = 0
        features = t[1]
        if str(features) in self.ranks:
            return self.ranks[str(features)]
        else:
            #if (len(self.validation) != len(self.data)):
            #    print("validation size:", len(self.validation))
            for i in self.validation:
                cur = self.get(i)
                for j in range(len(cur[1])):
                    if (features[j] == 0):
                        cur[2][j] = nan
                rank += self.rank(cur)[0]
            val = rank/len(self.validation)
            self.ranks[str(features)] = val
            return val
        
    def empty(self, p):
        for i in p:
            if not isnan(i):
                return False
        return True
    
    def compute_ranks(self, t):
        L = len(self.clustering.cluster_centers_)
        p = t[2][:len(t[2])-1]
        # project the centroids by the known features of p
        f = bool_feature(p)
        proj_p = proj(p,f)
        proj_cent = [ proj(self.clustering.cluster_centers_[i],f) for i in range(L) ]
        
        dists = [ (np.linalg.norm( proj_cent[i] - proj_p, 2),i) for i in range(L) ]
        dists.sort()
        ranks = [ dists[i][1] for i in range(L) ]
        return ranks
    
    def rank(self, t):
        
        L = len(self.clustering.cluster_centers_)
        p = t[2][:len(t[2])-1]
        # project the centroids by the known features of p
        f = bool_feature(p)
        proj_p = proj(p,f)
        proj_cent = [ proj(self.clustering.cluster_centers_[i],f) for i in range(L) ]
        
        dists = [ (np.linalg.norm( proj_cent[i] - proj_p, 2),i) for i in range(L) ]
        dists.sort()
        ranks = [ dists[i][1] for i in range(L) ]
        #print("pred:",[dists[i][1] for i in range(len(dists))] )
        
        ranks_true = self.true_ranks[t[0]]
        
        ind = np.array([ ranks.index(i)+1 for i in range(L) ])
        ind_true = np.array([ ranks_true.index(i)+1 for i in range(L)])
        
        if (self.empty(p)):
            return  (self.max_error, False)
        
        # using MSE
        MSE = np.linalg.norm(ind - ind_true,2) / L
        
        return MSE, (ranks_true[0] == ranks[0])
    
    # retrieves K most similar elements to t
    def retrieve(self, ranks, p, K):
        first = ranks.index(0)
        fs = bool_feature(p)
        p_fs = proj(p,fs)
        # measure the distance between the given ranks and ranks of each point in D
        dists = [ (np.linalg.norm( np.array(ranks) - np.array(self.true_ranks[i]),2)+
                   np.linalg.norm( p_fs - proj(self.data[i][1], fs),2), i) for i in range(len(self.data)) ]
        dists.sort()
        
        return [ self.data[dists[i][1]] for i in range(K) ]
    
     # retrieves K most similar elements to t
    def retrieve2(self, ranks, p, K):
        first = ranks.index(0)
        fs = bool_feature(p)
        p_fs = proj(p,fs)
        # measure the distance between the given ranks and ranks of each point in D
        dists = [ (0+
                   np.linalg.norm( p_fs - proj(self.data[i][1], fs),2), i) for i in range(len(self.data)) ]
        dists.sort()
        
        return [ self.data[dists[i][1]] for i in range(K) ]
    
    def nearest_points(self, p, K):
        dists = [ (np.linalg.norm( np.array(p) - np.array(self.data[i]), 2),i) for i in range(len(self.data)) ]
        dists.sort()
        return [ self.data[dists[i][1]] for i in range(K) ]
        
            
            
            
            
            
            
            
            
            
        