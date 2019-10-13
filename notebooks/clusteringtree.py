import numpy as np
import matplotlib.pylab as plt

def log2(p):
    if (p==0):
        return 0
    else:
        return np.log2(p)

def avg(d):
    # Returns the average of d=List[Integer], 0 if the list is empty
    if (len(d)==0):
        return 0
    return sum(d)/len(d)

def entropy(d):
    # Returns the entropy of cluster d
    return -p1*log2(p1) - p2*log2(p2)

class ClusteringTree:
    def __init__(self, linspace=np.linspace(0,20)):
        self.linspace = linspace
        self.ents=[]
        self.dists=[]
        self.boundary={}
        self.alpha = 0.01
        
    def cluster_multivariables(self, data):
        self.cluster_multivariables_helper(data, self.boundary)
        
    def cluster_multivariables_helper(self, data, btree):
        '''
            Data is a nd matrix.
            All values are assumed to be positive
        '''
        best_val, best_dim, min_ent, best_dist = (0,-1,1,float("inf"))
        
        # loop over all dim.
        for cur_dim in range(data.shape[1]):
            print("cur_dim:",cur_dim)
            ret_tuple = self.cluster_one_variable(data[:,cur_dim])
            if ret_tuple is None:
                continue
            
            if ret_tuple[1] <= min_ent:
                if ret_tuple[2] < best_dist:
                    best_val, min_ent, best_dist = ret_tuple
                    best_dim = cur_dim
                
        if best_dim > -1:
            btree["c"] = ((best_dim, best_val))
            btree["l"] = {}
            btree["r"] = {}
            
            data_slice = data[:, best_dim]
            self.cluster_multivariables_helper(data[data_slice < best_val], btree["l"])
            self.cluster_multivariables_helper(data[data_slice >= best_val], btree["r"])
                
    def cluster_one_variable(self, data):
        '''
            This algorithm only works with one dimension (label, x)
            This will return a single best boundary along the variable x.
            Will return None if there is no split amongst the samples.
        '''
        print('from', min(data), 'to', max(data))
        n = len(data)
        
        max_dist = (min(data) - max(data))**2
        
        best_x = 0
        min_ent = 1
        best_dist = float("inf")

        self.ents = []
        self.dists = []
        
        k = np.ceil(self.alpha*len(data))

        for x in self.linspace:
            d1 = []
            d2 = []
            n_clustered = 0

            # partition
            for d in data:
                if d <= x:
                    d1.append(d)
                else:
                    d2.append(d)

            # compute centroid for each partition
            c1 = avg(d1)
            c2 = avg(d2)

            # compute average energy to each centroid in the partition
            dist_to_c1 = 0
            dist_to_c2 = 0
            for d in d1:
                dist_to_c1 += (d-c1)**2/max_dist
            for d in d2:
                dist_to_c2 += (d-c2)**2/max_dist

            # compare square distances between points and boundary
            # left partition
            for i in range(len(d1)):
                dist = []
                d = d1[i]
                for j in range(len(d1)):
                    dd = d1[j]
                    if i != j:
                        dist.append((dd-d)**2/max_dist)
                if len(dist) != 0:
                    dist_p = min(dist)
                    dist_b = (d-x)**2/max_dist
                    if dist_p < dist_b:
                        n_clustered += 1
                        
            # right partition
            for i in range(len(d2)):
                dist = []
                d = d2[i]
                for j in range(len(d2)):
                    dd = d2[j]
                    if i != j:
                        dist.append((dd-d)**2/max_dist)
                if len(dist) != 0:
                    dist_p = min(dist)
                    dist_b = (d-x)**2/max_dist
                    if dist_p < dist_b:
                        n_clustered += 1

            # compute entropy for this boundary
            p1 = n_clustered/n
            p2 = 1-p1
            entropy = -p1*log2(p1) - p2*log2(p2)
            
            print('x:', x, 'total energy:', (dist_to_c1+dist_to_c2), 'entropy:', entropy, 'k', k)
            
            self.ents.append(entropy)
            self.dists.append((dist_to_c1+dist_to_c2))
            
            if entropy <= min_ent and (dist_to_c1+dist_to_c2) < best_dist:
                min_ent = entropy
                best_x = x
                best_dist = (dist_to_c1+dist_to_c2)
        
        print('best x:', best_x, 'total energy:', best_dist, 'entropy:', min_ent)
        # Split data
        d1 = data[data < best_x]
        d2 = data[data >= best_x]
                
        if len(d1) == 0 or len(d2) == 0:
            return None
        return (best_x, min_ent, best_dist)
        
    def plot_ent_dist(self):
        standardized_dist = self.dists/max(self.dists)
        plt.plot(self.linspace, self.ents, 'r', self.linspace, standardized_dist, 'b')
    
    def plot_boundries2D(self,y_upper_lim = float("inf"),y_lower_lim = float("-inf"),x_right_lim = float("inf"),x_left_lim = float("-inf")):
        self.plot_boundries2D_helper(y_upper_lim, y_lower_lim, x_right_lim, x_left_lim, self.boundary)
        
    def plot_boundries2D_helper(self,y_upper_lim,y_lower_lim,x_right_lim,x_left_lim,btree):
        if "c" not in btree:
            # Empty node case
            return
        dim, split_val = btree["c"]
        
        if dim == 0:
            # Split an a x-value
            plt.plot([split_val for i in range(2)], [y_lower_lim, y_upper_lim],'-')
            self.plot_boundries2D_helper(y_upper_lim, y_lower_lim, min(split_val, x_right_lim),  x_left_lim, btree["l"])
            self.plot_boundries2D_helper(y_upper_lim, y_lower_lim, x_right_lim,  max(split_val, x_left_lim), btree["r"])
        elif dim == 1:
            # Split an a y-value
            plt.plot([x_left_lim, x_right_lim], [split_val for i in range(2)], '-')
            self.plot_boundries2D_helper(min(split_val, y_upper_lim), y_lower_lim, x_right_lim,  x_left_lim, btree["l"])
            self.plot_boundries2D_helper(y_upper_lim, max(split_val, y_lower_lim), x_right_lim,  x_left_lim, btree["r"])
    