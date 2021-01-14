#ifndef __TREE_H__
#define __TREE_H__

#include "utility.h"

// pair <feature, confidence>
struct Suggestion{
	int feature;
	double conf;
};

class Tree;

struct Cluster{
	Tree *T;
	double conf;
};

class Tree{
	// hyper-parameters
	static double alpha;
	static double beta;
	static double gamma;
	
	const double tau = 0.05;

	Tree *left = nullptr;
	Tree *right = nullptr;

	// stores the split values for the next level
	int split_feature = -1; // stays negative if this is a leaf
	double split_gain = 0;
	double split_b = 0;

	std::vector<std::vector<int>> groups;

	// reference to the actual data tuples
	const Table &data;
	// stores the costs of acquire specific features; if a feature
	// has already been acquired, just set the cost of this feature
	// to be zero
	std::vector<double> costs;
	// stores the indices of the points stored
	std::vector<int> indices;
	// the centroid of the points stored in indices
	std::vector<double> centroid;

	// ignores these features when creating the trees
	std::vector<int> ignore;

	// computes the centroid for given indices	
	std::vector<double> compute_centroid(const std::vector<int> &ind);

	// computes the range for the given feature,
	// where res[0] is min, res[1] is max
	std::vector<double> range(const int feature);

	// computes the average distance to the centroid specified by the
	// given indices
	double avg_dist_to_centroid(const std::vector<int> &ind);

	// computes the gain if we split on bound b of the specified feature
	double compute_gain(const int feature, const double b,
			    const double avg_dist);
	double compute_reward(const int feature, const double b,
			      const double avg_dist,
			      const std::vector<int> &left,
			      const std::vector<int> &right);

	void find_split(int &best_feature, double &best_b, double &max_gain);

	void prune_param_rec(const int num_param, std::vector<int> params_used);

	void compute_N_A(std::vector<Tree *> &res,
			const Tuple &t,
			const std::vector<int> &k,
			const int A);

	void compute_N(std::vector<Tree *> &res, 
		       const Tuple &t, 
		       const std::vector<int> &k);

	// returns the first unknown feature of t highest on the tree,
	// -1 if all needed attributes are known
	int first_unknown(const Tuple &t, const std::vector<int> &k);

	void update_cost(const int f, std::vector<double> &costs);
public:
	static void set_hyper_param(const double, const double, const double);

	Tree(const Table &data, const std::vector<int> &ind);
	Tree(const Table &data, const std::vector<int> &ind, 
	     const std::vector<double> &costs);
	~Tree();

	void set_groups(const std::vector<std::vector<int>> &g);
	void set_ignore(const std::vector<int> &ig);
	// utility functions
	void print_centroid(); // prints the centroid
	void print_data(); // prints only the data stored in ind
	void print_inorder();
	double sim(const Tuple &t); // similiarity between t and D
	double dist_to_centroid(const Tuple &t);

	// main function for construction
	void split();

	// some pruning functions
	void prune_param(const int num_param);
	void prune_height(const int max_height);

	// computes the expected average distance of the known features
	// of the tuple t from the centroid at the leaves by 'guessing'
	// an unknown feature f. More precisely, we traverse down using these
	// rules:
	// 0) at 0, return prob(T) * dist(T_c,t) where prob(T) is the probability
	//    of being at the specific leaf and dist(T_c,t) is the average distance
	//    from the centroid of the partition to tuple t
	// 1) if the split feature is known, then ED[T] = ED[T'], where
	//    T' is the tree by following the split
	// 2) if the split feature is unknown, return
	//    ED[T] = prob(T_l) * dist(T_lc, t) + prob(T_r) * dist(T_rc,t)
	// 3) if the split feature is unknown and not f, then treat as a leaf
	double expected_distance(const Tuple &t, const int f);

	// produces the next suggested update for tuple t with a confidence score
	Suggestion suggest_next_feature(const Tuple &t);

	// produces list of suggested clusters and confidence
	std::vector<Cluster> suggest_clusters(const Tuple &t);

	// computes the score of updating feature f with value val
	double update_score(const Tuple &t, const int f, const double val);

	double knn_avg_dist(const Tuple &t, const Tuple &t_true, const int k);

};
#endif
