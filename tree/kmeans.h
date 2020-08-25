#ifndef __KMEANS_H__
#define __KMEANS_H__
#include "utility.h"

class KMeans{
	const Table &data;
	const std::vector<int> indices;
	const int K; // number of clusters

	std::vector<int> cluster_label; // same order and indices
	std::vector<Tuple> centroids;
	
	void update_label();
	void update_centroids();
	void EM();

	double compute_total_energy();
public:
	KMeans(const Table &T, const std::vector<int> &ind, int K);
	Tuple find_nearest_centroid(const Tuple &t);

};

#endif
