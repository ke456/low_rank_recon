#include "kmeans.h"
#include <stdlib.h>
#include <time.h>
#include <cmath>
#include <iostream>
using namespace std;
void KMeans::update_label(){
	for (int i = 0; i < indices.size(); i++){
		double min_dist = distance_l2(centroids[0], data[indices[0]]);
		int min_cluster = 0;
		for (int k = 0; k < K; k++){
			auto &c = centroids[k];
			auto dist = distance_l2(c, data[indices[i]]);
			if (dist < min_dist){
				min_cluster = k;
				min_dist = dist;
			}
		}
		cluster_label[i] = min_cluster;
	}
}

void KMeans::update_centroids(){
	// first set all centroids to be zero
	for (auto &c: centroids){
		for (auto &i : c) i = 0;
	}

	vector<int> nums;
	for (int i = 0; i < K; i++) nums.emplace_back(0);

	for (int i = 0; i < indices.size(); i++){
		auto &t = data[indices[i]];
		auto &c = centroids[cluster_label[i]];
		for (auto j = 0; j < c.size(); j++)
			c[j] += t[j];
		nums[cluster_label[i]]++;
	}
	for (int k = 0; k < K; k++)
		for (auto &ca : centroids[k])
			ca /= max(nums[k],1);
}

double KMeans::compute_total_energy(){
	double res = 0;
	for (int i = 0; i < indices.size(); i++){
		auto label = cluster_label[i];
		auto &t = data[indices[i]];
		auto &c = centroids[label];
		res += distance_l2(t,c);
	}
	return res;
}

void KMeans::EM(){
	vector<double> energies;
	energies.emplace_back(-1);
	energies.emplace_back(compute_total_energy());
	while (abs(energies[energies.size()-1] 
		- energies[energies.size()-2]) > 0.05){
		update_label();
		update_centroids();
		energies.emplace_back(compute_total_energy());
	}
}

KMeans::KMeans(const Table &T, const vector<int>& ind, const int K): 
	data{T}, indices{ind},K{K}{
	for (int k = 0; k < K; k++){
		int r = rand() % indices.size();
		centroids.emplace_back(data[indices[r]]);
	}
	for (int i = 0; i < ind.size(); i++)
		cluster_label.emplace_back(0);
	update_label();
	EM();
}

Tuple KMeans::find_nearest_centroid(const Tuple &t){
	double min_dist = distance_l2_known(t,centroids[0]); 
	int min_k = 0;
	for (int k = 1; k < K; k++){
		auto dist = distance_l2_known(t,centroids[k]);
		if (dist < min_dist){
			min_dist = dist;
			min_k = k;
		}
	}
	return centroids[min_k];
}





