#include "utility.h"
#include <iostream>
#include <iomanip>
#include <cmath>
using namespace std;

void normalize(Table &T){
	auto num_tuples = T.size();
	if (num_tuples == 0) return;
	auto num_features = T[0].size();

	for (int f = 0; f < num_features; f++){
		// find the max and min
		double max = T[0][f];
		double min = T[0][f];
		for (auto &t : T){
			if (t[f] < 0) continue;
			if (t[f] > max)
				max = t[f];
			if (t[f] < min)
				min = t[f];
		}
		for (auto &t : T){
			if (t[f] < 0) continue;
			auto den = max - min;
			if (den == 0) den = max;
			t[f] = (t[f] - min) / (den);
		}

	}
}

void print(const Tuple &t){
	for (auto &d : t)
		cout << setprecision(2) << setw(5) << d << " ";
	cout << endl;
}

void print(const Table &T){
	for (auto &t : T)
		print(t);
}

double distance_l2(const vector<double> &v1, const vector<double> &v2){
	double res = 0;
	for (long i = 0; i < v1.size(); i++)
		res += pow(v1[i] - v2[i],2);
	return sqrt(res);
}

// l2 distance only using known features
double distance_l2_known(const vector<double> &v1, const vector<double> &v2){
	double res = 0;
	for (long i = 0; i < v1.size(); i++)
		if (v1[i] >= 0 && v2[i] >= 0)
			res += pow(v1[i] - v2[i],2);
	return sqrt(res);
}

vector<int> known_features(const Tuple &t){
	vector<int> inds;
	for (int i = 0; i < t.size(); i++)
		if (t[i] != -1) inds.emplace_back(i);
	return inds;
}

