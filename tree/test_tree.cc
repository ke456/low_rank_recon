#include <fstream>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include "tree.h"
#include "utility.h"
#include "kmeans.h"
#include <cmath>
using namespace std;

void read(Table &data, const string &fname){
	ifstream ifs{fname};
	string line;
	while (getline(ifs,line)){
		vector <double> v;
		istringstream iss{line};
		double val;
		while (iss >> val){
			v.emplace_back(val);
			char c;
			iss >> c;
		}
		data.push_back(v);
	}
}


const int K = 15;
const int drop = 14;
void run_test(const Table &training, const Table &test, const Tuple &costs){
	vector<int> indices;
	for (int i = 0; i < training.size(); i++){
		indices.emplace_back(i);
	}
	Tree cbct{training, indices};
	cbct.split();
	cbct.prune_height(6);
	KMeans kmeans{training, indices,K};

	// set groups
	vector<vector<int>> groups;
	vector<int> g1,g2,g3;
	g1.emplace_back(4);
	g1.emplace_back(5);
	g2.emplace_back(8);
	g2.emplace_back(9);
	g2.emplace_back(10);
	g3.emplace_back(7);
	g3.emplace_back(12);
	Tree cbct_c{training, indices, costs};
	cbct_c.set_groups(groups);
	cbct_c.split();
	cbct_c.prune_height(5);

	vector<double> dist_t;
	vector<double> dist_k; 
	vector<double> conf_t;
	vector<double> conf_w;
	vector<double> dist_c;
	vector<double> cost_t;
	vector<double> cost_c;
	vector<double> cost_k;
	for (int f = 0; f < training[0].size(); f++){
		dist_t.emplace_back(0);
		dist_k.emplace_back(0);
		conf_t.emplace_back(0);
		conf_w.emplace_back(0);
		dist_c.emplace_back(0);
		cost_t.emplace_back(0);
		cost_c.emplace_back(0);
		cost_k.emplace_back(0);

	}

	const int NUM_WRONG = 0;
	for (int i = 0; i < test.size(); i++){
		auto t = test[i];
		while (t.size() - known_features(t).size() < drop){
			t[rand()%training[0].size()] = -1;
		}
		int nf = known_features(t).size();
		int wrong = 0;
		while (t.size() != nf){
			auto S = cbct.suggest_next_feature(t);
			conf_w[nf] += S.conf;
			if (S.feature == -1){
				int first = 0;
				for (first = 0; first < t.size(); first++)
					if (t[first] == -1)
						break;
				t[first] = 1-test[i][first];
			}else{
				t[S.feature] = 1-test[i][S.feature];
			}

			nf = known_features(t).size();
		}

	}
	for (int i = 0; i < test.size(); i++){
		auto t = test[i];
		while (t.size() - known_features(t).size() < drop){
			t[rand()%training[0].size()] = -1;
		}
		int nf = known_features(t).size();
		auto c = costs;
		// cbct test
		while(t.size() != nf){
			// guess the centroid	
			auto CS = cbct.suggest_clusters(t);
			auto *max_cluster = &CS[0];
			for (auto &c : CS){
				if (c.conf > max_cluster->conf)
					max_cluster = &c;
			}
			dist_t[nf] += max_cluster->T->dist_to_centroid(test[i]);


			// update next feature
			auto S = cbct.suggest_next_feature(t);
			conf_t[nf] += S.conf;

			if (S.feature == -1){
				int first = 0;
				for (first = 0; first < t.size(); first++)
					if (t[first] == -1)
						break;
				t[first] = test[i][first];
			}else{
				t[S.feature] = test[i][S.feature];
				cost_t[nf] += c[S.feature];
				int found = -1;
				for (int i = 0; i < groups.size(); i++){
					auto &g = groups[i];
					if (find(g.begin(), g.end(), S.feature) 
					    != g.end()){
						found = i;
						break;
					}
				}
				c[S.feature] = 0;
				if (found != -1)
					for (int i : groups[found]){
						c[i] = 0;
					}
			}

			nf = known_features(t).size();
		}

		// test cbct with cost
		t = test[i];
		c = costs;
		while (t.size() - known_features(t).size() < drop){
			t[rand()%training[0].size()] = -1;
		}
		nf = known_features(t).size();
		// cbct test
		while(t.size() != nf){
			// guess the centroid
			auto CS = cbct_c.suggest_clusters(t);
			auto *max_cluster = &CS[0];
			for (auto &c : CS){
				if (c.conf > max_cluster->conf)
					max_cluster = &c;
			}
			dist_c[nf] += max_cluster->T->dist_to_centroid(test[i]);


			// update next feature
			auto S = cbct_c.suggest_next_feature(t);
			//conf_t[nf] += S.conf;

			if (S.feature == -1){
				int first = 0;
				for (first = 0; first < t.size(); first++)
					if (t[first] == -1)
						break;
				t[first] = test[i][first];
			}else{
				t[S.feature] = test[i][S.feature];
				cost_c[nf] += c[S.feature];
				int found = -1;
				for (int i = 0; i < groups.size(); i++){
					auto &g = groups[i];
					if (find(g.begin(), g.end(), S.feature) 
					    != g.end()){
						found = i;
						break;
					}
				}
				c[S.feature] = 0;
				if (found != -1)
					for (int i : groups[found]){
						c[i] = 0;
					}
			}
			nf = known_features(t).size();
		}
		t = test[i];
		while (t.size() - known_features(t).size() < drop){
			t[rand()%training[0].size()] = -1;
		}
		nf = known_features(t).size();
		c = costs;
		while (t.size() != nf){
			auto ce = kmeans.find_nearest_centroid(t);
			dist_k[nf] += distance_l2(ce,test[i]);
			auto r = rand()%t.size();
			while (t[r] != -1){
				r = rand()%t.size();
			}
			t[r] = test[i][r];
			cost_k[nf] += c[r];
			int found = -1;
			for (int i = 0; i < groups.size(); i++){
				auto &g = groups[i];
				if (find(g.begin(), g.end(), r) 
				    != g.end()){
					found = i;
					break;
				}
			}
			c[r] = 0;
			if (found != -1)
				for (int i : groups[found]){
					c[i] = 0;
				}
			nf = known_features(t).size();
		}
	}
	for (auto &i : dist_t)
		i /= test.size();
	for (auto &i : dist_c)
		i /= test.size();
	for (auto &i : dist_k)
		i /= test.size();
	for (auto &i : conf_t)
		i /= test.size();
	for (auto &i : conf_w)
		i /= test.size();
	for (auto &i : cost_t)
		i /= test.size();
	for (auto &i : cost_c)
		i /= test.size();
	for (auto &i : cost_k)
		i /= test.size();

	// accumulate costs
	for (int i = 1; i < cost_t.size(); i++)
		cost_t[i] += cost_t[i-1];
	for (int i = 1; i < cost_c.size(); i++)
		cost_c[i] += cost_c[i-1];
	for (int i = 1; i < cost_k.size(); i++)
		cost_k[i] += cost_k[i-1];



	cout << "average distances tree" << endl;
	for (auto i : dist_t)
		cout << i << " ";
	cout << endl;

	cout << "average distances tree with cost" << endl;
	for (auto i : dist_c)
		cout << i << " ";
	cout << endl;

	cout << "average cost_t" << endl;
	for (auto i : cost_t)
		cout << i << " ";
	cout << endl;

	cout << "average cost_c" << endl;
	for (auto i : cost_c)
		cout << i << " ";
	cout << endl;



	
	cout << "average conf correct" << endl;
	for (auto i : conf_t)
		cout << i << " ";
	cout << endl;

	cout << "average conf wrong with " << NUM_WRONG << endl;
	for (auto i : conf_w)
		cout << i << " ";
	cout << endl;

	

	cout << "average distances k" << endl;
	for (auto i : dist_k)
		cout << i << " ";
	cout << endl;

	cout << "average cost_k" << endl;
	for (auto i : cost_k)
		cout << i << " ";
	cout << endl;
}

int main(int argc, char *argv[]){
	Table total_data;
	srand(time(NULL));
	read(total_data, argv[1]);
	random_shuffle(total_data.begin(), total_data.end());
	normalize(total_data);



	const double prop = 0.80;
	int num_train = (int)(prop*total_data.size());
	Table training;
	Table test;
	vector<double> costs;

	for (long i = 0; i < total_data.size(); i++){
		if (i < num_train)
			training.emplace_back(total_data[i]);
		else
			test.emplace_back(total_data[i]);
	}

	/*
	double c[13] = {0.006666667, 0.006666667, 0.006666667,0.006666667,
		        0.048466667,0.034666667,0.103333333,0.686,
			0.582,0.582,0.582,0.672666667,0.686};
	*/
	for (long i = 0; i < total_data[0].size(); i++)
		costs.emplace_back(0.8);
	

	run_test(training, test, costs);
}








