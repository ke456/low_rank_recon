#include <fstream>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include "tree.h"
#include "utility.h"
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



void run_test(const Table &training, const Table &test, const Tuple &costs){
  vector<int> indices;
  for (int i = 0; i < training.size(); i++)
    indices.emplace_back(i);
  Tree cbct{training, indices};
  cbct.split();
  cbct.prune_height(9);

  vector<double> dist;
  vector<double> cost;
  for (int f= 0; f < training[0].size(); f++){
    dist.emplace_back(0);
    cost.emplace_back(0);
  }
  for (int i = 0; i < test.size(); i++){
    // set everything to be unknown
    auto t = test[i];
    for (auto &i : t) i = -1;
    int nf = known_features(t).size();
    
    int at = 0;
    // cbct test
    while (nf != t.size()){
      // find the next feature to update
      auto S = cbct.suggest_next_feature(t);
      if (S.feature == -1){ // just update the next unknown feature
	int first = 0;
	for (; first < t.size(); first++)
	  if (t[first] == -1) break;
	t[first] = test[i][first];
      }else{
	t[S.feature] = test[i][S.feature];
      }

      nf = known_features(t).size();
      // find the cluster with highest conf
      auto CS = cbct.suggest_clusters(t);
      auto *max_cluster = &CS[0];
      for (auto &c : CS){
	if (c.conf > max_cluster->conf)
	  max_cluster = &c;
      }
      dist[at++] += max_cluster->T->knn_avg_dist(t, test[i], 5);
    }
  }
  cout << "res:";
  for (auto & d : dist)
	  cout << " " << d;
  cout << endl;
}

int main(int argc, char *argv[]){
  Table total_data;
  srand(time(NULL));
  read(total_data, argv[1]);
  random_shuffle(total_data.begin(), total_data.end());
  normalize(total_data);

  const double prop = 0.8;
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

  int n = total_data[0].size();
  for (long i = 0; i < n; i++)
    costs.emplace_back(1.0/n);
  
  run_test(training, test, costs);
}
