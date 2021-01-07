#include <fstream>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include "tree.h"
#include "utility.h"
#include "agent.h"
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



void run_test(Table &training, Table &test, Tuple &costs){
  vector<int> indices;
  for (int i = 0; i < training.size(); i++)
    indices.emplace_back(i);
  Tree cbct{training, indices, costs};
  cbct.split();
  cbct.prune_height(9);

  vector<double> dist;
  vector<double> dist_rand;
  for (int f= 0; f < training[0].size(); f++){
    dist.emplace_back(0);
    dist_rand.emplace_back(0);
  }
  int num_features = training[0].size();
  for (int bc = 0; bc < num_features; bc++){
    for (int i = 0; i < test.size(); i++){
      // set everything to be unknown
      auto t = test[i];
      for (auto &h : t) h = -1;

      double budget = (bc+1.0)/num_features;
      tree_agent ta(&cbct);
      ta.set_costs(&costs);
      ta.set_budget(budget);
      ta.set(&t, &test[i]);
      while(ta.next_action()){}
      dist[bc] += cbct.knn_avg_dist(t, test[i], 5);
    }
  }

  for (int bc = 0; bc < num_features; bc++){
    for (int i = 0; i < test.size(); i++){
      // set everything to be unknown
      auto t = test[i];
      for (auto &h : t) h = -1;

      double budget = (bc+1.0)/num_features;
      random_agent ta;
      ta.set_costs(&costs);
      ta.set_budget(budget);
      ta.set(&t, &test[i]);
      while(ta.next_action()){}
      dist_rand[bc] += cbct.knn_avg_dist(t, test[i], 5);
    }
  }

  cout << "res tree:";
  for (auto & d : dist)
    cout << " " << d;
  cout << endl;

 cout << "res rand:";
  for (auto & d : dist_rand)
    cout << " " << d;
  cout << endl;
}

int main(int argc, char *argv[]){
  Table total_data;
  srand(time(NULL));

  Table training_data;
  Table test_data;
  vector<double> costs;

  read(training_data, argv[1]);
  read(test_data, argv[2]);
  int n = training_data[0].size();

  ifstream cost_file(argv[3]);
  string cost_line;
  getline(cost_file, cost_line);
  istringstream iss(cost_line);

  for (long i = 0; i < n; i++){
    char c;
    double co;
    iss >> co >> c;
    costs.emplace_back(co);
  }
  for (auto &i : costs)
    cout << " " << i;
  cout << endl;

  run_test(training_data, test_data, costs);
}
