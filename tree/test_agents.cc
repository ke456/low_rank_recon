#include <fstream>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include "tree.h"
#include "utility.h"
#include "agent.h"
#include <cmath>
#include <string>
using namespace std;

string add(string s1, string s2){
  ostringstream oss;
  oss << s1 << s2;
  return oss.str();
}

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

void run_test(Table &training, Table &test, Tuple &costs, string to_runs){
  vector<int> indices;
  for (int i = 0; i < training.size(); i++)
    indices.emplace_back(i);
  Tree cbct{training, indices, costs};
  cbct.split();
  cbct.prune_height(9);

  //int num_features = training[0].size();
  int N = 10;
  
  vector<double> dist;
  vector<double> dist_rand;
  vector<double> dist_trained;
  for (int f= 0; f < N; f++){
    dist.emplace_back(0);
    dist_rand.emplace_back(0);
    dist_trained.emplace_back(0);
  }
  
  cout << "starting pre-trained test" << endl;
  for (int bc = 0; bc < N; bc++){
    for (int i = 0; i < test.size(); i++){
      // set everything to be unknown
      auto t = test[i];
      for (auto &h : t) h = -1;

      double budget = (bc+1.0)/N;
      string ap = add(to_string(bc+1), ".0.csv");
      string fname = add(to_runs,ap);
      cout << "at: " << i << ", fname: " << fname << endl;
      trained_agent ta(fname);
      ta.set_costs(&costs);
      ta.set_budget(budget);
      ta.set(&t, &test[i]);
      ta.set_index(i);
      while(ta.next_action()){} 
      dist_trained[bc] += cbct.knn_avg_dist(t, test[i], 5);
    }
  }
  
  cout << "res trained:";
  for (auto & d : dist_trained)
    cout << " " << d;
  cout << endl;
  
  cout << "starting tree test" << endl;
  for (int bc = 0; bc < N; bc++){
    for (int i = 0; i < test.size(); i++){
      // set everything to be unknown
      auto t = test[i];
      for (auto &h : t) h = -1;

      double budget = (bc+1.0)/N;
      tree_agent ta(&cbct);
      ta.set_costs(&costs);
      ta.set_budget(budget);
      ta.set(&t, &test[i]);
      while(ta.next_action()){}
      dist[bc] += cbct.knn_avg_dist(t, test[i], 5);
    }
  }
  
  cout << "res tree:";
  for (auto & d : dist)
    cout << " " << d;
  cout << endl;
  
  cout << "starting random test" << endl;
  for (int bc = 0; bc < N; bc++){
    for (int i = 0; i < test.size(); i++){
      // set everything to be unknown
      auto t = test[i];
      for (auto &h : t) h = -1;

      double budget = (bc+1.0)/N;
      random_agent ta;
      ta.set_costs(&costs);
      ta.set_budget(budget);
      ta.set(&t, &test[i]);
      while(ta.next_action()){} 
      dist_rand[bc] += cbct.knn_avg_dist(t, test[i], 5);
    }
  }
  
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
  
  string test_root = argv[1];
  string training_fn = add(test_root, "_training.csv");
  string test_fn = add(test_root, "_test.csv");
  string costs_fn = add(test_root, "_cost.csv");
  
  cout << "read in files" << endl;

  read(training_data, training_fn);
  //random_shuffle(training_data.begin(), training_data.end());
  read(test_data, test_fn);
  int n = training_data[0].size();

  ifstream cost_file(costs_fn);
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

  run_test(training_data, test_data, costs, argv[2]);
}
