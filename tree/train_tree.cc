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


// This only trains a CBCT and outputs the sequence of feature updates

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

void run_test(Table &training, Table &test, Tuple &costs, string out_root){
  vector<int> indices;
  for (int i = 0; i < training.size(); i++)
    indices.emplace_back(i);
  Tree cbct{training, indices, costs};
  cbct.split();
  cbct.prune_height(9);

  //int num_features = training[0].size();
  int N = 10;
  
  vector<double> dist;
  for (int f= 0; f < N; f++){
    dist.emplace_back(0);
  }

  cout << "starting tree test" << endl;
  for (int bc = 0; bc < N; bc++){
    string fname = add(add(out_root, add("_tree", to_string(bc+1))), ".csv");
    cout << "writing to: " << fname << endl;
    ofstream fout{fname};
    
    for (int i = 0; i < test.size(); i++){
      // set everything to be unknown
      auto t = test[i];
      for (auto &h : t) h = -1;

      vector<int> updates;

      double budget = (bc+1.0)/N;
      tree_agent ta(&cbct);
      ta.set_costs(&costs);
      ta.set_budget(budget);
      ta.set(&t, &test[i]);
      while(ta.next_action()){
        updates.emplace_back(ta.last_updated);
      }
      dist[bc] += cbct.knn_avg_dist(t, test[i], 5);
      for (int i = 0; i < updates.size(); i++){
	fout << updates[i];
	if (i == updates.size()-1)
	  fout << endl;
	else
	  fout << ",";
      } 
    }
  }
  
  cout << "res tree:";
  for (auto & d : dist)
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
