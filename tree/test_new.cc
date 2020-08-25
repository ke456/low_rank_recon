#include <fstream>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include "tree.h"
#include "utility.h"
#include <cmath>
using namespace std;

void run_test(cost Table &training, const Table &test, const Tuple &costs){
  vector<int> indices;
  for (int i = 0; i < training.size(); i++)
    indices.emplace_back(i);
  Tree cbct{training, indices};
  cbct.split();
  cbct.prune_height(6);

  vector<double> dist;
  vector<double> cost;
  for (int f= 0; f < training[0].size(); f++){
    dist.emplace_back(0);
    cost.emplace_back(0);
  }


}

int main(int argc, char *argv[]){
  Table total_data;
  srand(time(NULL));
  read(total_data, argv[1]);
  random_shuffle(total_data.begin, total_data.end());
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

  n = total_data[0].size();
  for (long i = 0; i < n; i++)
    costs.emplace_back(1.0/n);
  
  run_test(training, test, costs);
}
