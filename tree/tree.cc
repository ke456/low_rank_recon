#include "tree.h"
#include <iostream>
#include <cmath>
#include <iomanip>
#include <algorithm>

using namespace std;

double Tree::alpha = 1;
double Tree::beta = 1;
double Tree::gamma = 1;
const int WIDTH = 5;
const int PREC = 2;

void Tree::set_hyper_param(const double a, const double b, const double c){
  alpha = a;
  beta = b;
  gamma = c;
}

// utility
void print_vec(const vector<double> &vec){
  for (auto &d : vec)
    cout << setprecision(PREC) << setw(WIDTH) << d << " ";
  cout << endl;
}

void Tree::print_centroid(){
  print_vec(centroid);
}

double Tree::sim(const Tuple &t){	
  double avg_dist = 0;
  double min_dist = distance_l2_known(data[indices[0]],t);
  for (auto i : indices){
    double dist = distance_l2_known(data[i],t);
    avg_dist += dist;
    if (dist < min_dist) min_dist = dist;
  }
  avg_dist /= indices.size();
  if (avg_dist == 0){
    return INFINITY;
  }
  return 1 / avg_dist;

}

void Tree::print_data(){
  for (auto &ind : indices)
    print_vec(data[ind]);
}

// ctor + dtor
Tree::Tree(const Table &data, const vector<int> &ind):
  data{data},
  indices{ind}
{
  centroid = compute_centroid(indices);
  //assumes that all tests costs nothing
  for (int i = 0; i < data[0].size(); i++)
    costs.emplace_back(0);
}

Tree::Tree(const Table &data, const vector<int> &ind, const vector<double> &costs):
  data{data},
  indices{ind},
  costs{costs}
{
  centroid = compute_centroid(indices);
}

Tree::~Tree(){
  delete left;
  delete right;
}

// helpers
vector<double> Tree::compute_centroid(const vector<int> &ind){
  vector<double> centroid;
  const int tuple_size = data[0].size();
  const int num_tuples = ind.size();

  for (int i = 0; i < tuple_size; i++)
    centroid.emplace_back(0);

  for (auto &i : ind){
    for (int c = 0; c < tuple_size; c++)
      centroid[c] += data[i][c];
  }

  for (auto &d : centroid)
    d /= num_tuples;
  return centroid;
}

vector<double> Tree::range(const int feature){
  vector<double> res;
  double min = data[indices[0]][feature];
  double max = data[indices[0]][feature];
  for (auto &i : indices){
    if (data[i][feature] > max)
      max = data[i][feature];
    if (data[i][feature] < min)
      min = data[i][feature];
  }
  res.emplace_back(min);
  res.emplace_back(max);
  return res;
}

// given the range r, evenly divides the space into n partitions
vector<double> linspace(const vector<double> &r, const int n=100){
  vector<double> res;
  for (long i = 0; i < n; i++)
    res.emplace_back(r[0] + i*(r[1]-r[0])/(n-1));
  return res;
}

void Tree::set_groups(const vector<vector<int>> &g){
  groups = g;
}

void Tree::update_cost(const int f, vector<double> &costs){
  int found = -1;
  costs[f] = 0;
  for (int i = 0; i < groups.size(); i++){
    auto &group = groups[i];
    if (find(group.begin(), group.end(),f) != group.end()){
      found = i;
      break;
    }
  }
  if (found != -1)
    for (auto ind : groups[found])
      costs[ind] = 0;
}

void Tree::split(){
  int best_feature;
  double best_b,max_gain;
  find_split(best_feature, best_b, max_gain);	

  // partition according to the best feature/boundary pair
  vector<int> left;
  vector<int> right;
  for (auto i : indices){
    auto &d = data[i][best_feature];
    if (d > best_b) right.emplace_back(i);
    else left.emplace_back(i);
  }
  if (true){
    auto new_costs = costs;
    //new_costs[best_feature] = 0; //we already have feature
    update_cost(best_feature,costs);
    split_feature = best_feature;
    split_gain = max_gain;
    split_b = best_b;
    if (left.size() > 10 && right.size() > 10){
      this->left = new Tree(data, left, new_costs);
      this->left->split();
      this->right = new Tree(data, right, new_costs);
      this->right->split();
    }
  }

}

void Tree::find_split(int &best_feature, double &best_b, double &max_gain){
  auto avg_dist = avg_dist_to_centroid(indices);
  auto num_features = data[0].size();

  best_feature = 0;
  best_b = 0;
  max_gain = 0;
  for (long f = 0; f <  num_features; f++){
    double best_b_feature = 0;
    double max_gain_feature = 0;
    auto r = range(f);
    auto ls = linspace(r,50);
    for (auto &b : ls){
      auto cur_gain = compute_gain(f,b,avg_dist);
      if (cur_gain > max_gain_feature){
	max_gain_feature = cur_gain;
	best_b_feature = b;
      }
    }
    if (max_gain_feature > max_gain){
      best_feature = f;
      max_gain = max_gain_feature;
      best_b = best_b_feature;
    }
  }
}

double Tree::avg_dist_to_centroid(const vector<int> &ind){
  if (ind.size() == 0) return 0;
  auto centroid = compute_centroid(ind);
  double res = 0;
  for (auto &i : ind)
    res += distance_l2(centroid, data[i]);
  return res/ind.size();
}

double Tree::compute_reward(const int feature, const double b, 
    const double avg_dist,
    const vector<int> &left,
    const vector<int> &right){
  auto ldist = avg_dist_to_centroid(left);
  auto rdist = avg_dist_to_centroid(right);
  double p = (1.0*left.size()) / indices.size();
  return avg_dist - (p*ldist + (1-p)*rdist);
}

double Tree::compute_gain(const int feature, const double b, const double avg_dist){
  vector<int> left;
  vector<int> right;
  for (auto i: indices){
    auto &d = data[i][feature];
    if (d > b) right.emplace_back(i);
    else left.emplace_back(i);
  }
  auto reward = compute_reward(feature, b, avg_dist, left, right);

  return alpha*reward - beta*costs[feature]*reward;
}

void Tree::prune_param(const int num_param){
  vector<int> params_used;
  prune_param_rec(num_param, params_used);
}

// checks if f is in v
bool in(const vector<int> &v, const int f){
  for (auto i : v)
    if (i == f) return true;
  return false;
}

double Tree::dist_to_centroid(const Tuple &t){
  return distance_l2(t,centroid);
}

void Tree::prune_param_rec(const int num_params, vector<int> params_used){
  if (split_feature == -1) return; // at a leaf
  if (num_params == 0){
    // we check that the parameters used for children are 
    // already used
    if (in(params_used, split_feature)){
      if (left != nullptr) 
	left->prune_param_rec(0, params_used);
      if (right != nullptr)
	right->prune_param_rec(0, params_used);
    }else{ // we've used up all our params
      delete left;
      delete right;
      left = nullptr;
      right = nullptr;
      // turn this into a leaf
      split_feature = -1;
      split_b = 0;
      split_gain = 0;
    }
  }else{
    if (in(params_used, split_feature)){
      if (left != nullptr)
	left->prune_param_rec(num_params,
	    params_used);
      if (right != nullptr)
	right->prune_param_rec(num_params,
	    params_used);
    }else{
      params_used.emplace_back(split_feature);
      if (left != nullptr)
	left->prune_param_rec(num_params-1,
	    params_used);
      if (right != nullptr)
	right->prune_param_rec(num_params-1,
	    params_used);
    }
  }
}

void Tree::prune_height(const int max_height){
  if (left == nullptr && right == nullptr){
    split_feature = -1;
    split_b = 0;
    split_gain = 0;
  }
  if(max_height == 0){ // just the node itself
    delete left;
    delete right;
    left = right = nullptr;
    split_feature = -1;
    split_b = 0;
    split_gain = 0;
  }else{
    if (left != nullptr)
      left->prune_height(max_height-1);
    if (right != nullptr)
      right->prune_height(max_height-1);
  }	
}

void Tree::print_inorder(){
  cout << "num tuples: " << indices.size() << endl;
  cout << "(features, b, gain) = (" << split_feature << ", "
    << split_b << ", " << split_gain << ')' << endl;
  cout << "cur avg dist: " << avg_dist_to_centroid(this->indices)
    << endl;
  if (left != nullptr) left->print_inorder();
  if (right!= nullptr) right->print_inorder();
}

double Tree::expected_distance(const Tuple &t, const int f){
  return 0;
}


Suggestion Tree::suggest_next_feature(const Tuple &t){
  auto k = known_features(t);
  auto f = first_unknown(t,k);

  vector<Tree *> N_A;
  compute_N_A(N_A, t, k, f);

  double D_total = 0;
  for (auto p : N_A){
    D_total += p->indices.size();
  }

  double s = 0;
  for (auto p : N_A){
    s += (p->indices.size()/D_total)*p->sim(t);
  }
  return Suggestion{f,s};
}

int Tree::first_unknown(const Tuple &t, const vector<int> &k){
  if (k.end() == find(k.begin(), k.end(), split_feature))
    return split_feature;
  if (t[split_feature] > split_b)
    return right->first_unknown(t,k);
  return left->first_unknown(t,k);
}

void Tree::compute_N_A(vector<Tree *> &res, 
    const Tuple &t,
    const vector<int> &k, const int A){
  if (this->split_feature < 0){
    res.emplace_back(this);
  }
  else if (k.end() == find(k.begin(), k.end(), split_feature)){
    res.emplace_back(this);
  }
  else if (split_feature == A){
    left->compute_N_A(res,t,k,A);
    right->compute_N_A(res,t,k,A);
  }
  else{
    if (t[split_feature] > split_b)
      right->compute_N_A(res, t, k, A);
    else
      left->compute_N_A(res,t,k,A);
  }
}

vector<Cluster> Tree::suggest_clusters(const Tuple &t){
  vector<Tree *> N;
  auto k = known_features(t);
  compute_N(N, t, k);

  vector<Cluster> vC;
  double D_total = 0;
  for (auto p : N){
    D_total += p->indices.size();
  }
  for (auto p : N){
    double prob = p->indices.size() / D_total;
    double s = p->sim(t);
    Cluster c{p, prob*s};
    vC.emplace_back(c);
  }
  return vC;
}

void Tree::compute_N(vector<Tree *> &res, const Tuple &t, const vector<int> &k){
  if (split_feature < 0)
    res.emplace_back(this);
  else if (k.end() == find(k.begin(), k.end(), split_feature)){
    left->compute_N(res, t, k);
    right->compute_N(res, t, k);
  }
  else{
    if (t[split_feature] > split_b)
      right->compute_N(res,t,k);
    else
      left->compute_N(res,t,k);
  }
}

double Tree::update_score(const Tuple &t, const int f, const double val){
  auto vC = suggest_clusters(t);
  double max1 = 0;
  for (auto & c : vC){
    if (max1 < c.conf)
      max1 = c.conf;
  }

  auto t2 = t;
  t2[f] = val;
  vC = suggest_clusters(t2);
  double max2 = 0;
  for (auto & c : vC){
    if (max2 < c.conf)
      max2 = c.conf;
  }
  return max2-max1;
}


double Tree::knn_avg_dist(const Tuple &t, const Tuple &t_true, const int k){
  vector<double> dists;
  for (auto &i : indices){
    dists.emplace_back(distance_l2_known(t, data[i]));
  }
  double res = 0;
  vector<int> already_in;
  for (int i = 0; i < k; i++){
    double min_dist = dists[0];
    double ind = 0;
    for (int j = 1; j < dists.size(); j++){
      auto d = dists[j];
      // ensure that we don't add the same min over again
      if (min_dist > d && find(already_in.begin(), already_in.end(), j) == already_in.end() ){
	min_dist = d;
	ind = j;
      }
    }
    /*
    if (known_features(t).size() == 13){
      cout << "t and min" << endl;
      print(t);
      print(data[indices[ind]]);
      cout << "dist: " << distance_l2(t_true, data[indices[ind]]) << endl;
      cout << "mindist: " << min_dist << endl;
    }
    */
    res += distance_l2(t_true, data[indices[ind]]);
    already_in.emplace_back(ind);
    //dists.erase(dists.begin()+ind);
    //if (dists.size() == 0) break;

  }
  return res;
}
























