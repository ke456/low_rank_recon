#include "agent.h"
#include <iostream>
#include <stdlib.h>
#include <sstream>
#include <fstream>
const double epsilon = 0.0001;

/*********************************************/
/* SETTERS                                   */
/*********************************************/
void agent::set_budget(double b) { budget = b;}
void agent::set(Tuple *t, Tuple *tr){
  cur_cost = 0;
  cur = t;
  cur_true = tr;
} 
void agent::set_costs(vector<double> *c){ costs = c;}

tree_agent::tree_agent(Tree *t){tree = t;}

/***********************************************/
/* Next actions                                */
/***********************************************/
bool tree_agent::next_action(){
  auto S = tree->suggest_next_feature(*cur);
  if (S.feature != 1 && (*cur)[S.feature] == -1 && cur_cost+(*costs)[S.feature] <= budget){
    // we are allowed to take the action that the cbct suggests
    (*cur)[S.feature] = (*cur_true)[S.feature];
    cur_cost += (*costs)[S.feature];
    last_updated = S.feature;
    return true;
  }else{
    // find the feature that we are allowed to update
    int first = 0;
    for (; first < (*cur).size(); first++){
      if ((*cur)[first] == -1 && (*costs)[first] + cur_cost <= budget + epsilon){
	cur_cost += (*costs)[first];
	(*cur)[first] = (*cur_true)[first];
	last_updated = first;
	return true;
      }
    }
  }
  return false;
}

bool random_agent::next_action(){
  vector<int> available_actions;
  for (int i = 0; i < (*cur).size(); i++){
    if ((*cur)[i] == -1 && cur_cost+(*costs)[i] <= budget + epsilon)
      available_actions.emplace_back(i);
  }
  if (available_actions.size() == 0)
    return false;
  int ind = available_actions[rand() % available_actions.size()];
  (*cur)[ind] = (*cur_true)[ind];
  cur_cost += (*costs)[ind];
  last_updated = ind;
  return true;
}

trained_agent::trained_agent(string fname){
  ifstream ifs{fname};
  string line;
  while (getline(ifs,line)){
    vector<int> v;
    istringstream iss{line};
    int s;
    while (iss >> s){
      v.emplace_back(s);
      char c;
      iss >> c;
    }
    steps.emplace_back(v);
  }
  cur_index = -1;
  cur_step = -1;
}

void trained_agent::set_index(int ind){
  cur_index = ind;
  cur_step = 0;
}

bool trained_agent::next_action(){
  if (cur_step < steps[cur_index].size() ){
    int ind = steps[cur_index][cur_step++];
    cout << "ind: " << ind << endl;
    (*cur)[ind] = (*cur_true)[ind];
    cur_cost += (*costs)[ind];
    last_updated = ind;
    return true;
  }
  cur_index = -1;
  return false;
  
}

trained_agent::~trained_agent(){
}









