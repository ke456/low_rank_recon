#include "agent.h"
#include <iostream>
#include <stdlib.h>
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
    return true;
  }else{
    // find the feature that we are allowed to update
    int first = 0;
    for (; first < (*cur).size(); first++){
      if ((*cur)[first] == -1 && (*costs)[first] + cur_cost <= budget){
	cur_cost += (*costs)[first];
	(*cur)[first] = (*cur_true)[first];
	return true;
      }
    }
  }
  return false;
}

bool random_agent::next_action(){
  vector<int> available_actions;
  for (int i = 0; i < (*cur).size(); i++){
    if ((*cur)[i] == -1 && cur_cost+(*costs)[i] <= budget)
      available_actions.emplace_back(i);
  }
  if (available_actions.size() == 0)
    return false;
  int ind = available_actions[rand() % available_actions.size()];
  (*cur)[ind] = (*cur_true)[ind];
  cur_cost += (*costs)[ind];
  return true;
}