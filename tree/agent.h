#ifndef __agent_h__
#define __agent_h__

#include "utility.h"
#include "tree.h"
#include <vector>
using namespace std;
class agent{
  protected:
  double budget; // total amount to spend on actions
  
  double cur_cost; // current cost incurred
  Tuple *cur; // current tuple
  Tuple *cur_true; // true current tuple

  vector<double> *costs;


  public:
  // takes the next action and returns true if the 
  // agent was able to take the action under the budget constraints,
  // false otherwise
  virtual bool next_action()=0; 

  // sets the maximum budget
  void set_budget(double b);
  // sets the costs of feature tests
  void set_costs(vector<double> *c);
  // sets the current tuple and resets the incurred costs
  void set(Tuple * cur, Tuple * tr);

};

// class for agent that takes random actions
class random_agent : public agent{

  public:
  bool next_action();
};

// class for agent that uses cbct for actions
class tree_agent : public agent{
  Tree *tree;

  public:
  tree_agent(Tree *t);
  bool next_action();
};


#endif
