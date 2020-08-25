#ifndef __UTILITY_H__
#define __UTILITY_H__

#include <vector>

typedef std::vector<double> Tuple;
typedef std::vector<Tuple> Table;

void normalize(Table &T);
void print(const Tuple &t);
void print(const Table &T);
double distance_l2(const Tuple &v1, const Tuple &v2);
double distance_l2_known(const Tuple &v1, const Tuple &v2);
std::vector<int> known_features(const Tuple &t);

#endif
