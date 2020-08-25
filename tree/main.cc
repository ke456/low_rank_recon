#include <iostream>
#include <vector>
#include <sstream>
#include <fstream>
#include "tree.h"

using namespace std;

// assumes that negative values are unknown (thus, data must be
// recentered first)
void read(Table &data, const string &fname){
	ifstream ifs{fname+".csv"};
	string line;
	while (getline(ifs, line)){
		vector<double> v;
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

void print_data(const Table &data){
	for (auto &v: data){
		for (auto &d: v)
			if (d < 0)
				cout << "? ";
			else
				cout << d << " ";
		cout << endl;
	}
}

int main(){
	string line;
	Table data;
	cout << ">> ";
	while(getline(cin,line)){
		istringstream iss{line};
		string command;
		iss >> command;
		if (command == "quit"){
			cout << "quitting application" << endl;
			break;
		} else if (command == "set"){
			string subcommand;
			iss >> subcommand;

		} else if (command == "load"){
			string fname;
			iss >> fname;
			try{
				cout << "loading " << fname << endl;
				read(data,fname);
			}catch(...){
			}

		} else if (command == "print"){
			print_data(data);
		}
		else{
			cout << "command not found" << endl;
		}
		cout << ">> ";

	}
}
