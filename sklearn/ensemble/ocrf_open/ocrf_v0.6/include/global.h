#ifndef GLOBAL_H_
#define GLOBAL_H_

#define LINUX
#ifdef LINUX
#include <assert.h>
#define CLEARCMD system("clear")
#define PAUSECMD cout<<"appuyer sur une touche"<<endl; cin>>Utils::_c
#define DATADIR "./data/"
#else
#define CLEARCMD system("CLS")
#define PAUSECMD system("PAUSE")
#define DATADIR "..\\data\\"
#endif

#include <algorithm>
#include <limits>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <cstdlib>
#include <map>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/stat.h>
#include <sys/types.h>


#define u_int unsigned int
#define v_u_int vector<unsigned int>
#define v_inst vector<Instance *>
#define v_inst_it v_inst::iterator

#define PI 3.14159265358


#define DATA_ROOT "../data/learning"
#define DATA_RESULTS_ROOT "results/res"
#define DATA_LOG "results/res/log.txt"
#define OUTLIER 0
#define TARGET 1


using namespace std;

#endif /*GLOBAL_H_*/
