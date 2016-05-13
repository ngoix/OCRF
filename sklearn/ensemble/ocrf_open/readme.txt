Early version of OCRF for integration in Random Forest C++ Library
This is an early version of the library, for testing purposes; some extra log files may not be useful.

Datasets for the program are located in ../data/learning
See script_learning.sh for more info about location and other useful parameters

1) Install CMAKE http://www.cmake.org/
or generate Makefile from include and src folder (see CMakeLists.txt)
2) Run ./cmake_optim.sh, a simple configuration file, adding notably flag for C++11 standards
or make if manually generated Makefile
3) Run ./script_learning.sh to launch OCRF with parameters in this script

Results are located in results/res/database_name for the confusion matrix (TP/FN, TN/FP) from which we deduce MCC, Precision, sensitivity, specificity values and other structural statistics (training time, leaves)


Relevant classes for OCRF :
-OCForest, DForest
-OutlierGenerator


For the sake of simplicity, in this implementation : \alpha=1.2 is replaced in the article by \gamma=1+0.2=(1+value)

