#!/bin/sh
#http://linuxreviews.org/howtos/compiling/

rm -r CMakeFiles CMakeCache.txt cmake_install.cmake;
# export CXXFLAGS="-O3 -mtune=generic -pipe -fomit-frame-pointer -funrollloop";
##=>export CXXFLAGS="-g -O3 -pipe -mtune=generic -std=c++0x";
# export CXXFLAGS="-g3 -Wall -std=c++0x";
export CXXFLAGS="-O3 -std=c++0x -mtune=generic";

#cmake -DCMAKE_BUILD_TYPE=DEBUG .;
cmake -DCMAKE_BUILD_TYPE=RELEASE .;
#cmake .;
make;
