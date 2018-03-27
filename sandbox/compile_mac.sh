#!/usr/bin/env bash
c++ -O3 -Wall -shared -std=c++11 -Wl,-undefined,dynamic_lookup -fPIC `python3 -m pybind11 --includes` \
    house.cpp -o base_house.so
