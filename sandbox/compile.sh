c++ -O3 -Wall -shared -std=c++11 -fPIC `python3 -m pybind11 --includes` \
    house.cpp -o house.so
