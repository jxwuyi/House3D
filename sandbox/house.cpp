#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;

int add(int i, int j) {
    return i + j;
}

#include<array>
#include<vector>
int32_t accum(const std::vector<int32_t>&a) {
    int ret=0;
    for (auto&k: a)
        ret += k;
    return ret;
}

#include<tuple>
std::vector<int32_t> prod(const std::vector<std::tuple<int32_t,int32_t>>&a) {
    std::vector<int32_t> ret;
    for(auto&v: a)
        ret.push_back(std::get<0>(v) * std::get<1>(v));
    return ret;
}

#include<pybind11/numpy.h>
#include<iostream>
#include<cstdlib>
#include<cstdio>
#include<cstring>

template<class T>
inline T* _get_mem(int32_t sz) {
    auto ptr = new T[sz];
    std::memset(ptr, 0, sizeof(T) * sz);
    return ptr;
}

class MyClass {
public:
    py::array_t<int32_t> a;
    MyClass(int32_t n, int32_t m): a({n, m}, _get_mem<int32_t>(n * m + 2)){
        std::cout<<"owndata flag = "<<this->a.owndata()<<std::endl;
    }
    int32_t size() {return a.size();}
    void set(int32_t i, int32_t j, int32_t v) {
        *a.mutable_data(i,j) = v;
    }
};

PYBIND11_MODULE(example, m) {
    m.doc() = "pybind11 example plugin"; // optional module docstring
    m.def("add", &add, "A function which adds two numbers");
    m.def("accum", &accum, "A function which takes the summation of the input array");
    m.def("prod", &prod, "A function generates a prod list");

    py::class_<MyClass>(m, "MyClass")
        .def(py::init<int32_t, int32_t>())
        .def("size", &MyClass::size)
        .def("set", &MyClass::set)
        .def_readonly("a", &MyClass::a);
}

