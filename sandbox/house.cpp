#include<pybind11/pybind11.h>
#include<pybind11/stl.h>
#include<pybind11/numpy.h>
#include<iostream>
#include<cstdlib>
#include<cstdio>
#include<cstring>
#include<cassert>
#include<array>
#include<vector>
#include<string>
#include<tuple>
#include<map>
namespace py = pybind11;
using namespace std;

#define PII pair<int,int>
#define mkp(a,b) make_pair(a,b)
#define X first
#define Y second

const int dirs[4][2] = {{0,1},{1,0},{0,-1},{-1,0}};

class BaseHouse {
public:
    py::array_t<int> obsMap, moveMap;
private:
    int n;
    vector<py::array_t<int> > connMapLis;
    vector<vector<tuple<int,int> > > connCoorsLis;
    vector<int> maxConnDistLis;
    py::array_t<int>* cur_connMap;
    vector<tuple<int,int> >* cur_connCoors;
    int cur_maxConnDist;
    vector<vector<tuple<int,int> > > regValidCoorsLis;
    map<string, int> targetInd, regionInd;
    // find connected components
    vector<PII > _find_components(int x1, int y1, int x2, int y2, bool return_largest=false, bool return_open=false);
public:
    BaseHouse(int resolution): n(resolution), cur_connMap(nullptr), cur_connCoors(nullptr) {}
    // set obsMap from python
    void _setObsMap(const py::array_t<int>& val) {
        if (val.ndim() != 2 && val.shape(0) == val.shape(1) && val.shape(0) == n) {
            cerr << "[C++] <BaseHouse._setObsMap> Input ndarray must be a 2-d squared matrix!" << endl;
            assert(false);
        }
        int m = val.shape(0);
        obsMap = py::array_t<int>({m, m}, new int[m * m + 1]);
        for(int i=0;i<m;++i) for(int j=0;j<m;++j) *obsMap.mutable_data(i,j) = *val.data(i,j);
    }
    // set moveMap from python
    void _setMoveMap(const py::array_t<int>& val) {
        if (val.ndim() != 2 && val.shape(0) == val.shape(1) && val.shape(0) == n) {
            cerr << "[C++] <BaseHouse._setMoveMap> Input ndarray must be a 2-d squared matrix!" << endl;
            assert(false);
        }
        int m = val.shape(0);
        moveMap = py::array_t<int>({m, m}, new int[m * m + 1]);
        for(int i=0;i<m;++i) for(int j=0;j<m;++j) *moveMap.mutable_data(i,j) = *val.data(i,j);
    }
    // generate obstacle map
    //  -> if retObject is false, return nullptr and store in obsMap
    py::array_t<int>* _genObstacleMap(int n_row, bool retObject=false);
    // compute shortest distance
    void _genShortestDistMap(const vector<tuple<int,int,int,int>>&regions, const string& tag);
    // clear and set current distance info
    void _clearCurrentDistMap() {
        cur_connMap = nullptr;
        cur_connCoors = nullptr;
        cur_maxConnDist = 0;
    }
    void _setCurrentDistMap(const string& tag) {
        auto iter = targetInd.find(tag);
        if (iter == targetInd.end()) return ;
        int k = iter->second;
        cur_connMap = &connMapLis[k];
        cur_connCoors = &connCoorsLis[k];
        cur_maxConnDist = maxConnDistLis[k];
    }
    // compute valid positions in a region
    vector<tuple<int,int> >* _getValidCoors(int x1, int y1, int x2, int y2, const string& reg_tag);
    // get connMap
    py::array_t<int>* _getConnMap() {return cur_connMap;}
    // get connectedCoors
    vector<tuple<int,int> >* _getConnCoors() {return cur_connCoors;}
    // get maxConnDist
    int _getMaxConnDist() {return cur_maxConnDist;}
};

// find connected components
vector<PII > BaseHouse::_find_components(int x1, int y1, int x2, int y2, bool return_largest, bool return_open) {
    return vector<PII>({});
}

// generate obstacle map
py::array_t<int>* BaseHouse::_genObstacleMap(int n_row, bool retObject) {
    return nullptr;
}

// generate shortest distance map (connMap)
void BaseHouse::_genShortestDistMap(const vector<tuple<int,int,int,int>>&regions, const string& tag) {
}

// generate valid coors in a region
vector<tuple<int,int> >* _getValidCoors(int x1, int y1, int x2, int y2, const string& reg_tag) {
}

PYBIND11_MODULE(example, m) {
    m.doc() = "[House3D] <BaseHouse> C++ implementation of calculating basic House properties"; // optional module docstring
    m.def("_getConnMap", &BaseHouse::_getConnMap, "get the current connMap");
    m.def("_getConnCoors", &BaseHouse::_getConnCoors, "get the current connectedCoors");
    m.def("_getMaxConnDist", &BaseHouse::_getMaxConnDist, "get the current maxConnDist");

    py::class_<BaseHouse>(m, "BaseHouse")
        .def(py::init<int>())
        .def("_getConnMap", &BaseHouse::_getConnMap, py::return_value_policy::reference)
        .def("_getConnCoors", &BaseHouse::_getConnCoors, py::return_value_policy::reference)
        .def("_getMaxConnDist", &BaseHouse::_getMaxConnDist)
        .def("_setObsMap", &BaseHouse::_setObsMap)
        .def("_setMoveMap", &BaseHouse::_setMoveMap)
        .def_readonly("obsMap", &BaseHouse::obsMap)
        .def_readonly("moveMap", &BaseHouse::moveMap);
}

