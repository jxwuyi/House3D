#include<pybind11/pybind11.h>
#include<pybind11/stl.h>
#include<pybind11/numpy.h>
#include<iostream>
#include<cstdlib>
#include<cstdio>
#include<cstring>
#include<cmath>
#include<cassert>
#include<array>
#include<vector>
#include<string>
#include<tuple>
#include<map>
#include<set>
#include<memory>
#include<queue>
#include<algorithm>
namespace py = pybind11;
using namespace std;

#define PII pair<int,int>
#define MKP(a,b) make_pair(a,b)
#define X first
#define Y second
#define BOX_TP tuple<double,double,double,double>
#define REGION_TP tuple<int,int,int,int>
#define COMP_TP vector<PII>
#define COMP_PTR shared_ptr<vector<PII>>
#define INDEX(x,y,n) ((x) * (n) + (y))

const double eps = 1e-9;
const int DIRS[4][2] = {{0,1},{1,0},{0,-1},{-1,0}};

template<class T>
T* _get_mem(int size, int val=0) {
    T* ptr = new T[size];
    memset(ptr, val, sizeof(T) * size);
    return ptr;
}

template<class T>
void _fill_region(T* ptr, int n, int x1, int y1, int x2, int y2, int val) {
    for(int i=x1;i<=x2;++i)
        for(int j=y1;j<=y2;++j)
            ptr[INDEX(i,j,n)] = val;
}

class BaseHouse {
public:
    py::array_t<int> obsMap, moveMap;
private:
    int n;
    double L_lo, L_hi, L_det, grid_det, rad;
    vector<py::array_t<int> > connMapLis, inroomDistLis;
    vector<vector<tuple<int,int> > > connCoorsLis;
    vector<int> maxConnDistLis;
    py::array_t<int>* cur_connMap, cur_inroomDist;
    vector<tuple<int,int> >* cur_connCoors;
    int cur_maxConnDist;
    vector<vector<tuple<int,int> > > regValidCoorsLis;
    map<string, int> targetInd, regionInd;
    // find connected components
    vector<COMP_PTR> _find_components(int x1, int y1, int x2, int y2, bool return_largest=false, bool return_open=false);
public:
    BaseHouse(int resolution): n(resolution), cur_connMap(nullptr), cur_connCoors(nullptr) {}
    void _setScale(double _lo, double _hi, double _rad) {
        L_lo = _lo; L_hi = _hi; rad = _rad;
        L_det = _hi - _lo; grid_det = L_det / n;
    }

    //////////////////////////////////
    // Cache Setter Functions
    //////////////////////////////////
    // set obsMap from python
    void _setObsMap(const py::array_t<int>& val) {
        if (val.ndim() != 2 && val.shape(0) == val.shape(1) && val.shape(0) == n+1) {  // NOTE: n+1!!
            cerr << "[C++] <BaseHouse._setObsMap> Input ndarray must be a 2-d squared matrix!" << endl;
            assert(false);
        }
        int m = val.shape(0);
        obsMap = py::array_t<int>({m, m}, new int[m * m + 1]);
        for(int i=0;i<m;++i) for(int j=0;j<m;++j) *obsMap.mutable_data(i,j) = *val.data(i,j);
    }
    // set moveMap from python
    void _setMoveMap(const py::array_t<int>& val) {
        if (val.ndim() != 2 && val.shape(0) == val.shape(1) && val.shape(0) == n + 1) { // NOTE: n+1!!
            cerr << "[C++] <BaseHouse._setMoveMap> Input ndarray must be a 2-d squared matrix!" << endl;
            assert(false);
        }
        int m = val.shape(0);
        moveMap = py::array_t<int>({m, m}, new int[m * m + 1]);
        for(int i=0;i<m;++i) for(int j=0;j<m;++j) *moveMap.mutable_data(i,j) = *val.data(i,j);
    }

    ///////////////////////////////////
    // Core Generator Functions
    ///////////////////////////////////
    // generate obstacle map
    //  -> if retObject is false, return nullptr and store in obsMap
    py::array_t<int>* _genObstacleMap(double c_x1, double c_y1, double c_x2, double c_y2, int n_row,
                                      const BOX_TP& all_walls, const BOX_TP& door_obj, const BOX_TP& colide_obj,
                                      bool retObject=false);  // TODO
    // generate movable map
    void _genMovableMap(const vector<REGION_TP>& regions);
    // compute shortest distance
    //  --> return whether success (if the region has any open area)
    bool _genShortestDistMap(const vector<tuple<int,int,int,int>>&regions, const string& tag);
    // compute valid positions in a region
    vector<tuple<int,int> >* _getValidCoors(int x1, int y1, int x2, int y2, const string& reg_tag);

    /////////////////////////////////////////
    // Target Setter Functions
    /////////////////////////////////////////
    // clear and set current distance info
    void _clearCurrentDistMap() {
        cur_connMap = nullptr;
        cur_inroomDist = nullptr;
        cur_connCoors = nullptr;
        cur_maxConnDist = 0;
    }
    bool _setCurrentDistMap(const string& tag) {
        auto iter = targetInd.find(tag);
        if (iter == targetInd.end()) return false;
        int k = iter->second;
        cur_connMap = &connMapLis[k];
        cur_inroomDist = &inroomDistLis[k];
        cur_connCoors = &connCoorsLis[k];
        cur_maxConnDist = maxConnDistLis[k];
        return true;
    }

    ///////////////////////////
    // Getter Functions
    ///////////////////////////
    // get connMap
    py::array_t<int>* _getConnMap() {return cur_connMap;}
    // get inroomDist
    py::array_t<int>* _getInroomDist() {return cur_inroomDist;}
    // get connectedCoors
    vector<tuple<int,int> >* _getConnCoors() {return cur_connCoors;}
    // get maxConnDist
    int _getMaxConnDist() {return cur_maxConnDist;}

    //////////////////////////////////////////////////
    // range check and utility functions
    //////////////////////////////////////////////////
    bool _inside(int x, int y) {return x <= n && x >= 0 && y <= n && y >= 0;}
    bool _canMove(int x, int y) {return _inside(x,y) && *moveMap.data(x,y) > 0;}
    bool _isConnect(int x, int y) {return _inside(x, y) && *(*cur_connMap).data(x,y) != -1;}
    int _getDist(int x, int y) {return *(*cur_connMap).data(x,y);}
    double _getScaledDist(int x, int y) {
        int ret = *(*cur_connMap).data(x,y);
        if (ret < 0) return (double)(ret);
        return (double)(ret) / cur_maxConnDist;
    }
    tuple<int,int,int,int> _rescale(double x1, double y1, double x2, double y2, int n_row) {
        int tx1 = (int)floor((x1 - L_lo) / L_det * n_row+eps);
        int ty1 = (int)floor((y1 - L_lo) / L_det * n_row+eps);
        int tx2 = (int)floor((x2 - L_lo) / L_det * n_row+eps);
        int ty2 = (int)floor((y2 - L_lo) / L_det * n_row+eps);
        return make_tuple(tx1,ty1,tx2,ty2);
    }
    tuple<int,int> _to_grid(double x, double y, int n_row) {
        int tx = (int)floor((x - L_lo) / L_det * n_row+eps);
        int ty = (int)floor((y - L_lo) / L_det * n_row+eps);
        return make_tuple(tx, ty);
    }
    tuple<double,double> _to_coor(int x, int y, bool shft) {
        double tx = x * self.grid_det + self.L_lo;
        double ty = y * self.grid_det + self.L_lo;
        if (shft) {
            tx += 0.5 * self.grid_det;
            ty += 0.5 * self.grid_det;
        }
        return make_tuple(tx, ty);
    }
    bool _check_occupy(double cx, double cy) {
        int x1,y1,x2,y2;
        tie(x1,y1,x2,y2) = _rescale(cx-rad,cy-rad,cx+rad,cy+rad,n);
        for(int x=x1;x<=x2;++x)
            for(int y=y1;y<=y2;++y)
                if (!_inside(x,y) || *obsMap.data(x,y) == 1) return false;
        return true;
    }
};

// find connected components
vector<COMP_PTR> BaseHouse::_find_components(int x1, int y1, int x2, int y2, bool return_largest, bool return_open) {
    int n = this->n;
    vector<COMP_PTR > all_comps;
    vector<int> open_comps;
    map<int,int> visit;
    int k = 0;
    for(int x=x1;x<=x2;++x)
        for(int y=y1;y<=y2;++y) {
            int idx = INDEX(x,y,n);
            if (*moveMap.data(x,y) > 0 && visit.count(INDEX(x,y,n)) == 0) {
                vector<PII> comp;
                comp.push_back(MKP(x,y));
                visit[idx] = k;
                size_t ptr = 0;
                bool is_open = false;
                while (ptr < comp.size()) {
                    int px = comp[ptr].X, py = comp[ptr].Y;
                    ptr ++;
                    for (int d = 0; d < 4; ++ d) {
                        int tx = px + DIRS[d][0], ty = py + DIRS[d][1];
                        if (_canMove(tx, ty)) {
                            if (tx < x1 || tx > x2 || ty < y1 || ty > y2) {
                                is_open = true;
                                continue;
                            }
                            int nxt_idx = INDEX(tx,ty,n);
                            if (visit.count(nxt_idx) == 0) {
                                visit[nxt_idx] = k;
                                comp.push_back(MKP(tx, ty));
                            }
                        }
                    }
                }
                if (is_open) open_comps.push_back(k);
                k ++;
                all_comps.push_back(make_shared<COMP_TP>(comp));
            }
        }
    if (k == 0) return all_comps; // no components found
    if (return_open) {
        if (open_comps.size() == 0) {
             cerr << ('WARNING!!!! [House] <find components in Target Room> No Open Components Found!!!! Return Largest Instead!!!!') << endl;
             return_largest = true;
        } else {
            vector<COMP_PTR > tmp;
            for (auto i: open_comps)
                tmp.push_back(all_comps[i]);
            all_comps = tmp;
        }
    }
    if (return_largest) {
        int p = -1;
        size_t p_sz = 0;
        for(size_t i=0;i<all_comps.size();++i)
            if(all_comps[i]->size() > p_sz) {
                p_sz = all_comps[i]->size();
                p = i;
            }
        vector<COMP_PTR > tmp({all_comps[p]});
        all_comps = tmp;
    }
    return all_comps;
}

// generate valid coors in a region
vector<tuple<int,int> >* BaseHouse::_getValidCoors(int x1, int y1, int x2, int y2, const string& reg_tag) {
    auto iter = regionInd.find(reg_tag);
    if (iter != regionInd.end()) {
        int k = iter->second;
        return &regValidCoorsLis[k];
    } else {
        int k = regValidCoorsLis.size();
        regionInd[reg_tag] = k;
        regValidCoorsLis.push_back(vector<tuple<int,int> >({}));
        auto& coors = regValidCoorsLis[k];
        auto dat = _find_components(x1, y1, x2, y2, true, false);
        auto& comp = dat[0]; // the largest components
        for(auto& p: *comp)
            coors.push_back(make_tuple(p.X, p.Y));
        return &regValidCoorsLis[k];
    }
}

// generate obstacle map
py::array_t<int>* BaseHouse::_genObstacleMap(
    double c_x1, double c_y1, double c_x2, double c_y2, int n_row,
    const BOX_TP& all_walls, const BOX_TP& door_obj, const BOX_TP& colide_obj,
    bool retObject=false) {
    //TODO
    return nullptr;
}

// generate movable map
void BaseHouse::_genMovableMap(const vector<REGION_TP>& regions) {
    int x1,y1,x2,y2;
    moveMap = py::array_t<int>({n+1, n+1}, _get_mem((n+1) * (n+1) + 1, 0));
    for(auto& reg: regions) {
        tie(x1,y1,x2,y2) = reg;
        for(int x=x1;x<=x2;++x)
            for(int y=y1;y<=y2;++y) {
                double cx, cy;
                tie(cx,cy) = _to_coor(x,y,true);
                if _check_occupy(cx, cy):
                    *moveMap.mutable_data(x,y) = 1;
            }
    }
}

// generate shortest distance map (connMap)
bool BaseHouse::_genShortestDistMap(const vector<tuple<int,int,int,int>>&regions, const string& tag) {
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
        .def("_getValidCoors", &BaseHouse::_getValidCoors, py::return_value_policy::reference)
        .def("_getMaxConnDist", &BaseHouse::_getMaxConnDist)
        .def("_setObsMap", &BaseHouse::_setObsMap)
        .def("_setMoveMap", &BaseHouse::_setMoveMap)
        .def_readonly("obsMap", &BaseHouse::obsMap)
        .def_readonly("moveMap", &BaseHouse::moveMap);
}

