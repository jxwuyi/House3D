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
#include<random>
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
    vector<py::array_t<int> > connMapLis;
    vector<vector<tuple<int,int> > > connCoorsLis;
    vector<int> maxConnDistLis;
    py::array_t<int>* cur_connMap;
    vector<tuple<int,int> >* cur_connCoors;
    int cur_maxConnDist;
    vector<vector<tuple<int,int> > > regValidCoorsLis;
    vector<tuple<int,int> >* last_regValidCoors;
    map<string, int> targetInd;  // index for targets <connMapLis, connCoorsLis, maxConnDistLis>
    vector<string> targetNames;  // list of target names
    map<string, int> regionInd; // index for <regValidCoorsLis>
    vector<string> regionNames; // list of region names

    ////////////////////////////////////////
    // for connectivity
    int n_obj, n_target, n_room;
    py::array_t<int> targetMask;   // the mask of whether this cell belongs to a target
    vector<vector<int> > targetDist;  // pairwise distance over targets
public:
    int _gen_target_graph(int _n_obj);  // return the total number of targets
    vector<string> _compute_target_plan(double cx, double cy, const string& target);
    vector<double> _get_target_plan_dist(double cx, double cy, const vector<string>& plan);
    int _get_target_mask(double cx, double cy, bool only_object=false);
    vector<string> _get_target_mask_names(double cx, double cy, bool only_object=false);

private:
    ////////////////////////////////////////
    // find connected components
    //   -> when return_open == true, return only open-components
    //      ---> when no open components, will automatically set return_largest <- true
    //   -> when return_largest == true, return only the largest components
    vector<COMP_PTR> _find_components(int x1, int y1, int x2, int y2, bool return_largest=false, bool return_open=false);
    // check whether grid(gx,gy) will be covered when a robot stands at coor(cx,cy)
    bool _check_grid_occupy(double cx, double cy, int gx, int gy) {
        for(int x=gx;x<=gx+1;++x)
            for(int y=gy;y<=gy+1;++y) {
                double rx = x * grid_det + L_lo, ry = y * grid_det + L_lo;
                rx -= cx; ry -= cy;
                if (rx * rx + ry * ry <= rad * rad) return true;
            }
        return false;
    }
public:
    BaseHouse(int resolution): n(resolution), cur_connMap(nullptr), cur_connCoors(nullptr) {}
    void _setHouseBox(double _lo, double _hi, double _rad) {
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
                                      bool retObject);  // TODO
    // generate movable map
    //   -> input grid_indices
    void _genMovableMap(const vector<REGION_TP>& regions);
    // compute shortest distance
    //  --> return whether success (if the region has any open area)
    //  --> input real-coordinates
    bool _genShortestDistMap(const vector<BOX_TP>&boxes, const string& tag);
    // compute shortest distance map outside the regions in boxes
    bool _genOutsideDistMap(const vector<BOX_TP>&boxes, const string& tag);
    // compute and cache valid positions in a region
    bool _genValidCoors(int x1, int y1, int x2, int y2, const string& reg_tag);

    /////////////////////////////////////////
    // Target Dist Map Setter Functions
    /////////////////////////////////////////
    // clear and set current distance info
    void _clearCurrentDistMap() {
        cur_connMap = nullptr;
        cur_connCoors = nullptr;
        cur_maxConnDist = 0;
    }
    bool _setCurrentDistMap(const string& tag) {
        auto iter = targetInd.find(tag);
        if (iter == targetInd.end()) return false;
        int k = iter->second;
        cur_connMap = &connMapLis[k];
        cur_connCoors = &connCoorsLis[k];
        cur_maxConnDist = maxConnDistLis[k];
        return true;
    }

    ///////////////////////////
    // Getter Functions
    ///////////////////////////
    // get connMap
    py::array_t<int>* _getConnMap() {return cur_connMap;}
    // get connectedCoors
    vector<tuple<int,int> >* _getConnCoors() {return cur_connCoors;}
    // get maxConnDist
    int _getMaxConnDist() {return cur_maxConnDist;}
    // get validRoomCoors
    vector<tuple<int,int> >* _getValidCoors(const string& reg_tag) {
        auto iter = regionInd.find(reg_tag);
        if (iter == regionInd.end()) return nullptr;
        return &regValidCoorsLis[iter->second];
    }

    //////////////////////////////////////
    // location getter utility functions
    //////////////////////////////////////
    // get valid coors and cache it
    int _fetchValidCoorsSize(const string& reg_tag) {
        auto iter = regionInd.find(reg_tag);
        if (iter == regionInd.end()) {
            last_regValidCoors = nullptr;
            return 0;
        }
        last_regValidCoors = &regValidCoorsLis[iter->second];
        return last_regValidCoors->size();
    }
    // return indexed coor from cached list
    tuple<int,int> _getCachedIndexedValidCoor(int k) {
        if (last_regValidCoors == nullptr) return make_tuple(-1, -1);
        return (*last_regValidCoors)[k];
    }
    int _getConnectCoorsSize(const string& tag){
        auto iter = targetInd.find(tag);
        if (iter == targetInd.end()) return 0;
        return connCoorsLis[iter->second].size();
    }
    int _getConnectCoorsSize_Bounded(const string& tag, int bound){
        auto iter = targetInd.find(tag);
        if (iter == targetInd.end()) return 0;
        auto& coors = connCoorsLis[iter->second];
        auto& connMap = connMapLis[iter->second];
        int lo = -1, hi = coors.size(), mid;
        while(lo + 1 < hi) {
            mid = (lo + hi) / 2;
            int x, y;
            tie(x, y) = coors[mid];
            if (*connMap.data(x, y) <= bound) lo = mid; else hi = mid;
        }
        return hi;
    }
    tuple<int,int> _getIndexedConnectCoor(const string& tag, int k){
        auto iter = targetInd.find(tag);
        if (iter == targetInd.end()) return make_tuple(-1, -1);
        return connCoorsLis[iter->second][k];
    }
    int _getCurrConnectCoorsSize(){
        if (cur_connCoors == nullptr) return 0;
        return cur_connCoors->size();
    }
    int _getCurrConnectCoorsSize_Bounded(int bound){
        if (cur_connCoors == nullptr) return 0;
        auto& coors = *cur_connCoors;
        auto& connMap = *cur_connMap;
        int lo = -1, hi = coors.size(), mid;
        while(lo + 1 < hi) {
            mid = (lo + hi) / 2;
            int x, y;
            tie(x, y) = coors[mid];
            if (*connMap.data(x,y) <= bound) lo = mid; else hi = mid;
        }
        return hi;
    }
    tuple<int,int> _getCurrIndexedConnectCoor(int k){ return (*cur_connCoors)[k]; }

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
        double tx = x * grid_det + L_lo;
        double ty = y * grid_det + L_lo;
        if (shft) {
            tx += 0.5 * grid_det;
            ty += 0.5 * grid_det;
        }
        return make_tuple(tx, ty);
    }

    ///////////////////////////////////
    // Collision Check (assume (Ax, Ay) is a good point, check whether can move to (Bx, By))
    ///////////////////////////////////
    bool _check_occupy(double cx, double cy) {
        int x1,y1,x2,y2;
        tie(x1,y1,x2,y2) = _rescale(cx-rad,cy-rad,cx+rad,cy+rad,n);
        for(int x=x1;x<=x2;++x)
            for(int y=y1;y<=y2;++y)
                if ((!_inside(x,y) || *obsMap.data(x,y) == 1) && _check_grid_occupy(cx, cy, x, y)) return false;
        return true;
    }
    bool _full_collision_check(double Ax, double Ay, double Bx, double By, int num_samples) {
        double ratio = 1.0 / num_samples;
        Bx -= Ax; By -= Ay;
        for(int i=1;i<=num_samples;++i) {
            double px = i * ratio * Bx + Ax, py = i * ratio * By + Ay;
            if (!_check_occupy(px, py)) return false;
        }
        return true;
    }
    bool _fast_collision_check(double Ax, double Ay, double Bx, double By, int num_samples) {
        double ratio = 1.0 / num_samples;
        Bx -= Ax; By -= Ay;
        for(int i=1;i<=num_samples;++i) {
            double px = i * ratio * Bx + Ax, py = i * ratio * By + Ay;
            int gx, gy;
            tie(gx, gy) = _to_grid(px,py,n);
            if (!_canMove(gx, gy) || !_isConnect(gx, gy)) return false;
        }
        return true;
    }
};

///////////////////////////////
// build connectivity graph
///////////////////////////////
// init graph
int BaseHouse::_gen_target_graph(int _n_obj) {
  // assume in <targetNames>
  //     targetNames[0:n_target - n_obj] are rooms
  //     targetNames[-n_obj:] are object targets
  this->n_obj = _n_obj;
  this->n_target = connMapLis.size();
  this->n_room = n_target - n_obj;

  if (n_target > 30) {
    cerr << "[ERROR] We currently only support building object graph over *AT MOST* 30 objects!!!" << endl;
    return -1;
  }

  // build masks
  int m = obsMap.shape(0);
  targetMask = py::array_t<int>({m, m}, _get_mem<int>(m*m+2));
  for(int pt = 0; pt < n_target; ++ pt) {
    auto& coors = connCoorsLis[pt];
    auto& connMap = connMapLis[pt];
    for(auto& c: coors) {
      int x, y;
      tie(x,y) = c;
      if (*connMap.data(x,y) > 0) break;
      *targetMask.mutable_data(x,y) |= (1 << pt);
    }
  }

  // build connectivity graph over all targets
  targetDist.clear();
  for(int pt = 0; pt < n_target; ++ pt) {
    vector<int> cur_dist(n_target, -1);   // from a room/object to any other room/objects
    auto& coors = connCoorsLis[pt];
    auto& connMap = connMapLis[pt];
    int remain_mask = (1 << n_target) - 1;
    cur_dist[pt] = 0;
    remain_mask ^= 1 << pt;
    for (auto& c: coors) {
      int x, y;
      tie(x, y) = c;
      int mask = *targetMask.data(x,y) & remain_mask;  // list of targets associated to the gird(x,y)
      remain_mask ^= mask;   // the targets that needs to update its distance
      for(; mask > 0; mask &= mask - 1) {
        int i = __builtin_ctz(mask);
        cur_dist[i] = *connMap.data(x,y);
      }
      if (remain_mask == 0) break;  // no target needs to update distance
    }
    targetDist.push_back(cur_dist);
  }

  // return the total number of targets
  return this->n_target;
}

// compute the optimal object level plan to the target
vector<string> BaseHouse::_compute_target_plan(double cx, double cy, const string& target) {
  auto iter = targetInd.find(target);
  if (iter == targetInd.end()) {
    cerr << "[ERROR] Invalid Target <"<<target<<">! Distance Map Not Cached Yet!" << endl;
    return vector<string>();
  }

  int tar_ind = iter->second;
  vector<int> cur_dist(n_target, -1);
  vector<int> cur_steps(n_target, -1);
  vector<int> prev_node(n_target, -1);
  int gx, gy;
  tie(gx, gy) = _to_grid(cx, cy, n);
  // compute the distance to objects
  for(int i=0;i<n_target;++i) {
    auto& connMap = connMapLis[i];
    if (*connMap.data(gx, gy) >= 0) {
      cur_dist[i] = *connMap.data(gx, gy);
      cur_steps[i] = 1;
      prev_node[i] = -10;  // indicating the starting grid
    }
  }
  // Dijkstra SSP algorithm over rooms
  int ssp_mask = (1 << n_room) - 1;
  for (int i=0;i<n_room;++i) {
    int p = -1;
    for (int tmask = ssp_mask; tmask > 0; tmask &= tmask - 1) {
      int j = __builtin_ctz(tmask);
      if (cur_dist[j] < 0) continue;
      if (p < 0 || cur_dist[j] < cur_dist[p]
          || (cur_dist[j] == cur_dist[p] && cur_steps[j] < cur_steps[p])) p = j;
    }
    if (p < 0 || p == tar_ind) break;
    ssp_mask ^= 1 << p;
    for (int tmask = ssp_mask; tmask > 0; tmask &= tmask - 1) {
      int j = __builtin_ctz(tmask);
      if (targetDist[p][j] < 0) continue;
      int t_dist = targetDist[p][j] + cur_dist[p];
      int t_steps = cur_steps[p] + 1;
      if (cur_dist[j] < 0 || cur_dist[j] > t_dist
          || (cur_dist[j] == t_dist && cur_steps[j] > t_steps)) {
        cur_dist[j] = t_dist;
        cur_steps[j] = t_steps;
        prev_node[j] = p;
      }
    }
  }
  // Compute Best Distance to Objects
  for (int i=0;i<n_room;++i) {
    if (cur_dist[i] < 0) continue;
    int t_steps = cur_steps[i] + 1;
    for(int j=0;j<n_target;++j) {
      int t_dist = targetDist[i][j] + cur_dist[i];
      if (cur_dist[j] < 0 || cur_dist[j] > t_dist
         || (cur_dist[j] == t_dist && cur_steps[j] > t_steps)) {
        cur_dist[j] = t_dist;
        cur_steps[j] = t_steps;
        prev_node[j] = i;
      }
    }
  }
  // check final distance
  if (cur_dist[tar_ind] < 0) {
    cerr << "[ERROR] <BaseHouse::_compute_target_plan> Target <"<<target<<"> Not Connected from Coor<"<<cx<<","<<cy<<">!!" << endl;
    return vector<string>();
  }
  // retrieve the optimal plan
  vector<string> path;
  for (int p = tar_ind; p >= 0; p = prev_node[p]) path.push_back(targetNames[p]);
  reverse(path.begin(), path.end());
  return path;
}

// compute the distances for executing the plan from start points (cx, cy)
vector<double> BaseHouse::_get_target_plan_dist(double cx, double cy, const vector<string>& plan) {
  if (plan.size() == 0) return vector<double>();
  int gx, gy;
  tie(gx, gy) = _to_grid(cx, cy, n);
  vector<double> dist;
  int prev = -1;
  for(int i=0;i<plan.size();++i) {
    auto iter = targetInd.find(plan[i]);
    if (iter == targetInd.end()) {
      cerr << "[ERROR] <BaseHouse::_get_target_plan_dist> Invalid Plan!! Target Not Found <"<<plan[i]<<">"<<endl;
      return vector<double>();
    }
    int t = iter->second;
    int d = -1;
    if (prev < 0)
      d = *connMapLis[t].data(gx,gy);
    else
      d = targetDist[prev][t];
    if (d < 0) {
      cerr << "[ERROR] <BaseHouse::_get_target_plan_dist> Target <"<<plan[i]<<"> Not Connected!!!"<<endl;
      return vector<double>();
    }
    dist.push_back(d * grid_det);
    prev = t;
  }
  return dist;
}

// get the target mask for coor(cx, cy)
//    when only_object == True, only return object_target bits
int BaseHouse::_get_target_mask(double cx, double cy, bool only_object) {
  int gx, gy;
  tie(gx, gy) = _to_grid(cx, cy, n);
  int mask = *targetMask.data(gx, gy);
  if (only_object) mask &= (1 << n_room) - 1;
  return mask;
}

// return the target names associated with coor (cx, cy)
vector<string> BaseHouse::_get_target_mask_names(double cx, double cy, bool only_object) {
  int mask = this->_get_target_mask(cx, cy, only_object);
  vector<string> targets;
  for(; mask > 0; mask &= mask-1)
    targets.push_back(targetNames[__builtin_ctz(mask)]);
  return targets;
}


///////////////////////////////
// find connected components
vector<COMP_PTR> BaseHouse::_find_components(int x1, int y1, int x2, int y2, bool return_largest, bool return_open) {
    vector<COMP_PTR > all_comps;
    vector<int> open_comps;
    map<int,int> visit;
    int k = 0;
    int sz = this->n + 1;
    for(int x=x1;x<=x2;++x)
        for(int y=y1;y<=y2;++y) {
            int idx = INDEX(x,y,sz);
            if (*moveMap.data(x,y) > 0 && visit.count(INDEX(x,y,sz)) == 0) {
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
                            int nxt_idx = INDEX(tx,ty,sz);
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
             cerr << "WARNING!!!! [House] <find components> No Open Components Found!!!! Return Largest Instead!!!!" << endl;
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
bool BaseHouse::_genValidCoors(int x1, int y1, int x2, int y2, const string& reg_tag) {
    if (regionInd.count(reg_tag) > 0) return false;
    int k = regValidCoorsLis.size();
    regionNames.push_back(reg_tag);
    regionInd[reg_tag] = k;
    regValidCoorsLis.push_back(vector<tuple<int,int> >({}));
    auto& coors = regValidCoorsLis[k];
    auto dat = _find_components(x1, y1, x2, y2, true, false);
    auto& comp = dat[0]; // the largest components
    for(auto& p: *comp)
        coors.push_back(make_tuple(p.X, p.Y));
    return true;
}

// generate obstacle map
py::array_t<int>* BaseHouse::_genObstacleMap(
    double c_x1, double c_y1, double c_x2, double c_y2, int n_row,
    const BOX_TP& all_walls, const BOX_TP& door_obj, const BOX_TP& colide_obj,
    bool retObject) {
    //TODO
    return nullptr;
}

// generate movable map
void BaseHouse::_genMovableMap(const vector<REGION_TP>& regions) {
    int x1,y1,x2,y2;
    moveMap = py::array_t<int>({n+1, n+1}, _get_mem<int>((n+1) * (n+1) + 1, 0));
    for(auto& reg: regions) {
        tie(x1,y1,x2,y2) = reg;
        for(int x=x1;x<=x2;++x)
            for(int y=y1;y<=y2;++y) {
                double cx, cy;
                tie(cx,cy) = _to_coor(x,y,true);
                if (_check_occupy(cx, cy))
                    *moveMap.mutable_data(x,y) = 1;
            }
    }
}

// generate shortest distance map (connMap)
bool BaseHouse::_genShortestDistMap(const vector<BOX_TP>&boxes, const string& tag) {
    if (targetInd.count(tag) > 0) {
        cerr << "Warning!!! [House] <_genShortestDistMap> Invalid tag<" << tag << ">! tag already exists!" << endl;
        return false;
    }
    int sz = n + 1;
    int* connMap = _get_mem<int>(sz*sz+1, -1);
    vector<PII> que;
    for(auto& box: boxes) {
        double _x1, _y1, _x2, _y2;
        tie(_x1,_y1,_x2,_y2) = box;
        int x1,y1,x2,y2;
        tie(x1,y1,x2,y2) = _rescale(_x1,_y1,_x2,_y2,n);
        vector<COMP_PTR> cur_comps = _find_components(x1, y1, x2, y2, false, true); // find open components;
        if (cur_comps.size() == 0) {
            fprintf(stderr, "WARNING!!!! [House] No Space Found in tag[%s] <bbox=[%.2f, %2f] x [%.2f, %.2f]",
                            tag.c_str(), _x1, _y1, _x2, _y2);
            continue;
        }
        for(auto& comp_pt: cur_comps) {
            for(auto& p: *comp_pt) {
                int x=p.X, y=p.Y;
                connMap[INDEX(x,y,sz)]=0;
                que.push_back(p);
            }
        }
    }
    if (que.size() == 0) { // fully blocked region
        cerr << "Error!! [House] No space found for tag <" << tag << ">." << endl;
        return false;
    }
    // BFS and compute shortest distance
    size_t ptr = 0;
    int maxConnDist = 1;
    while (ptr < que.size()) {
        int x=que[ptr].X, y=que[ptr].Y;
        int cur_dist = connMap[INDEX(x,y,sz)];
        ptr ++;
        for (int d=0;d<4;++d) {
            int tx=x+DIRS[d][0], ty=y+DIRS[d][1];
            if (_canMove(tx,ty) && connMap[INDEX(tx,ty,sz)] < 0) {
                que.push_back(MKP(tx,ty));
                connMap[INDEX(tx,ty,sz)] = cur_dist + 1;
                if (cur_dist >= maxConnDist) maxConnDist = cur_dist + 1;
            }
        }
    }
    // create and cache results
    int k = maxConnDistLis.size();
    targetNames.push_back(tag);
    targetInd[tag] = k;
    maxConnDistLis.push_back(maxConnDist);
    connMapLis.push_back(py::array_t<int>({sz, sz}, new int[sz * sz + 1]));
    connCoorsLis.push_back(vector<tuple<int,int>>());
    py::array_t<int> &py_connMap = connMapLis[k];
    vector<tuple<int,int>> &coors = connCoorsLis[k];
    for(int i=0;i<sz;++i)
        for(int j=0;j<sz;++j)
            *py_connMap.mutable_data(i,j) = connMap[INDEX(i,j,sz)];
    delete connMap;
    for(auto&p: que)
        coors.push_back(make_tuple(p.X, p.Y));
    return true;
}

// generate a special region <outside boxes> and build the shortest distance map towards (connMap)
bool BaseHouse::_genOutsideDistMap(const vector<BOX_TP>&boxes, const string& tag) {
    if (targetInd.count(tag) > 0) {
        cerr << "Warning!!! [House] <_genOutsidetDistMap> Invalid tag<" << tag << ">! tag already exists!" << endl;
        return false;
    }
    int sz = n + 1;
    int* connMap = _get_mem<int>(sz*sz+1, -1);
    vector<PII> que;
    for(auto& box: boxes) {
        double _x1, _y1, _x2, _y2;
        tie(_x1,_y1,_x2,_y2) = box;
        int x1,y1,x2,y2;
        tie(x1,y1,x2,y2) = _rescale(_x1,_y1,_x2,_y2,n);
        for (int x=x1;x<=x2;++x)
            for (int y=y1;y<=y2;++y)
                connMap[INDEX(x,y,sz)] = -2;
    }
    for (int x=0;x<sz;++x)
        for (int y=0;y<sz;++y)
            if (connMap[INDEX(x,y,sz)] == -2) {
                for (int d=0;d<4;++d) {
                    int tx=x+DIRS[d][0], ty=y+DIRS[d][1];
                    if (_canMove(tx,ty) && connMap[INDEX(tx,ty,sz)] == -1) {
                        que.push_back(MKP(tx,ty));
                        connMap[INDEX(tx,ty,sz)] = 0;
                    }
                }
            }
    if (que.size() == 0) { // fully blocked region
        cerr << "Error!! [House] No outside space found for tag <" << tag << ">." << endl;
        return false;
    }
    // expand the outside region
    vector<PII> inside;
    size_t ptr = 0;
    while(ptr < que.size()) {
        int x=que[ptr].X, y=que[ptr].Y;
        ptr ++;
        for (int d=0;d<4;++d) {
            int tx=x+DIRS[d][0], ty=y+DIRS[d][1];
            if (_canMove(tx,ty)) {
                if (connMap[INDEX(tx,ty,sz)] == -1) {
                    que.push_back(MKP(tx,ty));
                    connMap[INDEX(tx,ty,sz)] = 0;
                } else
                if (connMap[INDEX(tx,ty,sz)] == -2) {
                    inside.push_back(MKP(tx,ty));
                    connMap[INDEX(tx,ty,sz)] = 1;
                }
            }
        }
    }
    // BFS and compute shortest distance
    ptr = 0;
    int maxConnDist = 1;
    while (ptr < inside.size()) {
        int x=inside[ptr].X, y=inside[ptr].Y;
        int cur_dist = connMap[INDEX(x,y,sz)];
        ptr ++;
        for (int d=0;d<4;++d) {
            int tx=x+DIRS[d][0], ty=y+DIRS[d][1];
            if (_canMove(tx,ty) && connMap[INDEX(tx,ty,sz)] == -2) {
                inside.push_back(MKP(tx,ty));
                connMap[INDEX(tx,ty,sz)] = cur_dist + 1;
                if (cur_dist >= maxConnDist) maxConnDist = cur_dist + 1;
            }
        }
    }
    // create and cache results
    int k = maxConnDistLis.size();
    targetNames.push_back(tag);
    targetInd[tag] = k;
    maxConnDistLis.push_back(maxConnDist);
    connMapLis.push_back(py::array_t<int>({sz, sz}, new int[sz * sz + 1]));
    connCoorsLis.push_back(vector<tuple<int,int>>());
    py::array_t<int> &py_connMap = connMapLis[k];
    vector<tuple<int,int>> &coors = connCoorsLis[k];
    for(int i=0;i<sz;++i)
        for(int j=0;j<sz;++j) {
            int v = connMap[INDEX(i,j,sz)];
            *py_connMap.mutable_data(i,j) = (v < 0 ? -1 : v);
        }
    delete connMap;
    for(auto&p: que)
        coors.push_back(make_tuple(p.X, p.Y));
    for(auto&p: inside)
        coors.push_back(make_tuple(p.X, p.Y));
    return true;
}

PYBIND11_MODULE(base_house, m) {
    m.doc() = "[House3D] <BaseHouse> C++ implementation of calculating basic House properties"; // optional module docstring
    m.def("_setHouseBox", &BaseHouse::_setHouseBox, "set the coordinate range and robot radius");
    m.def("_setObsMap", &BaseHouse::_setObsMap, "set the value of obsMap");
    m.def("_setMoveMap", &BaseHouse::_setMoveMap, "set the value of moveMap");
    //m.def("_genObstacleMap", &BaseHouse::_genObstacleMap, "generate the obstacleMap given obstacle information") // TODO
    m.def("_genMovableMap", &BaseHouse::_genMovableMap, "generate movability map and store in moveMap");
    m.def("_genShortestDistMap", &BaseHouse::_genShortestDistMap, "generate and cache shortest distance map from given box ranges for a name tag");
    m.def("_genOutsideDistMap", &BaseHouse::_genOutsideDistMap, "generate and cache shortest distance map from regions outside the given box ranges for a name tag");
    m.def("_genValidCoors", &BaseHouse::_genValidCoors, "generate and cache valid locations in a given region for a name tag");
    m.def("_clearCurrentDistMap", &BaseHouse::_clearCurrentDistMap, "clear all the current pointers related to distance map");
    m.def("_setCurrentDistMap", &BaseHouse::_setCurrentDistMap, "load the cached information of distance map corresponding to a given name tag");
    m.def("_getConnMap", &BaseHouse::_getConnMap, "get the current connMap");
    m.def("_getConnCoors", &BaseHouse::_getConnCoors, "get the current connectedCoors");
    m.def("_getMaxConnDist", &BaseHouse::_getMaxConnDist, "get the current maxConnDist");
    m.def("_getValidCoors", &BaseHouse::_getValidCoors, "get the cached valid locations corresponding to a region_tag");
    m.def("_fetchValidCoorsSize", &BaseHouse::_fetchValidCoorsSize, "cache the valid locations and return the size of it");
    m.def("_getCachedIndexedValidCoor", &BaseHouse::_getCachedIndexedValidCoor, "return the coor from the cached list by the index");
    m.def("_getConnectCoorsSize", &BaseHouse::_getConnectCoorsSize, "return the size of the connectedCoors w.r.t. the tag");
    m.def("_getConnectCoorsSize_Bounded", &BaseHouse::_getConnectCoorsSize_Bounded, "return the size of the connectedCoors of the tag with distance no larger than bound");
    m.def("_getIndexedConnectCoor", &BaseHouse::_getIndexedConnectCoor, "return the coor w.r.t. to the tag by the index");
    m.def("_getCurrConnectCoorsSize", &BaseHouse::_getCurrConnectCoorsSize, "return the size of the current connectedCoors");
    m.def("_getCurrConnectCoorsSize_Bounded", &BaseHouse::_getCurrConnectCoorsSize_Bounded, "return the size of the current connectedCoors with distance no larger than bound");
    m.def("_getCurrIndexedConnectCoor", &BaseHouse::_getCurrIndexedConnectCoor, "return coor of the current connectedCoors by the index");
    m.def("_inside", &BaseHouse::_inside, "util: whether inside the range");
    m.def("_canMove", &BaseHouse::_canMove, "util: whether the robot can stand at the location");
    m.def("_isConnect", &BaseHouse::_isConnect, "util: whether the grid is connected to the target");
    m.def("_getDist", &BaseHouse::_getDist, "util: get the distance in the current distance map");
    m.def("_getScaledDist", &BaseHouse::_getScaledDist, "util: get the scaled distance");
    m.def("_rescale", &BaseHouse::_rescale, "util: convert box range to a grid region");
    m.def("_to_grid", &BaseHouse::_to_grid, "util: convert coordinate to grid location");
    m.def("_to_coor", &BaseHouse::_to_coor, "util: convert grid location to coordinate");
    m.def("_check_occupy", &BaseHouse::_check_occupy, "re-check whether a *coordinate* is valid for the robot");
    m.def("_full_collision_check", &BaseHouse::_full_collision_check, "slow collision check");
    m.def("_fast_collision_check", &BaseHouse::_fast_collision_check, "fast collision check via moveMap");
    // Connectivity Graph related
    m.def("_gen_target_graph", &BaseHouse::_gen_target_graph, "graph: generate target connectivity graph");
    m.def("_compute_target_plan", &BaseHouse::_compute_target_plan, "graph: compute the optimal plan from a coor to a target");
    m.def("_get_target_plan_dist", &BaseHouse::_get_target_plan_dist, "graph: get the distances for executing a particular plan");
    m.def("_get_target_mask", &BaseHouse::_get_target_mask, "graph: get the target mask for a particular coor");
    m.def("_get_target_mask_names", &BaseHouse::_get_target_mask_names, "graph: get the names of targets associated with a coor");


    py::class_<BaseHouse>(m, "BaseHouse")
        // Init Function
        .def(py::init<int>())
        .def("_setHouseBox", &BaseHouse::_setHouseBox)
        // Cache Setter Function
        .def("_setObsMap", &BaseHouse::_setObsMap)
        .def("_setMoveMap", &BaseHouse::_setMoveMap)
        // Core Generation Function
        // .def("_genObstacleMap", &BaseHouse::_genObstacleMap, py::return_value_policy::reference) // TODO
        .def("_genMovableMap", &BaseHouse::_genMovableMap)
        .def("_genShortestDistMap", &BaseHouse::_genShortestDistMap)
        .def("_genOutsideDistMap", &BaseHouse::_genOutsideDistMap)
        .def("_genValidCoors", &BaseHouse::_genValidCoors)
        // Target Dist Map Setter Functions
        .def("_clearCurrentDistMap", &BaseHouse::_clearCurrentDistMap)
        .def("_setCurrentDistMap", &BaseHouse::_setCurrentDistMap)
        // Getter Functions
        .def("_getConnMap", &BaseHouse::_getConnMap, py::return_value_policy::reference)
        .def("_getConnCoors", &BaseHouse::_getConnCoors, py::return_value_policy::reference)
        .def("_getValidCoors", &BaseHouse::_getValidCoors, py::return_value_policy::reference)
        .def("_getMaxConnDist", &BaseHouse::_getMaxConnDist)
        // Location Getter Utility Functions
        .def("_fetchValidCoorsSize", &BaseHouse::_fetchValidCoorsSize)
        .def("_getCachedIndexedValidCoor", &BaseHouse::_getCachedIndexedValidCoor)
        .def("_getConnectCoorsSize", &BaseHouse::_getConnectCoorsSize)
        .def("_getConnectCoorsSize_Bounded", &BaseHouse::_getConnectCoorsSize_Bounded)
        .def("_getIndexedConnectCoor", &BaseHouse::_getIndexedConnectCoor)
        .def("_getCurrConnectCoorsSize", &BaseHouse::_getCurrConnectCoorsSize)
        .def("_getCurrConnectCoorsSize_Bounded", &BaseHouse::_getCurrConnectCoorsSize_Bounded)
        .def("_getCurrIndexedConnectCoor", &BaseHouse::_getCurrIndexedConnectCoor)
        // Utility Functions
        .def("_inside", &BaseHouse::_inside)
        .def("_canMove", &BaseHouse::_canMove)
        .def("_isConnect", &BaseHouse::_isConnect)
        .def("_getDist", &BaseHouse::_getDist)
        .def("_getScaledDist", &BaseHouse::_getScaledDist)
        .def("_rescale", &BaseHouse::_rescale)
        .def("_to_grid", &BaseHouse::_to_grid)
        .def("_to_coor", &BaseHouse::_to_coor)
        // Connectivity Graph Related
        .def("_gen_target_graph", &BaseHouse::_gen_target_graph)
        .def("_compute_target_plan", &BaseHouse::_compute_target_plan)
        .def("_get_target_plan_dist", &BaseHouse::_get_target_plan_dist)
        .def("_get_target_mask", &BaseHouse::_get_target_mask)
        .def("_get_target_mask_names", &BaseHouse::_get_target_mask_names)
        // Collision Checker
        .def("_check_occupy", &BaseHouse::_check_occupy)
        .def("_full_collision_check", &BaseHouse::_full_collision_check)
        .def("_fast_collision_check", &BaseHouse::_fast_collision_check)
        // Member Fields
        .def_readonly("obsMap", &BaseHouse::obsMap)
        .def_readonly("moveMap", &BaseHouse::moveMap);
}
