//File: house.cc

#include "house.hh"

namespace render {

#define MKP(a,b) make_pair(a,b)
#define X first
#define Y second
#define INDEX(x,y,n) ((x) * (n) + (y))
#define STATE_ID(x,y,d,n,nd) (INDEX(x,y,n)*(nd)+(d))

const double eps = 1e-9;
const int DIRS[4][2] = {{0,1},{1,0},{0,-1},{-1,0}};

template<class T>
T* _get_mem(int size, T val=0) {
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

//////////////////////////////////
// Cache Setter Functions
//////////////////////////////////
// set obsMap from python
void BaseHouse::_setObsMap(const py::array_t<int>& val) {
    if (val.ndim() != 2 && val.shape(0) == val.shape(1) && val.shape(0) == n+1) {  // NOTE: n+1!!
        cerr << "[C++] <BaseHouse._setObsMap> Input ndarray must be a 2-d squared matrix!" << endl;
        assert(false);
    }
    int m = val.shape(0);
    obsMap = py::array_t<int>({m, m}, new int[m * m + 1]);
    for(int i=0;i<m;++i) for(int j=0;j<m;++j) *obsMap.mutable_data(i,j) = *val.data(i,j);
}
// set moveMap from python
void BaseHouse::_setMoveMap(const py::array_t<int>& val) {
    if (val.ndim() != 2 && val.shape(0) == val.shape(1) && val.shape(0) == n + 1) { // NOTE: n+1!!
        cerr << "[C++] <BaseHouse._setMoveMap> Input ndarray must be a 2-d squared matrix!" << endl;
        assert(false);
    }
    int m = val.shape(0);
    moveMap = py::array_t<int>({m, m}, new int[m * m + 1]);
    for(int i=0;i<m;++i) for(int j=0;j<m;++j) *moveMap.mutable_data(i,j) = *val.data(i,j);
}

/////////////////////////////////////////
// Target Dist Map Setter Functions
/////////////////////////////////////////
// clear and set current distance info
void BaseHouse::_clearCurrentDistMap() {
    cur_connMap = nullptr;
    cur_connCoors = nullptr;
    cur_maxConnDist = 0;
}

bool BaseHouse::_setCurrentDistMap(const string& tag) {
    auto iter = targetInd.find(tag);
    if (iter == targetInd.end()) return false;
    int k = iter->second;
    cur_connMap = &connMapLis[k];
    cur_connCoors = &connCoorsLis[k];
    cur_maxConnDist = maxConnDistLis[k];
    return true;
}

//////////////////////////////////////
// location getter utility functions
//////////////////////////////////////
tuple<int,int> BaseHouse::_getRegionMask(const string& reg_tag) {
  auto iter = regionInd.find(reg_tag);
  if (iter == regionInd.end()) return make_tuple(-1,-1);
  return regExpandMaskLis[iter->second];
}
// get valid coors and cache it
int BaseHouse::_fetchValidCoorsSize(const string& reg_tag) {
    auto iter = regionInd.find(reg_tag);
    if (iter == regionInd.end()) {
        last_regValidCoors = nullptr;
        return 0;
    }
    last_regValidCoors = &regValidCoorsLis[iter->second];
    return last_regValidCoors->size();
}
// return indexed coor from cached list
tuple<int,int> BaseHouse::_getCachedIndexedValidCoor(int k) {
    if (last_regValidCoors == nullptr) return make_tuple(-1, -1);
    return (*last_regValidCoors)[k];
}
int BaseHouse::_getConnectCoorsSize(const string& tag){
    auto iter = targetInd.find(tag);
    if (iter == targetInd.end()) return 0;
    return connCoorsLis[iter->second].size();
}
int BaseHouse::_getConnectCoorsSize_Bounded(const string& tag, int bound){
    auto iter = targetInd.find(tag);
    if (iter == targetInd.end() || bound < 0) return 0;
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
tuple<int,int> BaseHouse::_getConnectCoorsSize_Range(const string& tag, int lo, int hi) {
    return make_tuple(_getConnectCoorsSize_Bounded(tag, lo - 1),
                      _getConnectCoorsSize_Bounded(tag, hi));
}
tuple<int,int> BaseHouse::_getIndexedConnectCoor(const string& tag, int k){
    auto iter = targetInd.find(tag);
    if (iter == targetInd.end()) return make_tuple(-1, -1);
    return connCoorsLis[iter->second][k];
}
int BaseHouse::_getCurrConnectCoorsSize(){
    if (cur_connCoors == nullptr) return 0;
    return cur_connCoors->size();
}
int BaseHouse::_getCurrConnectCoorsSize_Bounded(int bound){
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
tuple<int,int> BaseHouse::_getCurrIndexedConnectCoor(int k){ return (*cur_connCoors)[k]; }

// fetch function for supervision signals
bool BaseHouse::_fetchSupervisionMap(const string& tag) {
    auto iter = targetInd.find(tag);
    if (iter == targetInd.end() || iter->second >= (int)supMapLis.size()) return false;
    cur_supMap = &supMapLis[iter->second];
    return true;
}

// get entry of connMap under target <tag>
int BaseHouse::_getConnDistForTarget(const string& tag, int gx, int gy) {
    auto iter = targetInd.find(tag);
    if (iter == targetInd.end() || !_inside(gx, gy)) return -1;
    return *connMapLis[iter->second].data(gx, gy);
}

// get supervision for current grid (gx, gy, deg)
int BaseHouse::_getSupervise(int gx, int gy, int deg) {
    if (cur_supMap == nullptr) return -1;
    int mask = *(*cur_supMap).data(gx, gy, deg);
    if (mask == 0) return -1;
    return __builtin_ctz(mask);
}

//////////////////////////////////////////////////
// range check and utility functions
//////////////////////////////////////////////////
bool BaseHouse::_inside(int x, int y) {return x <= n && x >= 0 && y <= n && y >= 0;}
bool BaseHouse::_canMove(int x, int y) {return _inside(x,y) && *moveMap.data(x,y) > 0;}
bool BaseHouse::_isConnect(int x, int y) {return _inside(x, y) && *(*cur_connMap).data(x,y) != -1;}
int BaseHouse::_getDist(int x, int y) {return *(*cur_connMap).data(x,y);}
double BaseHouse::_getScaledDist(int x, int y) {
    int ret = *(*cur_connMap).data(x,y);
    if (ret < 0) return (double)(ret);
    return (double)(ret) / cur_maxConnDist;
}
tuple<int,int,int,int> BaseHouse::_rescale(double x1, double y1, double x2, double y2, int n_row) {
    int tx1 = (int)floor((x1 - L_lo) / L_det * n_row+eps);
    int ty1 = (int)floor((y1 - L_lo) / L_det * n_row+eps);
    int tx2 = (int)floor((x2 - L_lo) / L_det * n_row+eps);
    int ty2 = (int)floor((y2 - L_lo) / L_det * n_row+eps);
    return make_tuple(tx1,ty1,tx2,ty2);
}
tuple<int,int> BaseHouse::_to_grid(double x, double y, int n_row) {
    int tx = (int)floor((x - L_lo) / L_det * n_row+eps);
    int ty = (int)floor((y - L_lo) / L_det * n_row+eps);
    return make_tuple(tx, ty);
}
tuple<double,double> BaseHouse::_to_coor(int x, int y, bool shft) {
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
bool BaseHouse::_check_occupy(double cx, double cy) {
    int x1,y1,x2,y2;
    tie(x1,y1,x2,y2) = _rescale(cx-rad,cy-rad,cx+rad,cy+rad,n);
    for(int x=x1;x<=x2;++x)
        for(int y=y1;y<=y2;++y)
            if ((!_inside(x,y) || *obsMap.data(x,y) == 1) && _check_grid_occupy(cx, cy, x, y)) return false;
    return true;
}
bool BaseHouse::_full_collision_check(double Ax, double Ay, double Bx, double By, int num_samples) {
    double ratio = 1.0 / num_samples;
    Bx -= Ax; By -= Ay;
    for(int i=1;i<=num_samples;++i) {
        double px = i * ratio * Bx + Ax, py = i * ratio * By + Ay;
        if (!_check_occupy(px, py)) return false;
    }
    return true;
}
bool BaseHouse::_fast_collision_check(double Ax, double Ay, double Bx, double By, int num_samples) {
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

// get the target distance graph (a copy)
vector<vector<int> > BaseHouse::_get_target_graph() {
    return targetDist;
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
  for(size_t i=0;i<plan.size();++i) {
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

// get the target mask for grid(gx, gy)
//    when only_object == True, only return object_target bits
int BaseHouse::_get_target_mask_grid(int gx, int gy, bool only_object) {
  int mask = *targetMask.data(gx, gy);
  if (only_object) mask &= (1 << n_room) - 1;
  return mask;
}

// get the target mask for coor(cx, cy)
int BaseHouse::_get_target_mask(double cx, double cy, bool only_object) {
  int gx, gy;
  tie(gx, gy) = _to_grid(cx, cy, n);
  return _get_target_mask_grid(gx, gy, only_object);
}

// return the target names associated with coor (cx, cy)
vector<string> BaseHouse::_get_target_mask_names(double cx, double cy, bool only_object) {
  int mask = this->_get_target_mask(cx, cy, only_object);
  vector<string> targets;
  for(; mask > 0; mask &= mask-1)
    targets.push_back(targetNames[__builtin_ctz(mask)]);
  return targets;
}

//////////////////////////////////////////
// check whether agent can occupy (cx,cy)
//       by avoiding (gx,gy)
bool BaseHouse::_check_grid_occupy(double cx, double cy, int gx, int gy) {
    for(int x=gx;x<=gx+1;++x)
        for(int y=gy;y<=gy+1;++y) {
            double rx = x * grid_det + L_lo, ry = y * grid_det + L_lo;
            rx -= cx; ry -= cy;
            if (rx * rx + ry * ry <= rad * rad) return true;
        }
    return false;
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
    regInputBoxLis.push_back(make_tuple(x1,y1,x2,y2));
    regionInd[reg_tag] = k;
    regValidCoorsLis.push_back(vector<tuple<int,int> >({}));
    regExpandMaskLis.push_back(make_tuple(-1,-1));
    auto& coors = regValidCoorsLis[k];
    auto dat = _find_components(x1, y1, x2, y2, true, false);
    auto& comp = dat[0]; // the largest components
    for(auto& p: *comp)
        coors.push_back(make_tuple(p.X, p.Y));
    return true;
}

// generate expanded mask for a region
bool BaseHouse::_genExpandedRegionMask(const string& reg_tag) {
  auto iter = regionInd.find(reg_tag);
  if (iter == regionInd.end()) return false;  // no such region
  int k = iter->second;
  int in_msk, out_msk;
  tie(in_msk, out_msk) = regExpandMaskLis[k];
  if (in_msk > -1) return true; // do not need to compute again
  in_msk = out_msk = 0;
  bool is_open = false;
  int x, y, x1, y1, x2, y2;
  tie(x1,y1,x2,y2) = regInputBoxLis[k];
  for (auto& coor: regValidCoorsLis[k]) {
    tie(x,y) = coor;
    in_msk |= *targetMask.data(x,y);
    for (int d=0;d<4;++d) {
      int tx = x + DIRS[d][0], ty = y + DIRS[d][1];
      if (_canMove(tx, ty) &&
          (tx < x1 || tx > x2 || ty < y1 || ty > y2)) {
          out_msk |= *targetMask.data(tx,ty);
          is_open = true;
      }
    }
  }
  if (! is_open) out_msk = -1;
  regExpandMaskLis[k] = make_tuple(in_msk, out_msk);
  return true;
}

// generate expanded mask for a target region <tag>
bool BaseHouse::_genExpandedRegionMaskFromTargetMap(const string& tag) {
    auto iter = targetInd.find(tag);
    if (iter == targetInd.end()) return false;
    if (regionInd.count(tag) > 0) return false;
    auto& valid_coors = connCoorsLis[iter->second];
    int sz = n + 1;
    int* idxMap = _get_mem<int>(sz*sz+1, -1);
    for(auto& coor: valid_coors) {
        int x,y; tie(x,y) = coor;
        idxMap[INDEX(x,y,sz)] = 0;
    }
    vector<vector<tuple<int,int> >> comps;
    vector<tuple<int,int> > cp_msk;
    int best_cp_id = -1;
    for(auto& coor: valid_coors) {
        int x,y; tie(x,y) = coor;
        if (idxMap[INDEX(x,y,sz)] == 0) { // find connected components
            idxMap[INDEX(x,y,sz)] = 1;
            vector<tuple<int,int> > que;
            int in_mask = 0, out_mask = 0;
            bool is_open = true;
            que.push_back(make_tuple(x,y));
            for(size_t ptr = 0; ptr < que.size(); ++ ptr) {
                tie(x,y) = que[ptr];
                in_mask |= *targetMask.data(x,y);
                for (int d=0;d<4;++d) {
                    int tx = x + DIRS[d][0], ty = y + DIRS[d][1];
                    if (_canMove(tx, ty)) {
                        if (idxMap[INDEX(tx,ty,sz)] == 0) {
                            que.push_back(make_tuple(tx,ty));
                            idxMap[INDEX(tx,ty,sz)] = 1;
                        } else
                        if (idxMap[INDEX(tx,ty,sz)] < 0) {
                            out_mask |= *targetMask.data(tx,ty);
                            is_open = true;
                        }
                    }
                }
            }
            if (! is_open) out_mask = -1;
            cp_msk.push_back(make_tuple(in_mask, out_mask));
            if (best_cp_id < 0 || que.size() > comps[best_cp_id].size())
                best_cp_id = comps.size();
            comps.push_back(que);
        }
    }
    int k = regValidCoorsLis.size();
    regionNames.push_back(tag);
    regInputBoxLis.push_back(make_tuple(-1,-1,-1,-1));
    regionInd[tag] = k;
    regValidCoorsLis.push_back(comps[best_cp_id]);
    regExpandMaskLis.push_back(cp_msk[best_cp_id]);
    delete idxMap;
    return true;
}

// generate expanded mask for region with room mask <mask>
tuple<int,int> BaseHouse::_genExpandedRegionMaskForRoomMask(int mask, int n_room) {
    int sz = n + 1;
    int room_bits = (1 << n_room) - 1;
    int in_mask = 0, out_mask = 0;
    bool flag_found_desired_mask = false;
    for (int x=0;x<sz;++x)
        for (int y=0;y<sz;++y)
            if ((*moveMap.data(x,y) > 0) && (*targetMask.data(x,y) & room_bits) == mask) {
                flag_found_desired_mask = true;
                in_mask |= *targetMask.data(x,y);
                for (int d=0;d<4;++d) {
                    int tx = x + DIRS[d][0], ty = y + DIRS[d][1];
                    if (_inside(tx, ty) && *moveMap.data(tx,ty) > 0) {
                        out_mask |= *targetMask.data(tx, ty);
                    }
                }
            }
    if (!flag_found_desired_mask)
        return make_tuple(int(-1), int(-1));
    return make_tuple(in_mask, out_mask);
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


// compute supervision map for all the cached targets
//    --> if already computed, false will be returned
bool BaseHouse::_genSupervisionMap(const vector<tuple<double,double,double,double>>& angle_dirs,
                                   const vector<tuple<double,double,int>>& actions)
{
    if (supMapLis.size() == connMapLis.size()) return false;  // already computed before
    int n_deg = angle_dirs.size();
    int sz = n + 1;
    int n_target = connMapLis.size();
    int opt_size = sz * sz * n_deg + 1;
    int* opt = _get_mem<int>(opt_size,-1);
    unsigned char* avail_actions = _get_mem<unsigned char>(opt_size, 0);
    for (int gx=0;gx<sz;++gx)
        for (int gy=0;gy<sz;++gy) {
            if (*moveMap.data(gx,gy) <= 0) continue;
            // get continuous location (cx,cy)
            double cx, cy;
            tie(cx,cy) = _to_coor(gx,gy,true);
            for (int td=0;td<n_deg;++td) {
                int state = STATE_ID(gx,gy,td,sz,n_deg);
                // fetch direction vector
                double dx_frt, dy_frt, dx_rght, dy_rght;
                tie(dx_frt, dy_frt, dx_rght, dy_rght) = angle_dirs[td];
                // enumerate action
                for(size_t i=0;i<actions.size();++i) {
                    double det_frt, det_rght;
                    int det_deg;
                    tie(det_frt, det_rght, det_deg) = actions[i];
                    if (det_deg != 0) // only rotation
                        avail_actions[state] |= 1 << (unsigned char)(i);
                    else {  // move action
                        double sx, sy; // deg keeps unchanged!!
                        sx = cx - dx_frt * det_frt - dx_rght * det_rght;
                        sy = cy - dy_frt * det_frt - dy_rght * det_rght;
                        int tx, ty;
                        tie(tx, ty) = _to_grid(sx, sy, n);
                        if (tx <= 0 || ty <= 0 || tx >= n || ty >= n || *moveMap.data(tx, ty) <= 0) continue;
                        if (!_fast_collision_check(cx, cy, sx, sy, 10)) continue;
                        avail_actions[state] |= 1 << (unsigned char)(i);
                    }
                }
            }
        }
    for (int k = (int)supMapLis.size(); k < n_target; ++ k) {
        memset(opt, -1, sizeof(int) * opt_size);
        auto& connMap = connMapLis[k];
        auto& coors = connCoorsLis[k];
        supMapLis.push_back(py::array_t<unsigned char>({sz, sz, n_deg}, _get_mem<unsigned char>(sz * sz * n_deg + 1, 0)));
        auto& supMap = supMapLis[k];
        queue<int> que;
        for(size_t i=0;i<coors.size();++i) {
            int gx, gy;
            tie(gx,gy) = coors[i];
            int dist = *connMap.data(gx,gy);
            if (dist * grid_det > 0.15) break;  // start with locations within the target range
            for(int d=0;d<n_deg;++d) {
                int state = STATE_ID(gx,gy,d,sz,n_deg);
                opt[state] = 0;
                que.push(state);
            }
        }
        while(!que.empty()) {
            int state = que.front(); que.pop();
            int step = opt[state];
            // fetch (x,y,d)
            int gx,gy,td,_t_state=state;
            td = _t_state % n_deg; _t_state /= n_deg;
            gy = _t_state % sz; gx = _t_state / sz;
            // get continuous location (cx,cy)
            double cx, cy;
            tie(cx,cy) = _to_coor(gx,gy,true);
            // fetch direction vector
            double dx_frt, dy_frt, dx_rght, dy_rght;
            tie(dx_frt, dy_frt, dx_rght, dy_rght) = angle_dirs[td];
            // check all valid actions
            int action_mask = avail_actions[state];
            //if (step == 0) action_mask &= 1;   // NOTE: last action must be forward!!!!
            for(; action_mask > 0; action_mask &= action_mask - 1) {
                int i = __builtin_ctz(action_mask);
                double det_frt, det_rght;
                int det_deg;
                tie(det_frt, det_rght, det_deg) = actions[i];
                int nxt_x = gx, nxt_y = gy, nxt_d = td;
                if (det_deg == 0) {
                    // this is a move action!
                    double sx, sy;
                    sx = cx - dx_frt * det_frt - dx_rght * det_rght;
                    sy = cy - dy_frt * det_frt - dy_rght * det_rght;
                    tie(nxt_x, nxt_y) = _to_grid(sx, sy, n);
                } else {
                    nxt_d = (td - det_deg + n_deg) % n_deg;
                }
                // from (nxt_x,nxt_y,nxt_d) take action_i, lead to (gx, gy, td)
                int nxt_state = STATE_ID(nxt_x, nxt_y, nxt_d, sz, n_deg);
                if (opt[nxt_state] < 0) {
                    opt[nxt_state] = step + 1;
                    que.push(nxt_state);
                    *supMap.mutable_data(nxt_x, nxt_y, nxt_d) |= 1 << (unsigned char)(i);
                } else
                    if (opt[nxt_state] == step + 1) {
                        *supMap.mutable_data(nxt_x, nxt_y, nxt_d) |= 1 << (unsigned char)(i);
                    }
            }
        }
    }
    delete opt;
    delete avail_actions;
    return true;
}

}
