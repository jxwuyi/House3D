//File: house.hh

#pragma once

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


namespace render {

#define PII pair<int,int>
#define BOX_TP tuple<double,double,double,double>
#define REGION_TP tuple<int,int,int,int>
#define COMP_TP vector<PII>
#define COMP_PTR shared_ptr<vector<PII>>

// Implement some methods about houses that are too slow to do in python
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
    vector<py::array_t<unsigned char> > supMapLis;
    py::array_t<unsigned char>* cur_supMap;
    int cur_maxConnDist;
    vector<vector<tuple<int,int> > > regValidCoorsLis;
    vector<tuple<int,int,int,int> > regInputBoxLis;
    vector<tuple<int,int> > regExpandMaskLis;
    vector<tuple<int,int> >* last_regValidCoors;
    map<string, int> targetInd;  // index for targets <connMapLis, connCoorsLis, maxConnDistLis>
    vector<string> targetNames;  // list of target names
    map<string, int> regionInd; // index for <regValidCoorsLis>
    vector<string> regionNames; // list of region names

    ////////////////////////////////////////
    // for connectivity
    int n_obj, n_target, n_room;
    py::array_t<int> targetMask;   // the mask of whether this cell belongs to a target
    vector<vector<int> > targetDistGraph;  // pairwise distance over targets
  public:
    int _gen_target_graph(int _n_obj);  // return the total number of targets
    vector<vector<int> >* _get_target_graph(); // return a copy of the internal target graph
    vector<string> _compute_target_plan(double cx, double cy, const string& target);
    vector<double> _get_target_plan_dist(double cx, double cy, const vector<string>& plan);
    int _get_target_mask_grid(int gx, int gy, bool only_object=false);
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
    bool _check_grid_occupy(double cx, double cy, int gx, int gy);

public:
    BaseHouse(int resolution): n(resolution), cur_connMap(nullptr), cur_connCoors(nullptr), cur_supMap(nullptr) {}
    void _setHouseBox(double _lo, double _hi, double _rad) {
        L_lo = _lo; L_hi = _hi; rad = _rad;
        L_det = _hi - _lo; grid_det = L_det / n;
    }

    //////////////////////////////////
    // Cache Setter Functions
    //////////////////////////////////
    // set obsMap from python
    void _setObsMap(const py::array_t<int>& val);
    // set moveMap from python
    void _setMoveMap(const py::array_t<int>& val);

    ///////////////////////////////////
    // Core Generator Functions
    ///////////////////////////////////
    // generate obstacle map
    //  -> if retObject is false, return nullptr and store in obsMap
    /*
    // TODO
    py::array_t<int>* _genObstacleMap(double c_x1, double c_y1, double c_x2, double c_y2, int n_row,
                                      const BOX_TP& all_walls, const BOX_TP& door_obj, const BOX_TP& colide_obj,
                                      bool retObject);  // TODO
    */

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
    // compute target mask and connected mask for coors in a region
    bool _genExpandedRegionMask(const string& reg_tag);
    // compute target mask and connected mask for coors from a target region <tag>
    //  ---> NOTE: this only remains the largest component from <tag> region [TODO: decompose into several components?]
    //             this will also cache valid coors in <regValidCoorsLis> and <regionInd>
    bool _genExpandedRegionMaskFromTargetMap(const string& tag);
    // compute the target mask and connected mask for room mask <mask>
    //   ---> specially designed for computing masks for region <indoor>
    tuple<int,int> _genExpandedRegionMaskForRoomMask(int mask, int n_room);

    // compute supervision map for all the cached targets
    //    --> if already computed, false will be returned
    bool _genSupervisionMap(const vector<tuple<double,double,double,double>>& angle_dirs,
                            const vector<tuple<double,double,int>>& actions);

    /////////////////////////////////////////
    // Target Dist Map Setter Functions
    /////////////////////////////////////////
    // clear and set current distance info
    void _clearCurrentDistMap();
    bool _setCurrentDistMap(const string& tag);

    ///////////////////////////
    // Getter Functions
    ///////////////////////////
    // get connMap
    py::array_t<int>* _getConnMap() {return cur_connMap;}
    int _getConnDistForTarget(const string& tag, int gx, int gy);
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
    // get supervision signal
    int _getSupervise(int gx, int gy, int deg);

    //////////////////////////////////////
    // location getter utility functions
    //////////////////////////////////////
    tuple<int,int> _getRegionMask(const string& reg_tag);
    // get valid coors and cache it
    int _fetchValidCoorsSize(const string& reg_tag);
    // return indexed coor from cached list
    tuple<int,int> _getCachedIndexedValidCoor(int k);
    int _getConnectCoorsSize(const string& tag);
    int _getConnectCoorsSize_Bounded(const string& tag, int bound);
    tuple<int,int> _getConnectCoorsSize_Range(const string& tag, int lo, int hi);
    tuple<int,int> _getIndexedConnectCoor(const string& tag, int k);
    int _getCurrConnectCoorsSize();
    int _getCurrConnectCoorsSize_Bounded(int bound);
    tuple<int,int> _getCurrIndexedConnectCoor(int k);
    bool _fetchSupervisionMap(const string& tag);

    //////////////////////////////////////////////////
    // range check and utility functions
    //////////////////////////////////////////////////
    bool _inside(int x, int y);
    bool _canMove(int x, int y);
    bool _isConnect(int x, int y);
    int _getDist(int x, int y);
    double _getScaledDist(int x, int y);
    tuple<int,int,int,int> _rescale(double x1, double y1, double x2, double y2, int n_row);
    tuple<int,int> _to_grid(double x, double y, int n_row);
    tuple<double,double> _to_coor(int x, int y, bool shft);

    ///////////////////////////////////
    // Collision Check (assume (Ax, Ay) is a good point, check whether can move to (Bx, By))
    ///////////////////////////////////
    bool _check_occupy(double cx, double cy);
    bool _full_collision_check(double Ax, double Ay, double Bx, double By, int num_samples);
    bool _fast_collision_check(double Ax, double Ay, double Bx, double By, int num_samples);
};

}
