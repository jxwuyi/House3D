# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import time
import sys
import numpy as np
import json
import csv
import pickle
import itertools
import copy

import pdb

from base_house import BaseHouse
#from .base_house_api import BaseHouse

__all__ = ['House']

######################################
# Util Functions
######################################
# allowed target room types
# NOTE: consider "toilet" and "bathroom" the same thing
ALLOWED_TARGET_ROOM_TYPES = ['outdoor', 'kitchen', 'dining_room', 'living_room', 'bathroom', 'bedroom', 'office', 'garage']

ALLOWED_OBJECT_TARGET_TYPES = ['kitchen_cabinet','sofa','chair','toilet','table', 'sink','wardrobe_cabinet','bed',
                               'shelving','desk','television','household_appliance','dresser','vehicle','pool']
                               #'table_and_chair']


#ALLOWED_OBJECT_TARGET_TYPES = ['shower', 'sofa', 'toilet', 'bed', 'television',
#                               'table', 'kitchen_set', 'bathtub', 'vehicle', 'pool', 'kitchen_cabinet']

# allowed room types for auxiliary prediction task
ALLOWED_PREDICTION_ROOM_TYPES = dict(
    outdoor=0, kitchen=1, dining_room=2, living_room=3, bathroom=4, bedroom=5, office=6, garage=7)

ALLOWED_OBJECT_TARGET_INDEX = dict({'sofa': 9, 'desk': 17, 'sink': 13, 'wardrobe_cabinet': 14, 'bed': 15,
                                    'kitchen_cabinet': 8, 'shelving': 16, 'dresser': 20, 'chair': 10, 'television': 18,
                                    'toilet': 11, 'vehicle': 21, 'table': 12, 'pool': 22, 'household_appliance': 19})
                                    #'table_and_chair': (10, 12)})

def _equal_room_tp(room, target):
    """
    NOTE: Ensure <target> is always from <ALLOWED_TARGET_ROOM_TYPES>!!!!
            DO NOT swap the order of arguments
    """
    room = room.lower()
    return (room == target) or \
            ((target == 'bathroom') and (room == 'toilet')) or \
            ((target == 'bedroom') and (room == 'guest_room'))


def _equal_object_tp(obj, target):
    """
    NOTE: Ensure <target> is always from <ALLOWED_OBJECT_TARGET_TYPES>!!!!
            DO NOT swap the order of arguments
    """
    obj = obj.lower()
    return (obj == target) or \
           ((target == 'chair') and (obj == 'table_and_chair')) or \
           ((target == 'table') and (obj == 'table_and_chair'))


def _get_object_categories(obj):
    obj = obj.lower()
    if obj == 'table_and_chair':
        return ['table', 'chair']
    else:
        return [obj]


def _get_pred_room_tp_id(room):
    room = room.lower()
    if room == 'toilet':
        room = 'bathroom'
    elif room == 'guest_room':
        room = 'bedroom'
    if room not in ALLOWED_PREDICTION_ROOM_TYPES:
        return ALLOWED_PREDICTION_ROOM_TYPES['indoor']
    return ALLOWED_PREDICTION_ROOM_TYPES[room]


def parse_walls(objFile, lower_bound = 1.0):
    def create_box(vers):
        if len(vers) == 0:
            return None
        v_max = [-1e20, -1e20, -1e20]
        v_min = [1e20, 1e20, 1e20]
        for v in vers:
            for i in range(3):
                if v[i] < v_min[i]: v_min[i] = v[i]
                if v[i] > v_max[i]: v_max[i] = v[i]
        obj = dict()
        obj['bbox'] = dict()
        obj['bbox']['min']=v_min
        obj['bbox']['max']=v_max
        if v_min[1] < lower_bound:
            return obj
        return None
    walls = []
    with open(objFile, 'r') as file:
        vers = []
        for line in file.readlines():
            if len(line) < 2: continue
            if line[0] == 'g':
                if (vers is not None) and (len(vers) > 0): walls.append(create_box(vers))
                if ('Wall' in line):
                    vers = []
                else:
                    vers = None
            if (vers is not None) and (line[0] == 'v') and (line[1] == ' '):
                vals = line[2:]
                coor =[float(v) for v in vals.split(' ') if len(v)>0]
                if len(coor) != 3:
                    print('line = {}'.format(line))
                    print('coor = {}'.format(coor))
                    assert(False)
                vers.append(coor)
    if (vers is not None) and (len(vers) > 0): walls.append(create_box(vers))
    ret_walls = [w for w in walls if w is not None]
    return ret_walls


def fill_region(proj,x1,y1,x2,y2,c):
    proj[x1:(x2+1), y1:(y2+1)] = c

def fill_obj_mask(house, dest, obj, c=1):
    n_row = dest.shape[0]
    _x1, _, _y1 = obj['bbox']['min']
    _x2, _, _y2 = obj['bbox']['max']
    x1,y1,x2,y2 = house.rescale(_x1,_y1,_x2,_y2,n_row)
    fill_region(dest, x1, y1, x2, y2, c)


class House(BaseHouse):
    """core class for loading and processing a house from SUNCG dataset
    """
    def __init__(self, JsonFile, ObjFile, MetaDataFile,
                 MapTargetCatFile=None,
                 CachedFile=None,
                 StorageFile=None,
                 GenRoomTypeMap=False,
                 EagleViewRes=0,
                 DebugInfoOn=False,
                 ColideRes=1000,
                 RobotRadius=0.1,
                 RobotHeight=0.75,  # 1.0,
                 CarpetHeight=0.15,
                 ObjectTargetSuccRange = 1.0,
                 SetTarget=True,
                 IncludeOutdoorTarget=False,
                 _IgnoreSmallHouse=False  # should be only set true when called by "cache_houses.py"
                 ):
        """Initialization and Robot Parameters

         Note:
            Generally only the first 4 arguments are required to set up a house
            Ensure you run the script to generate cached data for all the houses

        Args:
            JsonFile (str): file name of the house json file (house.json)
            ObjFile (str): file name of the house object file (house.obj)
            MetaDataFile (str): file name of the meta data (ModelCategoryMapping.csv)
            MapTargetCatFile (str, required when using object target): file name of the target object category file (map_modelid_to_targetcat.json)
            CachedFile (str, recommended): file name of the pickled cached data for this house, None if no such cache (cachedmap1k.pkl)
            StorageFile (str, optional): if CachedFile is None, pickle all the data and store in this file
            GenRoomTypeMap (bool, optional): if turned on, generate the room type map for each location
            EagleViewRes (int, optional): resolution of the smaller topdown 2d map, default 0 (do not build eagle_map)
            DebugInfoOn (bool, optional): store additional debugging information when this option is on
            ColideRes (int, optional): resolution of the 2d map for collision check (generally should not changed)
            RobotRadius (double, optional): radius of the robot/agent (generally should not be changed)
            RobotHeight (double, optional): height of the robot/agent (generally should not be changed)
            CarpetHeight (double, optional): maximum height of the obstacles that agent can directly go through (gennerally should not be changed)
            ObjectTargetSuccRange (double, optional): range for determining success of finding a object target
            SetTarget (bool, optional): whether or not to choose a default target room and pre-compute the valid locations
            IncludeOutdoorTarget (bool, optional): when true, we allow target room <outdoor> indicating not inside any room regions
        """
        super(House, self).__init__(ColideRes)  # initialize parent class

        ts = time.time()
        print('Data Loading ...')

        self.metaDataFile = MetaDataFile
        self.objFile = ObjFile
        self.robotHei = RobotHeight
        self.carpetHei = CarpetHeight
        self.robotRad = RobotRadius
        self.objTargetRange = ObjectTargetSuccRange
        self.includeOutdoorTarget = IncludeOutdoorTarget
        self._debugMap = None if not DebugInfoOn else True
        with open(JsonFile) as jfile:
            self.house = house = json.load(jfile)

        # parse walls
        self.all_walls = parse_walls(ObjFile, RobotHeight)

        # validity check
        if abs(house['scaleToMeters'] - 1.0) > 1e-8:
            print('[Error] Currently <scaleToMeters> must be 1.0!')
            assert(False)
        if len(house['levels']) > 1:
            print('[Warning] Currently only support ground floor! <total floors = %d>' % (len(house['levels'])))

        # parse house coordinate range
        self.level = level = house['levels'][0]  # only support ground floor now
        self.L_min_coor = _L_lo = np.array(level['bbox']['min'])
        self.L_lo = min(_L_lo[0], _L_lo[2])
        self.L_max_coor = _L_hi = np.array(level['bbox']['max'])
        self.L_hi = max(_L_hi[0], _L_hi[2])
        self.L_det = self.L_hi - self.L_lo
        self.n_row = ColideRes
        self.eagle_n_row = EagleViewRes
        self.grid_det = self.L_det / self.n_row
        self._setHouseBox(self.L_lo, self.L_hi, self.robotRad)  # set coordinate range of the C++ side

        # parse objects and room/object types
        self.all_obj = [node for node in level['nodes'] if node['type'].lower() == 'object']
        self.all_rooms = [node for node in level['nodes'] if (node['type'].lower() == 'room') and ('roomTypes' in node)]
        self.all_roomTypes = [room['roomTypes'] for room in self.all_rooms]
        self.all_desired_roomTypes = []
        self.default_roomTp = None
        for roomTp in ALLOWED_TARGET_ROOM_TYPES:
            if any([any([_equal_room_tp(tp, roomTp) for tp in tps]) for tps in self.all_roomTypes]):
                self.all_desired_roomTypes.append(roomTp)
                if self.default_roomTp is None: self.default_roomTp = roomTp
        assert self.default_roomTp is not None, 'Cannot Find Any Desired Rooms!'
        # check whether need to include <outdoor>
        if self.includeOutdoorTarget: self.all_desired_roomTypes.append('outdoor')
        print('>> Default Target Room Type Selected = {}'.format(self.default_roomTp))

        # prepare object targets
        self.tar_obj_region = dict()
        self.all_desired_targetObj = []
        self.id_to_tar = dict()
        if MapTargetCatFile is not None:
            print('Preparing Target Object List ...')
            self.genTargetObjectList(MapTargetCatFile)
            self.all_desired_targetObj = list(self.tar_obj_region.keys())
            print('>> Total Target Object Types = <N={}> {}'.format(len(self.all_desired_targetObj),
                                                                    self.all_desired_targetObj))
        self.all_desired_targetTypes = self.all_desired_roomTypes + self.all_desired_targetObj

        print('  --> Done! Elapsed = %.2fs' % (time.time()-ts))
        if _IgnoreSmallHouse and ((len(self.all_desired_roomTypes) < 2) or ('kitchen' not in self.all_desired_roomTypes)):
            self.all_desired_roomTypes = []
            return

        # generate a low-resolution obstacle map
        self.tinyObsMap = None
        self.eagleMap = None
        if self.eagle_n_row > 1:
            print('Generating Low Resolution Obstacle Map ...')
            ts = time.time()
            self.tinyObsMap = np.ones((self.eagle_n_row, self.eagle_n_row), dtype=np.uint8)
            self.eagleMap = np.zeros((4, self.eagle_n_row, self.eagle_n_row), dtype=np.uint8)
            self.genObstacleMap(MetaDataFile, gen_debug_map=False, dest=self.tinyObsMap, n_row=self.eagle_n_row-1)
            self.eagleMap[0, ...] = self.tinyObsMap
            print('  --> Done! Elapsed = %.2fs' % (time.time()-ts))

        # load from cache
        if CachedFile is not None:
            assert not DebugInfoOn, 'Please set DebugInfoOn=False when loading data from cached file!'
            print('Loading Obstacle Map and Movability Map From Cache File ...')
            ts = time.time()
            with open(CachedFile, 'rb') as f:
                t_obsMap, t_moveMap = pickle.load(f)
            self._setObsMap(t_obsMap)
            self._setMoveMap(t_moveMap)
            del t_obsMap
            del t_moveMap
            print('  --> Done! Elapsed = %.2fs' % (time.time()-ts))
        else:
            # generate obstacle map
            print('Generate High Resolution Obstacle Map (For Collision Check) ...')
            ts = time.time()
            # obsMap was indexed by (x, y), not (y, x)
            t_obsMap = np.ones((self.n_row+1, self.n_row+1), dtype=np.uint8)  # a small int is enough
            if self._debugMap is not None:
                self._debugMap = np.ones((self.n_row+1, self.n_row+1), dtype=np.float)
            self.genObstacleMap(MetaDataFile, dest=t_obsMap)
            self._setObsMap(t_obsMap)
            del t_obsMap
            print('  --> Done! Elapsed = %.2fs' % (time.time()-ts))
            # generate movability map for robots considering the radius
            print('Generate Movability Map ...')
            ts = time.time()
            self.genMovableMap()
            print('  --> Done! Elapsed = %.2fs' % (time.time()-ts))

            if StorageFile is not None:
                print('Storing Obstacle Map and Movability Map to Cache File ...')
                ts = time.time()
                with open(StorageFile, 'wb') as f:
                    pickle.dump([self.obsMap, self.moveMap], f)
                print('  --> Done! Elapsed = %.2fs' % (time.time()-ts))

        # set target room connectivity
        print('Generate Target connectivity Map (Default <{}>) ...'.format(self.default_roomTp))
        ts = time.time()
        self.targetRoomTp = None
        if SetTarget:
            self.setTargetRoom(self.default_roomTp, _setEagleMap=True)
        self.setTargetRoom(self.default_roomTp)  # _setEagleMap=(self.eagle_n_row>0)
        print('  --> Done! Elapsed = %.2fs' % (time.time()-ts))

        self.roomTypeMap = None
        if GenRoomTypeMap:
            ts = time.time()
            print('Generate Room Type Map ...')
            self._generate_room_type_map()
            print('  --> Done! Elapsed = %.2fs' % (time.time() - ts))

    @property
    def connMap(self):
        return self._getConnMap()

    @property
    def inroomDist(self):
        return self._getInroomDist()

    @property
    def maxConnDist(self):
        return self._getMaxConnDist()

    @property
    def connectedCoors(self):
        return self._getConnCoors()

    def _generate_room_type_map(self):
        if self.roomTypeMap is None:
            self.roomTypeMap = np.zeros((self.n_row + 1, self.n_row + 1), dtype=np.uint16)
        rtMap = self.roomTypeMap
        # fill all the mask of rooms
        for room in self.all_rooms:
            msk = 1 << _get_pred_room_tp_id('indoor')
            for tp in room['roomTypes']:
                msk |= 1 << _get_pred_room_tp_id(tp)
            _x1, _, _y1 = room['bbox']['min']
            _x2, _, _y2 = room['bbox']['max']
            x1, y1, x2, y2 = self.rescale(_x1, _y1, _x2, _y2)
            array_msk = np.array(self.moveMap[x1:x2+1, y1:y2+1], dtype=np.uint16)
            np.clip(array_msk, 0, 1, out=array_msk)  # inplace clipping to 0-1 matrix
            array_msk *= msk
            rtMap[x1:x2+1, y1:y2+1] |= array_msk
        outdoor_msk = 1 << _get_pred_room_tp_id('outdoor')
        array_msk = np.ones(rtMap.shape, dtype=np.uint16) * outdoor_msk
        array_msk[rtMap > 0] = 0  # mark empty cells with positive flag
        array_msk[self.moveMap == 0] = 0  # exclude obstacle positions
        rtMap |= array_msk  # set <outdoor> flag to positions (1) movable & (2) originally not marked

    """
    Sets self.connMap to distances to target point with some margin
    """
    def setTargetPoint(self, x, y, margin_x=15, margin_y=15):
        tag = "targetPoint<x=%.5f,y=%.5f>" % (x, y)
        if self._setCurrentDistMap(tag):
            return True

        # compute shortest distance
        if not self._genShortestDistMap([(x-margin_x, y-margin_y, x+margin_x, y+margin_y)], tag):
            return False
        # shortest distance map computed successfully
        self._setCurrentDistMap(tag)
        self.targetRoomTp = tag
        return True

    """
    set the distance to a particular room type
    NOTE: Also support object targets
    """
    def setTargetRoom(self, targetRoomTp = 'kitchen', _setEagleMap = False):
        targetRoomTp = targetRoomTp.lower()
        if targetRoomTp not in self.all_desired_targetTypes:
            assert False, '[House] target type <{}> not supported in the current house!'.format(targetRoomTp)
        if targetRoomTp == self.targetRoomTp:
            return False  # room not changed!
        ###############################
        # Caching
        if self._setCurrentDistMap(targetRoomTp):
            self.targetRoomTp = targetRoomTp
            return True

        ################################
        # get destination range boxes
        flag_room_target = targetRoomTp in ALLOWED_TARGET_ROOM_TYPES

        if flag_room_target:
            if targetRoomTp == 'outdoor':
                targetRooms = \
                    [(room['bbox']['min'][0], room['bbox']['min'][2], room['bbox']['max'][0], room['bbox']['max'][2]) for room in self.all_rooms]
            else:
                targetRooms = \
                    [(room['bbox']['min'][0], room['bbox']['min'][2], room['bbox']['max'][0], room['bbox']['max'][2])
                     for room in self.all_rooms if any([ _equal_room_tp(tp, targetRoomTp) for tp in room['roomTypes']])]
        else:
            def _get_valid_expansion(x1,y1,x2,y2,rg):
                cx,cy = (x1+x2) * 0.5, (y1+y2) * 0.5
                covered_rooms = \
                    [(room['bbox']['min'][0], room['bbox']['min'][2], room['bbox']['max'][0], room['bbox']['max'][2])
                        for room in self.all_rooms if (room['bbox']['min'][0] < cx) and (room['bbox']['min'][2] < y1) \
                                                      and (room['bbox']['max'][0] > cx) and (room['bbox']['max'][2] > cy)]
                x_lo,y_lo,x_hi,y_hi = self.L_lo,self.L_lo,self.L_hi,self.L_hi
                for rx1,ry1,rx2,ry2 in covered_rooms:
                    x_lo = max(x_lo, rx1)
                    y_lo = max(y_lo, ry1)
                    x_hi = min(x_hi, rx2)
                    y_hi = min(y_hi, ry2)
                x_lo = max(x_lo, x1-rg)
                y_lo = max(y_lo, y1-rg)
                x_hi = min(x_hi, x2+rg)
                y_hi = min(y_hi, y2+rg)
                return (x_lo,y_lo,x_hi,y_hi)
            targetRooms = \
                [_get_valid_expansion(x1,y1,x2,y2,self.objTargetRange)
                 for x1,y1,x2,y2 in self.tar_obj_region[targetRoomTp]]
        assert (len(targetRooms) > 0), '[House] no target type <{}> in the current house!'.format(targetRoomTp)

        ###############################
        # generate destination mask map
        print('[House] Caching New ConnMap for Target <{}>! (total {} rooms involved)'.format(targetRoomTp, len(targetRooms)))
        if _setEagleMap and (self.eagle_n_row > 0):  # TODO: Currently a hack to speedup mult-target learning!!! So eagleMap become *WRONG*!
            inside_val, outside_val = (1, 0) if targetRoomTp != 'outdoor' else (0, 1)
            self.eagleMap[1, ...] = outside_val
            for room in targetRooms:
                _x1, _y1, _x2, _y2 = room
                x1,y1,x2,y2 = self.rescale(_x1,_y1,_x2,_y2,self.eagleMap.shape[1]-1)
                self.eagleMap[1, x1:(x2+1), y1:(y2+1)] = inside_val
        # compute shortest distance
        if targetRoomTp == 'ourdoor':
            okay_flag = self._genOutsideDistMap(targetRooms, targetRoomTp)
        else:
            okay_flag = self._genShortestDistMap(targetRooms, targetRoomTp)
        if not okay_flag:
            print("Error Occured for Target {}! Target Removed from Target List!".format(targetRoomTp))
            self.all_desired_targetTypes.remove(targetRoomTp)  # invalid target remove from list
            return False
        # shortest distance map computed successfully
        self._setCurrentDistMap(targetRoomTp)
        self.targetRoomTp = targetRoomTp
        del targetRooms
        print(' >>>> ConnMap Cached for targetRoomTp<{}>!'.format(targetRoomTp))
        return True

    def _getRoomBounds(self, room):
        _x1, _, _y1 = room['bbox']['min']
        _x2, _, _y2 = room['bbox']['max']
        return self.rescale(_x1, _y1, _x2, _y2)

    def _getRoomCoorBox(self, room):
        _x1, _, _y1 = room['bbox']['min']
        _x2, _, _y2 = room['bbox']['max']
        return _x1, _y1, _x2, _y2

    """
    returns a random location of a given room type
    """
    def getRandomLocation(self, roomTp, return_grid=False):
        roomTp = roomTp.lower()
        assert roomTp in self.all_desired_roomTypes, '[House] room type <{}> not supported!'.format(roomTp)
        if self._getConnectCoorsSize(roomTp) == 0:
            rooms = self._getRooms(roomTp)
            room_boxes = [self._getRoomCoorBox(room) for room in rooms]
            self._genShortestDistMap(room_boxes, roomTp)

        sz = self._getConnectCoorsSize_Bounded(roomTp, 0)  # only consider those locations inside the room
        if sz == 0: return None
        gx, gy = self._getIndexedConnectCoor(roomTp, np.random.randint(sz))
        if return_grid: return gx, gy
        return self.to_coor(gx, gy, True)

    def getRandomLocationForRoom(self, room_node, return_grid=False):
        reg_tag = room_node['id']
        x1, y1, x2, y2 = self._getRoomBounds(room_node)
        self._genValidCoors(x1, y1, x2, y2, reg_tag)
        sz = self._fetchValidCoorsSize(reg_tag)
        if sz == 0: return None
        gx, gy = self._getCachedIndexedValidCoor(np.random.randint(sz))
        if return_grid: return gx, gy
        return self.to_coor(gx, gy, True)

    """
    returns a random location for the current targetTp
      --> when max_allowed_dist is not None, only return location no further than the distance from the target
    """
    def getRandomConnectedLocation(self, return_grid=False, max_allowed_dist=None):
        sz = self._getCurrConnectCoorsSize() if max_allowed_dist is None else self._getCurrConnectCoorsSize_Bounded(max_allowed_dist)
        if sz == 0: return None
        gx, gy = self._getCurrIndexedConnectCoor(np.random.randint(sz))
        if return_grid: return gx, gy
        return self.to_coor(gx, gy, True)

    """
    return the total number of connected locations for the current targetTp
      --> when max_allowed_dist is not None, return the number of grids no further than the distance
    """
    def getConnectedLocationSize(self, max_allowed_dist=None):
        return self._getCurrConnectCoorsSize() if max_allowed_dist is None else self._getCurrConnectCoorsSize_Bounded(max_allowed_dist)

    """
    return the indexed entry in the current connected locations
    """
    def getIndexedConnectedLocation(self, idx, return_grid=False):
        sz = self._getCurrConnectCoorsSize()
        if (idx < 0) or (idx >= sz): return None  # out of range
        gx, gy = self._getCurrIndexedConnectCoor(idx)
        if return_grid: return gx, gy
        return self.to_coor(gx, gy, True)

    """
    return the number of allowed grids in this house for a particular 3D world step length
    """
    def getAllowedGridDist(self, max_allowed_step_dist):
        return int(np.floor(max_allowed_step_dist / self.grid_det + 1e-10))

    """
    cache the shortest distance to all the possible room types
    """
    def cache_all_target(self):
        avail_types = list(self.all_desired_targetTypes)
        for t in avail_types:
            self.setTargetRoom(t)
        self.init_graph()
        self.setTargetRoom(self.default_roomTp)

    """
    get the objects available for target
    """
    def genTargetObjectList(self, MapTargetCatFile):
        with open(MapTargetCatFile, 'r') as f:
            self.id_to_tar = json.load(f)
        target_obj = [(obj, self.id_to_tar[obj['modelId']]) for obj in self.all_obj if
                      (obj['bbox']['min'][1] < self.robotHei) and (obj['bbox']['max'][1] > self.carpetHei)
                      and (obj['modelId'] in self.id_to_tar)]
        for obj, cat in target_obj:
            if cat not in ALLOWED_OBJECT_TARGET_TYPES: continue
            _x1, _, _y1 = obj['bbox']['min']
            _x2, _, _y2 = obj['bbox']['max']
            list_of_cat = _get_object_categories(cat)
            for c in list_of_cat:
                if c not in self.tar_obj_region:
                    self.tar_obj_region[c] = []
                self.tar_obj_region[c].append((_x1, _y1, _x2, _y2))

    # TODO: maybe move this to C++ side?
    def genObstacleMap(self, MetaDataFile, gen_debug_map=True, dest=None, n_row=None):
        # load all the doors
        target_match_class = 'nyuv2_40class'
        target_door_labels = ['door', 'fence', 'arch']
        door_ids = set()
        fine_grained_class = 'fine_grained_class'
        ignored_labels = ['person', 'umbrella', 'curtain', 'basketball_hoop']
        person_ids = set()
        window_ids = set()
        with open(MetaDataFile) as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                if row[target_match_class] in target_door_labels:
                    door_ids.add(row['model_id'])
                if row[target_match_class] == 'window':
                    window_ids.add(row['model_id'])
                if row[fine_grained_class] in ignored_labels:
                    person_ids.add(row['model_id'])
        def is_door(obj):
            if obj['modelId'] in door_ids:
                return True
            if (obj['modelId'] in window_ids) and (obj['bbox']['min'][1] < self.carpetHei):
                return True
            return False

        solid_obj = [obj for obj in self.all_obj if (not is_door(obj)) and (obj['modelId'] not in person_ids)]  # ignore person
        door_obj = [obj for obj in self.all_obj if is_door(obj)]
        colide_obj = [obj for obj in solid_obj if obj['bbox']['min'][1] < self.robotHei and obj['bbox']['max'][1] > self.carpetHei]
        # generate the map for all the obstacles
        obsMap = dest if dest is not None else self.obsMap
        if n_row is None:
            n_row = obsMap.shape[0] - 1
        x1,y1,x2,y2 = self.rescale(self.L_min_coor[0],self.L_min_coor[2],self.L_max_coor[0],self.L_max_coor[2],n_row)  # fill the space of the level
        fill_region(obsMap,x1,y1,x2,y2,0)
        if gen_debug_map and (self._debugMap is not None):
            fill_region(self._debugMap, x1, y1, x2, y2, 0)
        # fill boundary of rooms
        maskRoom = np.zeros_like(obsMap,dtype=np.int8)
        for wall in self.all_walls:
            _x1, _, _y1 = wall['bbox']['min']
            _x2, _, _y2 = wall['bbox']['max']
            x1,y1,x2,y2 = self.rescale(_x1,_y1,_x2,_y2,n_row)
            fill_region(obsMap, x1, y1, x2, y2, 1)
            if gen_debug_map and (self._debugMap is not None):
                fill_region(self._debugMap, x1, y1, x2, y2, 1)
            fill_region(maskRoom, x1, y1, x2, y2, 1)
        # remove all the doors
        for obj in door_obj:
            _x1, _, _y1 = obj['bbox']['min']
            _x2, _, _y2 = obj['bbox']['max']
            x1,y1,x2,y2 = self.rescale(_x1,_y1,_x2,_y2,n_row)
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            # expand region
            if x2 - x1 < y2 - y1:
                while (x1 - 1 >= 0) and (maskRoom[x1-1,cy] > 0):
                    x1 -= 1
                while (x2 + 1 < maskRoom.shape[0]) and (maskRoom[x2+1,cy] > 0):
                    x2 += 1
            else:
                while (y1 - 1 >= 0) and (maskRoom[cx,y1-1] > 0):
                    y1 -= 1
                while (y2+1 < maskRoom.shape[1]) and (maskRoom[cx,y2+1] > 0):
                    y2 += 1
            fill_region(obsMap,x1,y1,x2,y2,0)
            if gen_debug_map and (self._debugMap is not None):
                fill_region(self._debugMap, x1, y1, x2, y2, 0.5)

        # mark all the objects obstacle
        for obj in colide_obj:
            _x1, _, _y1 = obj['bbox']['min']
            _x2, _, _y2 = obj['bbox']['max']
            x1,y1,x2,y2 = self.rescale(_x1,_y1,_x2,_y2,n_row)
            fill_region(obsMap,x1,y1,x2,y2,1)
            if gen_debug_map and (self._debugMap is not None):
                fill_region(self._debugMap, x1, y1, x2, y2, 0.8)


    def genMovableMap(self):
        roi_bounds = self._getRegionsOfInterest()
        self._genMovableMap(roi_bounds)

    def _getRegionsOfInterest(self):
        """Override this function for customizing the areas of the map to
        consider when marking valid movable locations
        Returns a list of (x1, y1, x2, y2) tuples (boundary inclusive) representing bounding boxes
        of valid areas.  Coordinates are normalized grid coordinates.
        """
        return [(0, 0, self.n_row, self.n_row)]


    """
    check whether the *grid* coordinate (x,y) is inside the house
    """
    def inside(self,x,y):
        return self._inside(x, y)

    """
    get the corresponding grid coordinate of (x, y) in the topdown 2d map
    """
    def get_eagle_view_grid(self, x, y, input_grid=False):
        if input_grid:
            x, y = self.to_coor(x, y, shft=True)
        return self.to_grid(x, y, n_row=self.eagle_n_row-1)

    """
    convert the continuous rectangle region in the SUNCG dataset to the grid region in the house
    """
    def rescale(self,x1,y1,x2,y2,n_row=None):
        if n_row is None: n_row = self.n_row
        return self._rescale(x1, y1, x2, y2, n_row)

    def to_grid(self, x, y, n_row=None):
        """
        Convert the true-scale coordinate in SUNCG dataset to grid location
        """
        if n_row is None: n_row = self.n_row
        return self._to_grid(x, y, n_row)

    def to_coor(self, x, y, shft=False):
        """
        Convert grid location to SUNCG dataset continuous coordinate (the grid center will be returned when shft is True)
        """
        return self._to_coor(x, y, shft)

    """
    suppose the robot stands at continuous coordinate (cx, cy), check whether it will touch any obstacles
    """
    def check_occupy(self, cx, cy):  # cx, cy are real coordinates
        return self._check_occupy(cx, cy)

    """
    check if an agent can reach grid location (gx, gy)
    """
    def canMove(self, gx, gy):
        return self._canMove(gx, gy)

    """
    check if grid location (gx, gy) is connected to the target room
    """
    def isConnect(self, gx, gy):
        return self._isConnect(gx, gy)

    """
    get the raw shortest distance from grid location (gx, gy) to the target room
    """
    def getDist(self, gx, gy):
        return self._getDist(gx, gy)

    """
    return a scaled shortest distance, which ranges from 0 to 1
    """
    def getScaledDist(self, gx, gy):
        return self._getScaledDist(gx, gy)


    """
    returns all rooms of a given type
    """
    def _getRooms(self, roomTp):
        rooms = [
            r for r in self.all_rooms
            if any([_equal_room_tp(tp, roomTp) for tp in r['roomTypes']])
        ]
        return rooms


    """
    return whether or not a given room type exists in the house
    """
    def hasRoomType(self, roomTp):
        return len(self._getRooms(roomTp)) > 0


    """
    return whether the robot can move from pA to pB (real coordinate) via moveMap
    """
    def collision_check_fast(self, pA, pB, num_samples):
        return self._fast_collision_check(pA[0], pA[1], pB[0], pB[1], num_samples)

    """
    return whether the robot can move from pA to pB (real coordinate) via check_occupy()
    """
    def collision_check_slow(self, pA, pB, num_samples):
        return self._full_collision_check(pA[0], pA[1], pB[0], pB[1], num_samples)

    #######################
    # GRAPH functionality #
    #######################
    def init_graph(self):
        ret = self._gen_target_graph(len(self.all_desired_targetTypes) - len(self.all_desired_roomTypes))
        if ret <= 0:
            print('[House] ERROR when building object graph!')
        print("Connectivity Graph Built!")

    def get_target_mask(self, cx, cy):
        tags = self._get_target_mask_names(cx, cy, False)
        return tags

    def get_optimal_plan(self, cx, cy, target):
        assert target in self.all_desired_targetTypes
        plan = self._compute_target_plan(cx, cy, target)
        assert(len(plan) > 0)
        dist = self._get_target_plan_dist(cx, cy, plan)
        return [(p, d) for p, d in zip(plan, dist)]


    #######################
    # DEBUG functionality #
    #######################
    def _showDebugMap(self, filename=None):
        if self._debugMap is None:
            print('[Warning] <showDebugMap>: Please set DebugInfoOn=True before calling this method!')
        else:
            import matplotlib.pyplot as plt
            import seaborn as sns
            ax = sns.heatmap(self._debugMap[:,::-1])
            if filename is None:
                plt.show()
            else:
                ax.get_figure().savefig(filename)

    def _showObsMap(self):
        import matplotlib.pyplot as plt
        import seaborn as sns
        plt.clf()
        sns.heatmap(self.obsMap[:,::-1])
        plt.show()

    def _showMoveMap(self, visualize=True):
        import matplotlib.pyplot as plt
        import seaborn as sns
        proj = np.array(self.obsMap, dtype=np.float32)
        for x in range(self.n_row+1):
            for y in range(self.n_row+1):
                if self.canMove(x, y):
                    proj[x,y] = 0.5
        if visualize:
            plt.clf()
            ax = sns.heatmap(proj[:,::-1])
            if visualize:
                plt.show()
        return proj

    def _showConnMap(self):
        import matplotlib.pyplot as plt
        import seaborn as sns
        proj = self._showMoveMap(False)
        for x in range(self.n_row+1):
            for y in range(self.n_row+1):
                if self.isConnect(x,y):
                    proj[x,y] = 0.25
        plt.clf()
        sns.heatmap(proj[:,::-1])
        plt.show()
        return proj
