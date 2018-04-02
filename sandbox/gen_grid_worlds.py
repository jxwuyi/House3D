import sys, os, platform
import numpy as np
import random
import time

from meta_house import MetaHouse
from core import Environment, MultiHouseEnv
from roomnav import RoomNavTask

import json
CFG = json.load(open('config.json','r'))
prefix = CFG['prefix']
csvFile = CFG['modelCategoryFile']
colorFile = CFG['colorFile']
roomTargetFile = CFG['roomTargetFile']
objectTargetFile = CFG['objectTargetFile'] if 'objectTargetFile' in CFG else None
modelObjectMapFile = CFG['modelObjectMap'] if 'modelObjectMap' in CFG else None

tmp_storage_dir = './storage/'
eagle_view_resolution = 39

flag_parallel_init = (sys.platform != 'darwin')
flag_env_set = 'train'

flag_object_success_range = 0.3

house_ids_dict = json.load(open('all_house_ids.json','r'))
all_houseIDs = house_ids_dict[flag_env_set]

def genCacheFile(houseID):
    return prefix + houseID + '/cachedmap1k.pkl'

def create_house(houseID, genRoomTypeMap=False, cacheAllTarget=False):
    objFile = prefix + houseID + '/house.obj'
    jsonFile = prefix + houseID + '/house.json'
    cachedFile = genCacheFile(houseID)
    storageFile = tmp_storage_dir + houseID + '_data.pkl'
    if not os.path.isfile(cachedFile):
        print('Generating Cached Map File for House <{}>!'.format(houseID))
        house = MetaHouse(jsonFile, objFile, csvFile,
                          EagleViewRes=eagle_view_resolution, DebugInfoOn=True, StorageFile=storageFile)
    else:
        house = MetaHouse(jsonFile, objFile, csvFile,
                          EagleViewRes=eagle_view_resolution, DebugInfoOn=True, StorageFile=storageFile)
    #house = House(jsonFile, objFile, csvFile,
    #              ColideRes=colide_res,
    #              CachedFile=cachedFile, EagleViewRes=default_eagle_resolution,
    #              GenRoomTypeMap=genRoomTypeMap)
    #if cacheAllTarget:
    #    house.cache_all_target()
    return house

def create_house_from_index(k, genRoomTypeMap=False, cacheAllTarget=False):
    if k >= 0:
        if k >= len(all_houseIDs):
            print('k={} exceeds total number of houses ({})! Randomly Choose One!'.format(k, len(all_houseIDs)))
            houseID = random.choice(all_houseIDs)
        else:
            houseID = all_houseIDs[k]
        return create_house(houseID, genRoomTypeMap, cacheAllTarget)
    else:
        k = -k
        print('Multi-House Environment! Total Selected Houses = {}'.format(k))
        if k > len(all_houseIDs):
            print('  >> k={} exceeds total number of houses ({})! use all houses!')
            k = len(all_houseIDs)
        import time
        ts = time.time()
        print('Caching All Worlds ...')
        # use the first k houses
        if not flag_parallel_init:
            ret_worlds = [create_house(houseID, genRoomTypeMap, cacheAllTarget) for houseID in all_houseIDs[:k]]
        else:
            from multiprocessing import Pool
            _args = [(all_houseIDs[j], genRoomTypeMap, cacheAllTarget) for j in range(k)]
            with Pool(k) as pool:
                ret_worlds = pool.starmap(create_house, _args)  # parallel version for initialization
        print('  >> Done! Time Elapsed = %.4f(s)' % (time.time() - ts))
        return ret_worlds

print('Start Generating House ....')
ts = time.time()
num_houses = len(all_houseIDs)
all_houses = create_house_from_index(-num_houses, cacheAllTarget=True)

room_stats = dict()
obj_stats = dict()
for house in all_houses:
    for k in house.room_stats.keys():
        if k not in room_stats:
            room_stats[k] = 1
        else:
            room_stats[k] += 1
    for k in house.obj_stats.keys():
        if k not in obj_stats:
            obj_stats[k] = 1
        else:
            obj_stats[k] += 1


import operator
sorted_rooms = sorted(room_stats.items(), key=operator.itemgetter(1))
sorted_objs = sorted(obj_stats.items(), key=operator.itemgetter(1))

with open('room_stats.json','w') as f:
    json.dump(room_stats, f)
with open('obj_stats.json','w') as f:
    json.dump(obj_stats, f)

logfile = 'log_stats.txt'
with open(logfile, 'w') as f:
    print('++++++++++++++++++++++++++++++++++\nroom stats:', file=f)
    for r, c in sorted_rooms:
        print('room = %s, cnt = %d, frac = %.4f' % (r, c, c / len(all_houses)), file=f)
    print('++++++++++++++++++++++++++++++++++\nobject stats:', file=f)
    for r, c in sorted_objs:
        print('object = %s, cnt = %d, frac = %.4f' % (r, c, c / len(all_houses)), file=f)


"""
    import objrender
    api = objrender.RenderAPI(w=400, h=300, device=0)
    env = MultiHouseEnv(api, all_houses, config=CFG, parallel_init=flag_parallel_init)
    #env = Environment(api, all_houses[0], config=CFG)
    task = RoomNavTask(env, hardness=0.6, discrete_action=True, include_object_target=True)
    dur = time.time() - ts
    print('  --> Time Elapsed = %.6f (s)' % dur)
"""
