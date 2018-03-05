import sys, os, platform
import numpy as np
import random
import time

from house import House
from core import Environment, MultiHouseEnv

import json
CFG = json.load(open('config.json','r'))
prefix = CFG['prefix']
csvFile = CFG['modelCategoryFile']
colorFile = CFG['colorFile']
roomTargetFile = CFG['roomTargetFile']
objectTargetFile = CFG['objectTargetFile'] if 'objectTargetFile' in CFG else None
modelObjectMapFile = CFG['modelObjectMap'] if 'modelObjectMap' in CFG else None

all_houseIDs = ['00065ecbdd7300d35ef4328ffe871505',
    'cf57359cd8603c3d9149445fb4040d90', '31966fdc9f9c87862989fae8ae906295', 'ff32675f2527275171555259b4a1b3c3',
    '7995c2a93311717a3a9c48d789563590', '8b8c1994f3286bfc444a7527ffacde86', '775941abe94306edc1b5820e3a992d75',
    '32e53679b33adfcc5a5660b8c758cc96', '4383029c98c14177640267bd34ad2f3c', '0884337c703e7c25949d3a237101f060',
    '492c5839f8a534a673c92912aedc7b63', 'a7e248efcdb6040c92ac0cdc3b2351a6', '2364b7dcc432c6d6dcc59dba617b5f4b',
    'e3ae3f7b32cf99b29d3c8681ec3be321', 'f10ce4008da194626f38f937fb9c1a03', 'e6f24af5f87558d31db17b86fe269cf2',
    '1dba3a1039c6ec1a3c141a1cb0ad0757', 'b814705bc93d428507a516b866efda28', '26e33980e4b4345587d6278460746ec4',
    '5f3f959c7b3e6f091898caa8e828f110', 'b5bd72478fce2a2dbd1beb1baca48abd', '9be4c7bee6c0ba81936ab0e757ab3d61']

def genCacheFile(houseID):
    return prefix + houseID + '/cachedmap1k.pkl'

def create_house(houseID, genRoomTypeMap=False, cacheAllTarget=False):
    objFile = prefix + houseID + '/house.obj'
    jsonFile = prefix + houseID + '/house.json'
    cachedFile = genCacheFile(houseID)
    if not os.path.isfile(cachedFile):
        print('Generating Cached Map File for House <{}>!'.format(houseID))
        house = House(jsonFile, objFile, csvFile,
                      StorageFile=cachedFile, GenRoomTypeMap=genRoomTypeMap,
                      MapTargetCatFile=modelObjectMapFile)
    else:
        house = House(jsonFile, objFile, csvFile,
                      CachedFile=cachedFile, GenRoomTypeMap=genRoomTypeMap,
                      MapTargetCatFile=modelObjectMapFile)
    #house = House(jsonFile, objFile, csvFile,
    #              ColideRes=colide_res,
    #              CachedFile=cachedFile, EagleViewRes=default_eagle_resolution,
    #              GenRoomTypeMap=genRoomTypeMap)
    if cacheAllTarget:
        house.cache_all_target()
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
        from multiprocessing import Pool
        _args = [(all_houseIDs[j], genRoomTypeMap, cacheAllTarget) for j in range(k)]
        with Pool(k) as pool:
            ret_worlds = pool.starmap(create_house, _args)  # parallel version for initialization
        print('  >> Done! Time Elapsed = %.4f(s)' % (time.time() - ts))
        return ret_worlds
        # return [create_world(houseID, genRoomTypeMap) for houseID in all_houseIDs[:k]]

if __name__ == '__main__':
    import time
    #import objrender
    print('Start Generating House ....')
    ts = time.time()
    all_houses = create_house_from_index(-20, cacheAllTarget=True)
    #api = objrender.RenderAPI(w=resolution[0], h=resolution[1], device=render_device)
    #env = MultiHouseEnv(api, all_houses, config=CFG)
    dur = time.time() - ts
    print('  --> Time Elapsed = %.6f (s)' % dur)
