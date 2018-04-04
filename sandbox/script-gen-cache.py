import sys, os, platform
import numpy as np
import random
import time

import House3D
from House3D import House

import json
CFG = json.load(open('config.json','r'))
prefix = CFG['prefix']
csvFile = CFG['modelCategoryFile']
colorFile = CFG['colorFile']
roomTargetFile = CFG['roomTargetFile']
objectTargetFile = CFG['objectTargetFile'] if 'objectTargetFile' in CFG else None
modelObjectMapFile = CFG['modelObjectMap'] if 'modelObjectMap' in CFG else None

flag_parallel_init = (sys.platform != 'darwin')
flag_env_set = 'train' #'small'   # 'train'

flag_object_success_range = 0.5

house_ids_dict = json.load(open('all_house_ids.json','r'))
all_houseIDs = house_ids_dict[flag_env_set]

def cache_house(houseID, config):
    print('Loading house {}'.format(houseID))
    objFile = os.path.join(config['prefix'], houseID, 'house.obj')
    jsonFile = os.path.join(config['prefix'], houseID, 'house.json')
    assert (os.path.isfile(objFile) and os.path.isfile(jsonFile)), '[Environment] house objects not found! objFile=<{}>'.format(objFile)
    cachefile = os.path.join(config['prefix'], houseID, 'cachedmap1k.pkl')
    house = House(jsonFile, objFile, csvFile, StorageFile=cachefile)
    return house


if __name__ == '__main__':
    from multiprocessing import Pool
    k = len(all_houseIDs)
    ts = time.time()
    print('Caching Total <{}> Houses in <{}> Set...'.format())
    _args = [(all_houseIDs[j], CFG) for j in range(k)]
    max_pool_size = min(40, k)
    with Pool(k) as pool:
        ret_worlds = pool.starmap(cache_house, _args)  # parallel version for initialization
    dur = time.time() - ts
    print('>> Done! Elapsed Time = %.3fs' % (dur))
