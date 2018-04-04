import sys, os, platform
import numpy as np
import random
import time

from house import House
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

flag_parallel_init = False
flag_env_set = 'train' #'small'   # 'train'

flag_object_success_range = 0.5

house_ids_dict = json.load(open('all_house_ids.json','r'))
all_houseIDs = house_ids_dict[flag_env_set]

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
                      MapTargetCatFile=modelObjectMapFile,
                      ObjectTargetSuccRange=flag_object_success_range,
                      IncludeOutdoorTarget=True)
    else:
        house = House(jsonFile, objFile, csvFile,
                      CachedFile=cachedFile, GenRoomTypeMap=genRoomTypeMap,
                      MapTargetCatFile=modelObjectMapFile,
                      ObjectTargetSuccRange=flag_object_success_range,
                      IncludeOutdoorTarget=True)
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
        if not flag_parallel_init:
            return [create_house(houseID, genRoomTypeMap, cacheAllTarget) for houseID in all_houseIDs[:k]]
        from multiprocessing import Pool
        _args = [(all_houseIDs[j], genRoomTypeMap, cacheAllTarget) for j in range(k)]
        with Pool(k) as pool:
            ret_worlds = pool.starmap(create_house, _args)  # parallel version for initialization
        print('  >> Done! Time Elapsed = %.4f(s)' % (time.time() - ts))
        return ret_worlds


if __name__ == '__main__':
    print('Start Generating House ....')
    ts = time.time()
    num_houses = len(all_houseIDs)#50
    #all_houses = create_house_from_index(-num_houses, cacheAllTarget=True)
    #all_houses = all_houseIDs[:num_houses]
    all_houses = create_house_from_index(-num_houses, cacheAllTarget=True)
    import objrender
    api = objrender.RenderAPI(w=400, h=300, device=0)
    env = MultiHouseEnv(api, all_houses, config=CFG, parallel_init=flag_parallel_init)
    #env = Environment(api, all_houses[0], config=CFG)
    task = RoomNavTask(env, hardness=0.6, reward_type='new', max_birthplace_steps=30, discrete_action=True, include_object_target=True)
    dur = time.time() - ts
    print('  --> Time Elapsed = %.6f (s)' % dur)

    action_dict = dict(k=7,l=9,j=10,o=3,u=4,f=5,a=6,i=8,d=11,s=12,r=-1,h=-2,q=-3)
    def print_help():
        print('Usage: ')
        print('> Actions: (Simplified Version) Total 10 Actions')
        print('  --> j, k, l, i, u, o: left, back, right, forward, left-forward, right-forward')
        print('  --> a, s, d, f: left-rotate, small left-rot, small right-rot, right-rotate')
        print('> press r: reset')
        print('> press h: show helper again')
        print('> press m: show <HouseID>')
        print('> press q: exit')

    import cv2
    print_help()
    eval = []
    itr = 0
    while True:
        step = 0
        rew = 0
        good = 0
        task.reset(target='any-object', house_id = itr)
        itr += 1
        target = task.info['target_room']
        while True:
            print('Step#%d, Instruction = <go to %s>' % (step, target))
            if step == 0:
                _i = env.house._id
                print('>>> HouseID = {}, set = {}, index = {}'.format(all_houseIDs[_i], flag_env_set, _i))
            mat = task.debug_show()
            cv2.imshow("aaa", mat)
            while True:
                key = cv2.waitKey(0)
                key = chr(key)
                if key == 'm':
                    _i = env.house._id
                    print('>>> HouseID = {}, set = {}, index = {}'.format(all_houseIDs[_i], flag_env_set, _i))
                    continue
                if key in action_dict:
                    if key == 'h':
                        print_help()
                    else:
                        break
                else:
                    print('>> invalid key! press q to quit; r to reset; h to get helper')
            if action_dict[key] < 0:
                break
            step += 1
            obs, reward, done, info = task.step(action_dict[key])
            rew += reward
            print('>> r = %.2f, done = %f, accu_rew = %.2f, step = %d' % (reward, done, rew, step))
            print('   info: collision = %d, raw_dist = %d, scaled_dist = %.3f, opt_steps = %d' % (info['collision'], info['dist'], info['scaled_dist'], info['optsteps']))

            #############
            # Plan Info
            #############
            print('   S_aux = {}'.format(task.get_aux_tags()))
            print('   plan info: {}'.format(task.get_optimal_plan()))

            if done:
                good = 1
                print('Congratulations! You reach the Target!')
                print('>> Press any key to restart!')
                key = cv2.waitKey(0)
                break
        eval.append((step, rew, good))
        if key == 'q':
            break
    if len(eval) > 0:
        print('++++++++++ Task Stats +++++++++++')
        print("Episode Played: %d" % len(eval))
        succ = [e for e in eval if e[2] > 0]
        print("Success = %d, Rate = %.3f" % (len(succ), len(succ) / len(eval)))
        print("Avg Reward = %.3f" % (sum([e[1] for e in eval])/len(eval)))
        if len(succ) > 0:
            print("Avg Success Reward = %.3f" % (sum([e[1] for e in succ]) / len(succ)))
        print("Avg Step = %.3f" % (sum([e[0] for e in eval]) / len(eval)))
        if len(succ) > 0:
            print("Avg Success Step = %.3f" % (sum([e[0] for e in succ]) / len(succ)))
