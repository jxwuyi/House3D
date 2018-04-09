# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import time
import sys, os
import numpy as np
import random
import csv
import copy
import cv2
import pickle

import gym
from gym import spaces
from .house import House, _equal_room_tp, ALLOWED_OBJECT_TARGET_INDEX, ALLOWED_OBJECT_TARGET_TYPES, ALLOWED_TARGET_ROOM_TYPES
from .core import Environment, MultiHouseEnv
from .roomnav import RoomNavTask, reset_see_criteria

objnav_pixel_see_reward = 1.0
objnav_success_stay_steps = 3
objnav_pixel_for_object_see_rate = 0.20
objnav_leave_penalty = 1.0
objnav_time_penalty = 0.05

__all__ = ['ObjNavTask']


def _create_global_obj_list(obj_msk, obj_target_list):
    g_list = []
    for i, obj in enumerate(obj_target_list):
        if (obj_msk & (1 << i)) > 0:   # contains obj
            g_list.append(obj)
    return g_list

class ObjNavTask(RoomNavTask):
    def __init__(self, env,
                 seed=None,
                 reward_type='none',
                 hardness=None,
                 max_birthplace_steps=None,
                 move_sensitivity=None,
                 segment_input=True,
                 joint_visual_signal=False,
                 depth_signal=True,
                 max_steps=-1,
                 success_measure='see',
                 discrete_action=False,
                 include_object_target=True,
                 reward_silence=0,
                 birthplace_curriculum_schedule=None,
                 false_rate=0):
        assert include_object_target, 'Object Target Must Be Turned ON!'
        assert birthplace_curriculum_schedule is None, 'currently do not support curriculum'

        reset_see_criteria(env.resolution, up_rate=objnav_pixel_for_object_see_rate)

        super(ObjNavTask, self).__init__(env, seed, reward_type='none',
                                         hardness=hardness, max_birthplace_steps=max_birthplace_steps,
                                         move_sensitivity=move_sensitivity,
                                         segment_input=segment_input,
                                         joint_visual_signal=joint_visual_signal,
                                         depth_signal=depth_signal,
                                         max_steps=max_steps,
                                         success_measure=success_measure,
                                         discrete_action=discrete_action,
                                         include_object_target=True,
                                         reward_silence=reward_silence,
                                         birthplace_curriculum_schedule=None)
        self.false_rate = false_rate
        self.inroomRew = 0.0
        self.goodMoveRew = 0.0
        self.pixelRew = objnav_pixel_see_reward if reward_type != 'none' else 0.0
        self.succSeeSteps = objnav_success_stay_steps
        self.timePenalty = objnav_time_penalty

        self._prev_reach = False
        self._curr_reach = False
        self._is_valid_case = True

        self._curr_room_stats = None
        self._cached_rooms = dict()

    def _process_house(self, house):
        all_valid_room = []
        n_room = len(house.all_desired_roomTypes)
        n_obj = len(house.all_desired_targetObj)
        full_n_obj_mask = (1 << n_obj) - 1

        def _process_room_info(in_msk, out_msk, room):
            obj_msk = (in_msk >> n_room)
            out_obj_msk = (out_msk >> n_room)
            if obj_msk > 0:  # has some object targets
                global_invalid_obj_list = _create_global_obj_list((obj_msk | out_obj_msk) ^ full_n_obj_mask,
                                                                  house.all_desired_targetObj)
                global_valid_obj_list = _create_global_obj_list(obj_msk, house.all_desired_targetObj)
                curr_room_msk = 0
                room_types = [room] if isinstance(room, str) else room['roomTypes']
                for i, roomTp in enumerate(house.all_desired_roomTypes):
                    if any([_equal_room_tp(tp, roomTp) for tp in room_types]):
                        curr_room_msk |= 1 << i
                room_id = room if isinstance(room, str) else room['id']
                return room_id, curr_room_msk, global_valid_obj_list, global_invalid_obj_list
            else:
                return None

        for room in house.all_rooms:
            in_msk, out_msk = house.getRegionMaskForRoom(room)
            val = _process_room_info(in_msk, out_msk, room)
            if val is not None:
                all_valid_room.append(val)
        in_msk, out_msk = house.getRegionMaskForTarget('outdoor')
        val = _process_room_info(in_msk, out_msk, 'outdoor')
        if val is not None:
            all_valid_room.append(val)
        return all_valid_room

    def reset(self, target=None):
        # increase episode counter
        self.total_episode_cnt += 1

        # clear episode steps
        self.current_episode_step = 0
        self.success_stay_cnt = 0
        self._object_cnt = 0
        self._prev_object_see_rate = 0.0

        # specialized to ObjNav
        self._prev_reach = False
        self._is_valid_case = (np.random.rand() < self.false_rate)

        # reset house
        self.env.reset_house()
        self.house.targetRoomTp = None  # [NOTE] IMPORTANT! clear this!!!!!

        # cache house
        if not hasattr(self.house, '_id'):   # single house env
            self.house._id = 0
        if self.house._id not in self._cached_rooms:
            self._cached_rooms[self.house._id] = self._process_house(self.house)
        self._curr_room_stats = np.random.choice(self.cached_rooms[self.house._id])  # tag, room_mask, valid_obj, invalid_obj
        self._curr_room_mask = self._curr_room_stats[1]

        # reset target room
        if target is None:
            if self._is_valid_case:
                target = np.random.choice(self._curr_room_stats[2])
            else:
                target = np.random.choice(self._curr_room_stats[3])
        self.reset_target(target=target)  # randomly reset
        self.collision_flag = False

        # general birth place
        x, y = self.house.getRandomLocationForRoom(self._curr_room_stats[0], is_cached=True)

        # generate state
        self.env.reset(x=x, y=y)
        self.last_obs = self.env.render()
        if self.joint_visual_signal:
            self.last_obs = np.concatenate([self.env.render(mode='rgb'), self.last_obs], axis=-1)
        ret_obs = self.last_obs
        if self.depth_signal:
            dep_sig = self.env.render(mode='depth')
            if dep_sig.shape[-1] > 1:
                dep_sig = dep_sig[..., 0:1]
            ret_obs = np.concatenate([ret_obs, dep_sig], axis=-1)
        self.last_info = self.info
        return ret_obs

    def _is_success(self, raw_dist, grid):
        if self._is_valid_case:
            self._curr_reach = (raw_dist == 0)
            return super(ObjNavTask, self)._is_success(raw_dist, grid)
        else:
            curr_mask = self.house.get_target_mask_grid(grid[0], grid[1], False)
            if (curr_mask & self._curr_room_mask) == 0:  # get out!
                self._curr_reach = True
                self.success_stay_cnt += 1
                return self.success_stay_cnt >= self.succSeeSteps
            else:
                self._curr_reach = False
                self.success_stay_cnt = 0
                return False

    def step(self, action):
        obs, reward, done, cur_info = super(ObjNavTask, self).step(action)
        # add leave penalty
        if self._prev_reach and (not self._curr_reach):
            reward -= objnav_leave_penalty
        self._prev_reach = self._curr_reach
        return obs, reward, done, cur_info
