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
from .house import House
from .core import Environment, MultiHouseEnv
from .roomnav import RoomNavTask, reset_see_criteria

objnav_pixel_see_reward = 0.2
objnav_success_stay_steps = 3
objnav_pixel_for_object_see_rate = 0.20

__all__ = ['ObjNavTask']

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
                 false_rate=0):
        assert include_object_target, 'Object Target Must Be Turned ON!'

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
                                         reward_silence=reward_silence)
        self.false_rate = false_rate
        self.inroomRew = 0.0
        self.goodMoveRew = 0.0
        self.pixelRew = objnav_pixel_see_reward if reward_type != 'none' else 0.0
        self.succSeeSteps = objnav_success_stay_steps

        self.prev_grid = None
        self._is_valid_case = True

    def reset(self, target=None):
        # TODO: change the following
        # clear episode steps
        self.current_episode_step = 0
        self.success_stay_cnt = 0
        self._object_cnt = 0

        # reset house
        self.env.reset_house()
        self.house.targetRoomTp = None  # [NOTE] IMPORTANT! clear this!!!!!

        # reset target room
        self.reset_target(target=target)  # randomly reset
        self.collision_flag = False

        # general birth place
        x, y = self.house.getIndexedConnectedLocation(np.random.randint(self.availCoorsSize))

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
            return super(ObjNavTask, self)._is_success(raw_dist, grid)
        else:
            # TODO:
            return False

    def step(self, action):
        obs, reward, done, cur_info = super(ObjNavTask, self).step(action)
        # TODO: add leave penalty
        cur_grid = cur_info['grid']
        self.prev_grid = cur_info['grid']
        return obs, reward, done, cur_info
