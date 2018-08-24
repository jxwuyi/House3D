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
import heapq

import gym
from gym import spaces
from .house import House
from .core import Environment, MultiHouseEnv

__all__ = ['RoomNavTask']

###############################################
# Task related definitions and configurations
###############################################
flag_print_debug_info = False  # flag for printing debug info

dist_reward_scale = 1.0  # 2.0
collision_penalty_reward = 0.3  # penalty for collision
correct_move_reward = None  # reward when moving closer to target
stay_room_reward = 0.1  # reward when staying in the correct target room
indicator_reward = 0.5
success_reward = 10   # reward when success

time_penalty_reward = 0.1   # penalty for each time step
delta_reward_coef = 0.5
speed_reward_coef = 1.0

success_stay_time_steps = 5
success_see_target_time_steps = 2   # time steps required for success under the "see" criteria

wrong_stop_penalty = -2

######################################
# reward function parameters for new
######################################
new_time_penalty_reward = 0.1   # penalty for each time step
new_reward_coef = 1.0
new_reward_bound = 0.5
new_leave_penalty = 0.1
new_stay_room_reward = 0.05
new_success_stay_time_steps = 3
new_success_reward = 10
new_pixel_object_reward = 1
new_pixel_reward_rate = 0.10
#####################################


# sensitivity setting
rotation_sensitivity = 30  # 45   # maximum rotation per time step
discrete_rotation_sensitivity = 15
default_move_sensitivity = 0.5  # 1.0   # maximum movement per time step

"""
# discrete action space actions, totally <13> actions
# Fwd, L, R, LF, RF, Lrot, Rrot, Bck, s-Fwd, s-L, s-R, s-Lrot, s-Rrot, Stay
# NOTE: at most 8 actions!!!!
# NOTE: allowed_actions_for_supervision[0] must be 0!!!!!!
allowed_actions_for_supervision = [0, 8, 3, 4, 5, 6, 11, 12]  # Fwd, LF, RF, Lrot, Rrot, s-Fwd, s-Lrot, s-Rrot
#allowed_actions_for_supervision = [0, 1, 2, 5, 6, 8, 11, 12]  # Fwd, L, R, Lrot, Rrot, s-Fwd, s-Lrot, s-Rrot
discrete_angle_delta_value = [0, 0, 0, 0, 0, 2, -2, 0, 0, 0, 0, 1, -1, 0]
discrete_actions=[(1.,0.,0.), (0.,1.,0.), (0.,-1.,0.), (0.4,0.4,0.), (0.4,-0.4,0.),
                  (0.,0.,1.), (0.,0.,-1.),
                  (-0.4,0.,0.),
                  (0.4,0.,0.), (0.,0.4,0.), (0.,-0.4,0.),
                  (0.,0.,0.5), (0.,0.,-0.5), (0., 0., 0.)]
discrete_action_names = ['Forward', 'Left', 'Right', 'Left-Fwd', 'Right-Fwd', 'Left-Rotate', 'Right-Rotate',
                         'Backward', 'Small-Forward', 'Small-Left', 'Small-Right', 'Small-Left-Rot', 'Small-Right-Rot', 'Stay']
"""

# discrete action space actions, totally <9> actions
# Fwd, LF, RF, Lrot, Rrot, s-Fwd, s-Lrot, s-Rrot, Stay
# NOTE: allowed_actions_for_supervision[0] must be 0 (Fwd) !!!!!!
allowed_actions_for_supervision = list(range(8))
discrete_angle_delta_value = [0, 0, 0, 2, -2, 0, 1, -1, 0]
discrete_actions=[(1.,0.,0.), (0.4,0.4,0.), (0.4,-0.4,0.),
                  (0.,0.,1.), (0.,0.,-1.),
                  (0.4,0.,0.),
                  (0.,0.,0.5), (0.,0.,-0.5), (0., 0., 0.)]
discrete_action_names = ['Forward', 'Left-Fwd', 'Right-Fwd', 'Left-Rotate', 'Right-Rotate',
                         'Small-Forward', 'Small-Left-Rot', 'Small-Right-Rot', 'Stay']
_full_discrete_actions = discrete_actions + [(0, 0.4, 0), (0, -0.4, 0), (-0.4, 0, 0)]  # L, R, Bck, total <12> actions
_full_discrete_angle_delta_value = _full_discrete_actions + [0, 0, 0]
_full_discrete_action_names = discrete_action_names + ['Left', 'Right', 'Backward']

n_discrete_actions = len(discrete_actions)
n_discrete_angles = 360 // discrete_rotation_sensitivity

# criteria for seeing the object
n_pixel_for_object_see = 450   # need at least see 450 pixels for success under default resolution 120 x 90
n_pixel_for_object_sense = 50
L_pixel_reward_range = n_pixel_for_object_see - n_pixel_for_object_sense
pixel_object_reward = 0.4


#################
# Util Functions
#################

def reset_see_criteria(resolution, lo_rate=0.005, up_rate=0.045):
    total_pixel = resolution[0] * resolution[1]
    global n_pixel_for_object_see, n_pixel_for_object_sense, L_pixel_reward_range
    n_pixel_for_object_see = max(int(total_pixel * up_rate), 5)
    n_pixel_for_object_sense = max(int(total_pixel * lo_rate), 1)
    L_pixel_reward_range = n_pixel_for_object_see - n_pixel_for_object_sense


class RoomNavTask(gym.Env):
    def __init__(self, env,
                 seed=None,
                 reward_type='delta',
                 hardness=None,
                 max_birthplace_steps=None,
                 move_sensitivity=None,
                 segment_input=True,
                 joint_visual_signal=False,
                 depth_signal=True,
                 max_steps=-1,
                 success_measure='see',
                 discrete_action=False,
                 include_object_target=False,
                 reward_silence=0,
                 birthplace_curriculum_schedule=None,
                 target_mask_signal=False,
                 false_rate=0.0,
                 discrete_angle=False,
                 supervision_signal=False,
                 min_birth_grid_dist=0,
                 cache_discrete_angles=False):
        """RoomNav task wrapper with gym api
        Note:
            all the settings are the default setting to run a task
            only the <env> argument is necessary to launch the task

        Args:
            env: an instance of environment (multi-house or single-house)
            seed: if not None, set the random seed
            reward_type (str, optional): reward shaping, currently available: none, linear, indicator, delta and speed.
                                         <new> means temporary reward function under development
            hardness (double, optional): if not None, must be a real number between 0 and 1, indicating the hardness
                                         namely the distance from birthplace to target (1 is the hardest)
                                         None means 1.0
            max_birthplace_steps (int, optional): if not None, the birthplace will be no further than <this> amount of action steps from target
            move_sensitivity (double, optional): if not None, set the maximum movement per time step (generally should not be changed)
            segment_input (bool, optional): whether to use semantic segmentation mask for observation
            joint_visual_signal (bool, optional): when true, use both visual signal and segmentation mask as observation
                                                  when true, segment_input will be set true accordingly
            target_mask_signal (bool, optional): when true, the last channel of input will be a 0/1 mask indicating the pixel of the target category
            depth_signal (bool, optional): whether to include depth signal in observation
            max_steps (int, optional): when max_steps > 0, the task will be cut after <max_steps> steps
            success_measure (str, optional): criteria for success, currently support 'see' and 'stay'
            discrete_action (bool, optional):  when true, use discrete actions; otherwise use continuous actions
            include_object_target (bool, optional): when true, target can be an object category
            reward_silence (int, optional): when set, the first <reward_silence> steps in each episode will not have reward other than collision penalty
            birthplace_curriculum_schedule (<int,int,int>, optional): when set, it is <start_birthplace_steps, incremental, update_frequency>
            false_rate: the rate of task that the target is not reachable
            discrete_angle: when True, the angle is always a multiplier of <discrete_rotation_sensitivity>
            supervision_signal: when True, we will cache the supervision signal for all the targets in all the houses
            min_birth_grid_dist: the minimum grid distance between birthplace and the target (connMap[b_x, b_y] >= min_dist)
            cache_discrete_angles: when True, we will pre-compute the actions w.r.t. each discrete rotation angles
        """
        assert false_rate < 1e-8, 'Currently Only Support Valid Tasks!!!! False_Rate Must Be 0.0!!!'

        self.env = env
        #assert isinstance(env, Environment), '[RoomNavTask] env must be an instance of Environment!'
        if env.resolution != (120, 90): reset_see_criteria(env.resolution)
        self.resolution = resolution = env.resolution
        assert reward_type in [None, 'none', 'linear', 'indicator', 'delta', 'speed', 'new']
        self.reward_type = reward_type
        self.reward_silence = reward_silence
        self.colorDataFile = self.env.config['colorFile']
        self.segment_input = segment_input
        self.joint_visual_signal = joint_visual_signal
        self.depth_signal = depth_signal
        self.target_mask_signal = target_mask_signal
        n_channel = 3
        if segment_input:
            self.env.set_render_mode('semantic')
        else:
            self.env.set_render_mode('rgb')
        if joint_visual_signal: n_channel += 3
        if depth_signal: n_channel += 1
        if target_mask_signal: n_channel += 1
        self._observation_shape = (resolution[1], resolution[0], n_channel)
        self._observation_space = spaces.Box(0, 255, shape=self._observation_shape)

        self.max_steps = max_steps

        self.discrete_action = discrete_action
        if discrete_action:
            self._action_space = spaces.Discrete(n_discrete_actions)
        else:
            self._action_space = spaces.Tuple([spaces.Box(0, 1, shape=(4,)), spaces.Box(0, 1, shape=(2,))])

        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        self.include_object_target = include_object_target  # whether to use object targets

        # configs
        self.move_sensitivity = (move_sensitivity or default_move_sensitivity)  # at most * meters per frame
        self.rot_sensitivity = rotation_sensitivity
        self.dist_scale = dist_reward_scale or 1.0
        self.successRew = success_reward if reward_type != 'new' else new_success_reward
        self.inroomRew = (stay_room_reward if reward_type != 'new' else new_stay_room_reward) or 0.2
        self.colideRew = collision_penalty_reward or 0.02
        self.goodMoveRew = correct_move_reward or 0.0
        self.pixelRew = (pixel_object_reward if reward_type != 'new' else new_pixel_object_reward) or 0.0
        self.pixelRewBase = L_pixel_reward_range if reward_type != 'new' else int(new_pixel_reward_rate * resolution[0] * resolution[1])
        self.timePenalty = time_penalty_reward if reward_type != 'new' else new_time_penalty_reward
        self.succSeeSteps = (success_see_target_time_steps if reward_type != 'new' else new_success_stay_time_steps)

        self.last_obs = None
        self.last_info = None
        self._cached_obs = None
        self._cached_seg = None
        self._cached_mask = None if not target_mask_signal else np.zeros((self.resolution[1],self.resolution[0],1), dtype=np.uint8)
        self._object_cnt = 0

        # config hardness
        self.hardness = None
        self.max_birthplace_steps = None
        self.min_birth_grid_dist = 0
        self.availCoorsSize = None
        self.availCoorsLower = 0
        self._availCoorsSizeDict = None
        if birthplace_curriculum_schedule is not None:
            if (not hasattr(birthplace_curriculum_schedule, '__len__')) \
                    or (len(birthplace_curriculum_schedule) != 3) \
                    or (min(birthplace_curriculum_schedule) < 1):
                print('>>> [RoomNav] Error! <birthplace_curriculum_schedule> must be a positive int triple! reset to None!')
                birthplace_curriculum_schedule = None
        self.curriculum_schedule = birthplace_curriculum_schedule
        self.reset_hardness(hardness,
                            max_birthplace_steps if birthplace_curriculum_schedule is None else birthplace_curriculum_schedule[0],
                            min_birth_grid_dist)
        if self.curriculum_schedule is not None:
            self.curriculum_schedule = (max_birthplace_steps, birthplace_curriculum_schedule[1], birthplace_curriculum_schedule[2])
        self.discrete_angle = None
        self._yaw_ind = None
        if discrete_angle:
            assert 360 % discrete_rotation_sensitivity == 0, 'When discrete_angle, 360 must be divided by <discrete_rotation_sensitivity = {}>!'.format(rotation_sensitivity)
            self.discrete_angle = 360 // discrete_rotation_sensitivity
        self.supervision_signal = supervision_signal
        if supervision_signal or cache_discrete_angles:
            assert (discrete_angle and discrete_action), 'When <supervision_signal>, <discrete_angle> AND <discrete_action> must be True!'
            assert (len(allowed_actions_for_supervision) <= 8), 'Only support <allowed_actions_for_supervision> containing at most 8 actions!'
            # assume shortest paths have been cached
            self._angle_dir = []
            _t_info = self.env.info
            _cx, _cy = _t_info['loc']
            _yaw = _t_info['yaw']
            for d in range(self.discrete_angle):
                self.env.reset(x=_cx,y=_cy,yaw=d * discrete_rotation_sensitivity - 180.0)
                _frt = self.env.get_front_dir()
                _rght = self.env.get_right_dir()
                self._angle_dir.append((_frt[0], _frt[1], _rght[0], _rght[1]))
            self._allowed_sup_action_idx = allowed_actions_for_supervision
            self._allowed_sup_actions = []
            for a in self._allowed_sup_action_idx:
                dx,dy,dd = discrete_actions[a]
                if dd < -0.7:
                    action=(dx,dy,-2)
                elif dd < -0.1:
                    action=(dx,dy,-1)
                elif dd < 0.1:
                    action=(dx,dy,0)
                elif dd < 0.7:
                    action=(dx,dy,1)
                else:
                    action=(dx,dy,2)
                self._allowed_sup_actions.append(action)
            if supervision_signal:
                self.env.cache_supervision(self._angle_dir, self._allowed_sup_actions)

        # temp storage
        self.collision_flag = False

        # episode counter
        self.current_episode_step = 0
        self.total_episode_cnt = 0
        self._prev_object_see_rate = 0

        # config success measure
        assert success_measure in ['stay', 'see', 'stop', 'see-stop']
        if 'stop' in success_measure:
            assert self.discrete_action, 'success_measure <stop> only supports discrete action'
        self.success_measure = success_measure
        print('[RoomNavTask] >> Success Measure = <{}>'.format(success_measure))
        self.success_stay_cnt = 0
        if 'see' in success_measure:
            self.room_target_object = dict(outdoor=[])  # outdoor is a special category
            self.room_target_object_names = dict(outdoor=[])
            self._load_target_object_data(self.env.config['roomTargetFile'])
            if self.include_object_target:
                self._load_target_object_data(self.env.config['objectTargetFile'])


    def _load_target_object_data(self, roomTargetFile):
        with open(roomTargetFile) as csvFile:
            reader = csv.DictReader(csvFile)
            for row in reader:
                c = np.array((row['r'],row['g'],row['b']), dtype=np.uint8)
                room = row['target_room']
                if room not in self.room_target_object:
                    self.room_target_object[room] = []
                    self.room_target_object_names[room] = []
                self.room_target_object[room].append(c)
                self.room_target_object_names[room].append(row['coarse_grained_class'])

    """
    reset the target room type to navigate to
    when target is None, a valid target will be randomly selected
    """
    def reset_target(self, target=None):
        if target is None:
            desired_target_list = self.house.all_desired_targetTypes if self.include_object_target else self.house.all_desired_roomTypes
            target = random.choice(desired_target_list)
        elif target == 'any-room':
            desired_target_list = self.house.all_desired_roomTypes
            target = random.choice(desired_target_list)
        elif target == 'any-object':
            desired_target_list = self.house.all_desired_targetObj
            target = random.choice(desired_target_list)
        #else:
        #    assert target in desired_target_list, '[RoomNavTask] desired target <{}> does not exist in the current house!'.format(target)
        if self.house.setTargetRoom(target):  # target room changed!!!
            if self.supervision_signal: self.house.load_supervision_map(target)
            _id = self.house._id
            if self.house.targetRoomTp not in self._availCoorsSizeDict[_id]:
                self.availCoorsLower = 0
                if (self.hardness is None) and (self.max_birthplace_steps is None) and (self.min_birth_grid_dist == 0):
                    self.availCoorsSize = self.house.getConnectedLocationSize()
                else:
                    allowed_dist = self.house.maxConnDist
                    if self.hardness is not None:
                        allowed_dist = min(int(self.house.maxConnDist * self.hardness + 1e-10), allowed_dist)
                    if self.max_birthplace_steps is not None:
                        allowed_dist = min(self.house.getAllowedGridDist(self.max_birthplace_steps * self.move_sensitivity), allowed_dist)
                    self.availCoorsSize = self.house.getConnectedLocationSize(max_allowed_dist=allowed_dist)
                    curr_min_dist = min(self.min_birth_grid_dist, allowed_dist)   # in case of blocked region
                    if curr_min_dist > 0:
                        self.availCoorsLower = self.house.getConnectedLocationSize(max_allowed_dist=curr_min_dist-1)
                        self.availCoorsSize -= self.availCoorsLower
                self._availCoorsSizeDict[_id][self.house.targetRoomTp] = (self.availCoorsLower, self.availCoorsSize)
            else:
                self.availCoorsLower, self.availCoorsSize = self._availCoorsSizeDict[_id][self.house.targetRoomTp]


    @property
    def house(self):
        return self.env.house

    def reset_obs_mode(self, segment_input, depth_signal=True, target_mask_signal=False, joint_visual_signal=False):
        self.segment_input = segment_input
        self.joint_visual_signal = joint_visual_signal
        self.depth_signal = depth_signal
        self.target_mask_signal = target_mask_signal
        n_channel = 3
        if segment_input:
            self.env.set_render_mode('semantic')
        else:
            self.env.set_render_mode('rgb')
        if joint_visual_signal: n_channel += 3
        if depth_signal: n_channel += 1
        if target_mask_signal: n_channel += 1
        self._observation_shape = (self.resolution[1], self.resolution[0], n_channel)
        self._observation_space = spaces.Box(0, 255, shape=self._observation_shape)
        self._re_render()

    def get_obs_mode(self):
        return dict(segment_input=self.segment_input,
                    depth_signal=self.depth_signal,
                    target_mask_signal=self.target_mask_signal,
                    joint_visual_signal=self.joint_visual_signal)

    def _re_render(self):
        # generate state
        self._cached_seg = None
        self.last_obs = self.env.render()
        if self.joint_visual_signal:
            self.last_obs = np.concatenate([self.env.render(mode='rgb'), self.last_obs], axis=-1)
        ret_obs = self.last_obs
        if self.depth_signal:
            dep_sig = self.env.render(mode='depth')
            if dep_sig.shape[-1] > 1:
                dep_sig = dep_sig[..., 0:1]
            ret_obs = np.concatenate([ret_obs, dep_sig], axis=-1)
        if self.target_mask_signal:
            ret_obs = np.concatenate([ret_obs, self._gen_target_mask()], axis=-1)
        self._cached_obs = ret_obs
        return ret_obs

    """
    gym api: reset function
    when target is not None, we will set the target room type to navigate to
    """
    def reset(self, target=None, reset_house=True, birthplace=None):
        # increase episode counter
        self.total_episode_cnt += 1
        if (self.curriculum_schedule is not None) \
            and (self.total_episode_cnt % self.curriculum_schedule[2] == 0) \
            and (self.max_birthplace_steps < self.curriculum_schedule[0]):  # curriculum learning
            next_birthplace_steps = min(self.max_birthplace_steps + self.curriculum_schedule[1],
                                        self.curriculum_schedule[0])
            self.reset_hardness(self.hardness, max_birthplace_steps=next_birthplace_steps)

        # clear episode steps
        self.current_episode_step = 0
        self.success_stay_cnt = 0
        self._object_cnt = 0
        self._prev_object_see_rate = 0.0

        # reset house
        if reset_house:
            self.env.reset_house()
            self.house.targetRoomTp = None  # [NOTE] IMPORTANT! clear this!!!!!

        # reset target room
        self.reset_target(target=target)  # randomly reset
        self.collision_flag = False

        # general birth place
        if birthplace is None:
            x, y = self.house.getIndexedConnectedLocation(np.random.randint(self.availCoorsSize) + self.availCoorsLower)
        else:
            x, y = birthplace
        yaw = None
        if self.discrete_angle is not None:
            self._yaw_ind = np.random.randint(self.discrete_angle)
            yaw = self._yaw_ind * discrete_rotation_sensitivity - 180.0

        # generate state
        self._cached_seg = None
        self.env.reset(x=x, y=y, yaw=yaw)
        self.last_obs = self.env.render()
        if self.joint_visual_signal:
            self.last_obs = np.concatenate([self.env.render(mode='rgb'), self.last_obs], axis=-1)
        ret_obs = self.last_obs
        if self.depth_signal:
            dep_sig = self.env.render(mode='depth')
            if dep_sig.shape[-1] > 1:
                dep_sig = dep_sig[..., 0:1]
            ret_obs = np.concatenate([ret_obs, dep_sig], axis=-1)
        if self.target_mask_signal:
            ret_obs = np.concatenate([ret_obs, self._gen_target_mask()], axis=-1)
        self.last_info = self.info
        self._cached_obs = ret_obs
        return ret_obs

    def _apply_action(self, action):
        if self.discrete_action:
            if action < n_discrete_actions:
                return discrete_actions[action]
            else:
                return _full_discrete_actions[action]
        else:
            rot = action[1][0] - action[1][1]
            act = action[0]
            return (act[0] - act[1]), (act[2] - act[3]), rot

    def _fetch_cached_segmentation(self):
        if (self.last_obs is not None) and self.segment_input:
            seg_obs = self.last_obs if not self.joint_visual_signal else self.last_obs[:,:,3:6]
        else:
            if self._cached_seg is None: self._cached_seg = self.env.render(mode='semantic')
            seg_obs = self._cached_seg
        return seg_obs

    """
    return 0/1 binary mask, indicating the target pixels
    """
    def _gen_target_mask(self):
        if self._cached_mask is None:
            self._cached_mask = np.zeros((self.resolution[1],self.resolution[0],1), dtype=np.uint8)
        else:
            self._cached_mask[:, :] = 0
        seg_obs = self._fetch_cached_segmentation()
        object_color_list = self.room_target_object[self.house.targetRoomTp]
        for c in object_color_list:
            self._cached_mask[np.all(seg_obs==c, axis=2),:]=250   # NOTE: when processed by NN, image will be scaled
        return self._cached_mask

    def _is_success(self, raw_dist, grid, act):
        if raw_dist > 0:
            self.success_stay_cnt = 0
            return False
        if self.success_measure == 'stay':
            self.success_stay_cnt += 1
            return self.success_stay_cnt >= success_stay_time_steps
        if self.success_measure == 'stop':
            return act == (n_discrete_actions - 1)   # stay action
        if (self.success_measure == 'see-stop') and (act != n_discrete_actions - 1):
            return False
        # self.success_measure == 'see'
        object_color_list = self.room_target_object[self.house.targetRoomTp]
        flag_see_target_objects = (len(object_color_list) == 0)
        if self._cached_mask is not None:
            self._object_cnt = np.sum(self._cached_mask)
            if self._object_cnt >= n_pixel_for_object_see:
                flag_see_target_objects = True
        else:
            seg_obs = self._fetch_cached_segmentation()
            self._object_cnt = 0
            for c in object_color_list:
                cur_n = np.sum(np.all(seg_obs == c, axis=2))
                self._object_cnt += cur_n
                if self._object_cnt >= n_pixel_for_object_see:
                    flag_see_target_objects = True
                    break
        if self.success_measure == 'see-stop':
            return flag_see_target_objects
        if flag_see_target_objects:
            self.success_stay_cnt += 1
        else:
            self.success_stay_cnt = 0  # did not see any target objects!
        return self.success_stay_cnt >= self.succSeeSteps

    """
    gym api: step function
    return: obs, reward, done, info (a dictionary containing some state information)
    """
    def step(self, action):
        reward = 0
        det_fwd, det_hor, det_rot = self._apply_action(action)
        move_fwd = det_fwd * self.move_sensitivity
        move_hor = det_hor * self.move_sensitivity
        rotation = det_rot * self.rot_sensitivity

        self.env.rotate(rotation)
        if not self.env.move_forward(move_fwd, move_hor):
            if flag_print_debug_info:
                print('Collision! No Movement Performed!')
            reward -= self.colideRew
            self.collision_flag = True
        else:
            self.collision_flag = False
            if (self.discrete_angle is not None) and (action < n_discrete_actions):
                self._yaw_ind = (self._yaw_ind + discrete_angle_delta_value[action] + self.discrete_angle) % self.discrete_angle
            if flag_print_debug_info:
                print('Move Successfully!')

        # generate observation
        self._cached_seg = None
        self.last_obs = obs = self.env.render()
        if self.joint_visual_signal:
            self.last_obs = obs = np.concatenate([self.env.render(mode='rgb'), obs], axis=-1)
        if self.depth_signal:
            dep_sig = self.env.render(mode='depth')
            if dep_sig.shape[-1] > 1:
                dep_sig = dep_sig[..., 0:1]
            obs = np.concatenate([obs, dep_sig], axis=-1)
        if self.target_mask_signal:
            obs = np.concatenate([obs, self._gen_target_mask()], axis=-1)

        # compute reward
        cur_info = self.info
        raw_dist = cur_info['dist']
        orig_raw_dist = self.last_info['dist']
        raw_scaled_dist = cur_info['scaled_dist']
        dist = raw_scaled_dist * self.dist_scale
        done = False
        if self._is_success(raw_dist, cur_info['grid'], action):
            reward += self.successRew
            done = True
        # accumulate episode step
        self.current_episode_step += 1
        if ('stop' in self.success_measure) and (action == n_discrete_actions - 1):
            if not done: reward += wrong_stop_penalty
            done = True
        if (self.max_steps > 0) and (self.current_episode_step >= self.max_steps): done = True

        # reward shaping: distance related reward
        if self.current_episode_step > self.reward_silence:
            if raw_dist < orig_raw_dist:
                reward += self.goodMoveRew
            if raw_dist == 0:
                reward += self.inroomRew

        # reward shaping: general
        if self.current_episode_step > self.reward_silence:
            if self.reward_type == 'linear':
                reward -= dist
            elif self.reward_type == 'indicator':
                if raw_dist != orig_raw_dist:  # indicator reward
                    reward += indicator_reward if raw_dist < orig_raw_dist else -indicator_reward
                if raw_dist >= orig_raw_dist: reward -= self.timePenalty
            elif self.reward_type == 'delta':
                delta_raw_dist = orig_raw_dist - raw_dist
                ratio = self.move_sensitivity / self.house.grid_det
                delta_reward = delta_raw_dist / ratio * delta_reward_coef
                delta_reward = np.clip(delta_reward, -indicator_reward, indicator_reward)
                reward += delta_reward
                if raw_dist >= orig_raw_dist: reward -= self.timePenalty
            elif self.reward_type == 'speed':
                movement = np.sqrt((self.last_info['loc'][0]-cur_info['loc'][0])**2
                                   + (self.last_info['loc'][1]-cur_info['loc'][1])**2)
                sign = np.sign(orig_raw_dist - raw_dist)
                det_dist = movement * sign * speed_reward_coef
                det_dist = np.clip(det_dist, -indicator_reward, indicator_reward)
                reward += det_dist
                if raw_dist >= orig_raw_dist: reward -= self.timePenalty
            elif self.reward_type in 'new':
                # utilize delta reward but with different parameters
                delta_raw_dist = orig_raw_dist - raw_dist
                ratio = self.move_sensitivity / self.house.grid_det
                new_reward = delta_raw_dist / ratio * new_reward_coef
                new_reward = np.clip(new_reward, -new_reward_bound, new_reward_bound)
                reward += new_reward
                if raw_dist >= orig_raw_dist: reward -= self.timePenalty   # always deduct time penalty
                if (orig_raw_dist == 0) and (raw_dist > 0): reward -= new_leave_penalty  # big penalty when leave target room

        # object seen reward
        if self.current_episode_step > self.reward_silence:
            if (raw_dist == 0) and ('see' in self.success_measure):  # inside target room and success measure is <see>
                if not done:
                    curr_obj_see_rate = np.clip((self._object_cnt - n_pixel_for_object_sense) / self.pixelRewBase, 0., 1.)
                    object_reward = self.pixelRew * (curr_obj_see_rate if self.reward_type != 'new' else (curr_obj_see_rate - self._prev_object_see_rate))
                    self._prev_object_see_rate = curr_obj_see_rate
                    reward += object_reward
        self.last_info = cur_info
        self._cached_obs = obs
        return obs, reward, done, cur_info

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    @property
    def info(self):
        ret = self.env.info
        gx, gy = ret['grid']
        ret['dist'] = dist = self.house.connMap[gx, gy]
        ret['scaled_dist'] = self.house.getScaledDist(gx, gy)
        ret['meters'] = dist * self.house.grid_det
        ret['optsteps'] = int(dist / (self.move_sensitivity / self.house.grid_det) + 0.5)
        ret['collision'] = int(self.collision_flag)
        ret['target_room'] = self.house.targetRoomTp
        if self.supervision_signal:
            sup_act = self.house.get_supervision(gx, gy, self._yaw_ind)
            if sup_act >= 0: sup_act = self._allowed_sup_action_idx[sup_act]
            ret['supervision'] = sup_act
        if self.discrete_angle is not None:
            ret['_yaw_ind'] = self._yaw_ind
        return ret

    def get_current_target(self):
        return self.house.targetRoomTp

    """
    [Sandbox/Graph] return auxiliary mask
    """
    def get_aux_tags(self):
        cx, cy = self.env.info['loc']
        return self.house.get_target_mask(cx, cy)

    """
    [Sandbox/Graph] return auxiliary mask in global feature representation
    """
    def get_feature_mask(self):
        cx, cy = self.env.info['loc']
        return self.house.get_global_mask_feature(cx, cy)

    """
    [Sandbox/Graph] return optimal plan info
    """
    def get_optimal_plan(self):
        cx, cy = self.env.info['loc']
        return self.house.get_optimal_plan(cx, cy, self.house.targetRoomTp)

    """
    return all the available target room types of the current house
    """
    def get_avail_targets(self):
        return self.house.all_desired_roomTypes

    """
    reset the hardness of the task
      --> Note: hardness will be only updated when some param is not None
                if None is given, the original hardness param will be kept
    """
    def reset_hardness(self, hardness=None, max_birthplace_steps=None, min_birth_grid_dist=None):
        self.hardness = hardness or self.hardness
        self.max_birthplace_steps = max_birthplace_steps or self.max_birthplace_steps
        self.min_birth_grid_dist = min_birth_grid_dist or self.min_birth_grid_dist
        self.availCoorsSizeLower = 0
        if (self.hardness is None) and (self.max_birthplace_steps is None) and (self.min_birth_grid_dist == 0):
            self.availCoorsSize = self.house.getConnectedLocationSize()
        else:
            allowed_dist = self.house.maxConnDist
            if hardness is not None:
                allowed_dist = min(int(self.house.maxConnDist * hardness + 1e-10), allowed_dist)
            if max_birthplace_steps is not None:
                allowed_dist = min(self.house.getAllowedGridDist(max_birthplace_steps * self.move_sensitivity), allowed_dist)
            self.availCoorsSize = self.house.getConnectedLocationSize(max_allowed_dist=allowed_dist)
            curr_min_dist = min(self.min_birth_grid_dist, allowed_dist)
            if curr_min_dist > 0:
                self.availCoorsLower = self.house.getConnectedLocationSize(max_allowed_dist=curr_min_dist - 1)
                self.availCoorsSize -= self.availCoorsLower

        n_house = self.env.num_house
        self._availCoorsSizeDict = [dict() for _ in range(n_house)]
        self._availCoorsSizeDict[self.house._id][self.house.targetRoomTp] = (self.availCoorsLower, self.availCoorsSize)

    """
    recover the state (location) of the agent from the info dictionary
    """
    def set_state(self, info):
        self.env.reset(x=info['loc'][0], y=info['loc'][1], yaw=info['yaw'])
        if self.discrete_angle is not None:
            self._yaw_ind = info['_yaw_ind']

    """
    return 2d topdown map
    """
    def get_2dmap(self):
        return self.env.gen_2dmap()

    """
    show a image.
    if img is not None, it will visualize the img to monitor
    if img is None, it will return a img object of the observation
    Note: whenever this function is called with img not None, the task cannot perform rendering any more!!!!
    """
    def show(self, img=None):
        return self.env.show(img=img,
                             renderMapLoc=(None if img is not None else self.info['grid']),
                             renderSegment=True,
                             display=(img is not None))

    def debug_show(self):
        return self.env.debug_render()

    def gen_supervised_plan(self, birth_state=None,
                            return_numpy_frames=False, max_allowed_steps=None,
                            mask_feature_dim=None, logging=False):
        assert len(self._angle_dir) == n_discrete_angles
        assert self.discrete_angle == n_discrete_angles

        if not hasattr(self, '_flag_cached_objdist'):
            self._flag_cached_objdist = dict()

        # prepare heuristic function distance map
        cached_key = (self.env.info['house_id'], self.house.targetRoomTp)
        if cached_key not in self._flag_cached_objdist:
            flag_okay = self.house._genObjDistForTargetRoom(self.house.targetRoomTp, self.room_target_object_names[self.house.targetRoomTp])
            if flag_okay:
                self._flag_cached_objdist[cached_key] = self.house.targetRoomTp + '-obj'
            else:
                self._flag_cached_objdist[cached_key] = self.house.targetRoomTp
        h_func_map_tag = self._flag_cached_objdist[cached_key]


        print('--> h_func_map_tag = {}'.format(h_func_map_tag))


        if birth_state is None:
            birth_cx, birth_cy = self.env.info['loc']
            birth_rot_ind = self.info['_yaw_ind']
        else:
            birth_cx, birth_cy, birth_rot_ind = birth_state
            if not isinstance(birth_rot_ind):
                birth_rot_ind = int(birth_rot_ind + 180 + 1e-10) // discrete_rotation_sensitivity

        prec = 10
        birth_rot_ind %= n_discrete_angles
        def h_func(cx, cy):
            gx, gy = self.house.to_grid(cx, cy)
            dist = self.house.targetDist(h_func_map_tag, gx, gy)
            return self.house.getOptSteps(dist, self.move_sensitivity)

        opt = {}

        def check_update(cx, cy, rot, step):
            cx = round(cx, prec)
            cy = round(cy, prec)
            t_state = (cx, cy, rot)
            if (t_state not in opt) or (step < opt[t_state][0]):
                return t_state
            return None

        def check_action(cx, cy, rot, act, is_rev=False):
            dx = self._allowed_sup_actions[act][0] * self._angle_dir[rot][0] \
                 + self._allowed_sup_actions[act][1] * self._angle_dir[rot][2]
            dy = self._allowed_sup_actions[act][0] * self._angle_dir[rot][1] \
                 + self._allowed_sup_actions[act][1] * self._angle_dir[rot][3]
            dr = self._allowed_sup_actions[act][2]
            if is_rev:
                dx, dy, dr = -dx, -dy, -dr
            t_cx = cx + dx
            t_cy = cy + dy
            if is_rev or self.env._check_collision((cx, 0, cy), (t_cx, 0, t_cy)):
                t_rot = (rot + dr) % n_discrete_angles
                return t_cx, t_cy, t_rot
            return None

        state = check_update(birth_cx, birth_cy, birth_rot_ind, 0)
        hp = [(h_func(birth_cx, birth_cy), 0, (birth_cx, birth_cy, birth_rot_ind))]
        opt[state] = (0, -1)

        self.last_obs = None
        self._cached_seg = None
        flag_found = False

        iters = 0
        st_cnt = 1
        stop_act = len(allowed_actions_for_supervision)  # index for STOP action
        while len(hp) > 0:
            iters += 1
            dat = heapq.heappop(hp)
            cur_step = dat[1]
            cx, cy, rot = dat[2]
            gx, gy = self.house.to_grid(cx, cy)
            raw_dist = self.house.connMap[gx, gy]

            if iters % 100 == 0:
                print('>>>>>>> current <%d> states expanded!! current step = %d, heuristic = %d, raw_dist = %d' % (iters, cur_step, dat[0]-cur_step, raw_dist))


            # set up camera for success checking
            yaw = rot * discrete_rotation_sensitivity - 180.0
            self.env.reset(x=cx, y=cy, yaw=yaw)
            # clear render cache
            self.last_obs = None
            self._cached_seg = None
            if self._is_success(raw_dist, (gx, gy), stop_act) or self.success_stay_cnt > 0:
                flag_found = True
                break  # succeed
            for a_id in allowed_actions_for_supervision:
                next = check_action(cx, cy, rot, a_id)
                if next is None:
                    continue
                t_state = check_update(next[0], next[1], next[2], cur_step + 1)
                if t_state is not None:
                    opt[t_state] = (cur_step + 1, a_id)
                    h_val = h_func(next[0], next[1])  # A* heuristic function
                    heapq.heappush(hp, (cur_step + 1 + h_val, cur_step + 1, next))
                    st_cnt += 1
        assert flag_found, '[RoomNav] Gen_Supervised_Plan Error!! Not Available Plan Found! Birth = {}, target = <{}>'.format((birth_cx, birth_cy, birth_rot_ind), self.house.targetRoomTp)

        print('Path Found!!!')

        # Trace Back Shortest Path
        def get_prev_state(raw_cx, raw_cy, rot):
            cx = round(raw_cx, prec)
            cy = round(raw_cy, prec)
            state = (cx, cy, rot)
            act = opt[state][1]
            if act < 0:  # birth place
                prev = None
            else:
                prev = check_action(raw_cx, raw_cy, rot, act, is_rev=True)
            return prev, act

        act = stop_act
        prev = (cx, cy, rot)
        plan = []
        while prev is not None:
            plan.append((prev[0], prev[1], prev[2] * discrete_rotation_sensitivity - 180.0, act))
            prev, act = get_prev_state(*prev)
        if max_allowed_steps is not None:
            plan = plan[:max_allowed_steps]
        plan.reverse()
        if not return_numpy_frames:
            if logging: return plan, {"iterations": iters, "state_expanded": st_cnt}
            return plan

        # render frames and return numpy arrays
        n_plan = len(plan)
        np_act = np.zeros(n_plan, dtype=np.int32)
        np_frames = np.zeros((n_plan, self._observation_shape[0], self._observation_shape[1], self._observation_shape[2]), dtype=np.uint8)
        np_mask_feat = np.zeros((n_plan, mask_feature_dim), dtype=np.uint8) if mask_feature_dim is not None else None
        for i, dat in enumerate(plan):
            cx, cy, yaw, a = dat
            self.env.reset(x=cx, y=cy, yaw=yaw)
            np_frames[i, ...] = self._re_render()
            np_act[i] = a
            if np_mask_feat is not None:
                np_mask_feat[i, :] = self.env.get_feature_mask()[: mask_feature_dim]
        ret_data = [np_frames, np_act]
        if np_mask_feat is not None:
            ret_data.append(np_mask_feat)
        if logging:
            ret_data.append({"iterations": iters, "state_expanded": st_cnt})
        return ret_data

    ###################
    # DEBUG USE
    ###################
    def _sanity_check_supervised_plan(self, info, actions):
        print('Sanity Checking ....')
        # init birth state
        if info['house_id'] != self.info['house_id']:
            self.env.reset_house(info['house_id'])
        self.reset(target=info['target_room'], reset_house=False, birthplace=info['loc'])
        self.env.reset(x=info['loc'][0], y=info['loc'][1], yaw=info['yaw'])
        # start simulation
        n = len(actions)
        accu_reward = 0
        flag_done = False
        flag_full = False
        for i in range(n):
            a = actions[i]
            _, reward, done, _ = self.step(a)
            accu_reward += reward
            if done:
                flag_done = reward > 4
                flag_full = (i == n - 1)
                break
        print('  >> Finished!')
        print('    -> Total Input Actions = {}'.format(n))
        print('    -> Accu Rew = {}'.format(accu_reward))
        print('    -> Success = {}'.format(flag_done))
        print('    -> Terminate in the last step = {}'.format(flag_full))
        return flag_done and flag_full
