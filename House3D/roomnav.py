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

######################################
# reward function parameters for new
######################################
new_time_penalty_reward = 0.1   # penalty for each time step
new_reward_coef = 1.0
new_reward_bound = 0.5
new_leave_penalty = 0.5
new_stay_room_reward = 0.05
new_success_stay_time_steps = 3
new_success_reward = 10
new_pixel_object_reward = 0.1
#####################################


# sensitivity setting
rotation_sensitivity = 30  # 45   # maximum rotation per time step
default_move_sensitivity = 0.5  # 1.0   # maximum movement per time step

# discrete action space actions, totally <13> actions
# Fwd, L, R, LF, RF, Lrot, Rrot, Bck, s-Fwd, s-L, s-R, s-Lrot, s-Rrot, Stay
discrete_actions=[(1.,0.,0.), (0.,1.,0.), (0.,-1.,0.), (0.5,0.5,0.), (0.5,-0.5,0.),
                  (0.,0.,1.), (0.,0.,-1.),
                  (-0.4,0.,0.),
                  (0.4,0.,0.), (0.,0.4,0.), (0.,-0.4,0.),
                  (0.,0.,0.4), (0.,0.,-0.4), (0., 0., 0.)]
n_discrete_actions = len(discrete_actions)

# criteria for seeing the object
n_pixel_for_object_see = 450   # need at least see 450 pixels for success under default resolution 120 x 90
n_pixel_for_object_sense = 50
L_pixel_reward_range = n_pixel_for_object_see - n_pixel_for_object_sense
pixel_object_reward = 0.4


#################
# Util Functions
#################

def reset_see_criteria(resolution):
    total_pixel = resolution[0] * resolution[1]
    global n_pixel_for_object_see, n_pixel_for_object_sense, L_pixel_reward_range
    n_pixel_for_object_see = max(int(total_pixel * 0.045), 5)
    n_pixel_for_object_sense = max(int(total_pixel * 0.005), 1)
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
                 reward_silence=0):
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
            depth_signal (bool, optional): whether to include depth signal in observation
            max_steps (int, optional): when max_steps > 0, the task will be cut after <max_steps> steps
            success_measure (str, optional): criteria for success, currently support 'see' and 'stay'
            discrete_action (bool, optional):  when true, use discrete actions; otherwise use continuous actions
            include_object_target (bool, optional): when true, target can be an object category
            reward_silence (int, optional): when set, the first <reward_silence> steps in each episode will not have reward other than collision penalty
        """
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
        n_channel = 3
        if segment_input:
            self.env.set_render_mode('semantic')
        else:
            self.env.set_render_mode('rgb')
        if joint_visual_signal: n_channel += 3
        if depth_signal: n_channel += 1
        self._observation_shape = (resolution[0], resolution[1], n_channel)
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
        self.succSeeSteps = (success_see_target_time_steps if reward_type != 'new' else new_success_stay_time_steps)

        self.last_obs = None
        self.last_info = None
        self._object_cnt = 0

        # config hardness
        self.hardness = None
        self.max_birthplace_steps = None
        self.availCoorsSize = None
        self._availCoorsSizeDict = None
        self.reset_hardness(hardness, max_birthplace_steps)

        # temp storage
        self.collision_flag = False

        # episode counter
        self.current_episode_step = 0

        # config success measure
        assert success_measure in ['stay', 'see']
        self.success_measure = success_measure
        print('[RoomNavTask] >> Success Measure = <{}>'.format(success_measure))
        self.success_stay_cnt = 0
        if success_measure == 'see':
            self.room_target_object = dict(outdoor=[])  # outdoor is a special category
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
                self.room_target_object[room].append(c)

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
            _id = self.house._id
            if self.house.targetRoomTp not in self._availCoorsSizeDict[_id]:
                if (self.hardness is None) and (self.max_birthplace_steps is None):
                    self.availCoorsSize = self.house.getConnectedLocationSize()
                else:
                    allowed_dist = self.house.maxConnDist
                    if self.hardness is not None:
                        allowed_dist = min(int(self.house.maxConnDist * self.hardness + 1e-10), allowed_dist)
                    if self.max_birthplace_steps is not None:
                        allowed_dist = min(self.house.getAllowedGridDist(self.max_birthplace_steps * self.move_sensitivity), allowed_dist)
                    self.availCoorsSize = self.house.getConnectedLocationSize(max_allowed_dist=allowed_dist)
                self._availCoorsSizeDict[_id][self.house.targetRoomTp] = self.availCoorsSize
            else:
                self.availCoorsSize = self._availCoorsSizeDict[_id][self.house.targetRoomTp]

    @property
    def house(self):
        return self.env.house

    """
    gym api: reset function
    when target is not None, we will set the target room type to navigate to
    """
    def reset(self, target=None):
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

    def _apply_action(self, action):
        if self.discrete_action:
            return discrete_actions[action]
        else:
            rot = action[1][0] - action[1][1]
            act = action[0]
            return (act[0] - act[1]), (act[2] - act[3]), rot

    def _is_success(self, raw_dist):
        if raw_dist > 0:
            self.success_stay_cnt = 0
            return False
        if self.success_measure == 'stay':
            self.success_stay_cnt += 1
            return self.success_stay_cnt >= success_stay_time_steps
        # self.success_measure == 'see'
        object_color_list = self.room_target_object[self.house.targetRoomTp]
        flag_see_target_objects = (len(object_color_list) == 0)
        if (self.last_obs is not None) and self.segment_input:
            seg_obs = self.last_obs if not self.joint_visual_signal else self.last_obs[:,:,3:6]
        else:
            seg_obs = self.env.render(mode='semantic')
        self._object_cnt = 0
        for c in object_color_list:
            cur_n = np.sum(np.all(seg_obs == c, axis=2))
            self._object_cnt += cur_n
            if self._object_cnt >= n_pixel_for_object_see:
                flag_see_target_objects = True
                break
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
            if flag_print_debug_info:
                print('Move Successfully!')

        self.last_obs = obs = self.env.render()
        if self.joint_visual_signal:
            self.last_obs = obs = np.concatenate([self.env.render(mode='rgb'), obs], axis=-1)
        cur_info = self.info
        raw_dist = cur_info['dist']
        orig_raw_dist = self.last_info['dist']
        raw_scaled_dist = cur_info['scaled_dist']
        dist = raw_scaled_dist * self.dist_scale
        done = False
        if self._is_success(raw_dist):
            reward += self.successRew
            done = True
        # accumulate episode step
        self.current_episode_step += 1
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
                if raw_dist >= orig_raw_dist: reward -= time_penalty_reward
            elif self.reward_type == 'delta':
                delta_raw_dist = orig_raw_dist - raw_dist
                ratio = self.move_sensitivity / self.house.grid_det
                delta_reward = delta_raw_dist / ratio * delta_reward_coef
                delta_reward = np.clip(delta_reward, -indicator_reward, indicator_reward)
                reward += delta_reward
                if raw_dist >= orig_raw_dist: reward -= time_penalty_reward
            elif self.reward_type == 'speed':
                movement = np.sqrt((self.last_info['pos'][0]-cur_info['pos'][0])**2
                                   + (self.last_info['pos'][1]-cur_info['pos'][1])**2)
                sign = np.sign(orig_raw_dist - raw_dist)
                det_dist = movement * sign * speed_reward_coef
                det_dist = np.clip(det_dist, -indicator_reward, indicator_reward)
                reward += det_dist
                if raw_dist >= orig_raw_dist: reward -= time_penalty_reward
            elif self.reward_type == 'new':
                # utilize delta reward but with different parameters
                delta_raw_dist = orig_raw_dist - raw_dist
                ratio = self.move_sensitivity / self.house.grid_det
                new_reward = delta_raw_dist / ratio * new_reward_coef
                new_reward = np.clip(new_reward, -new_reward_bound, new_reward_bound)
                reward += new_reward
                if raw_dist >= orig_raw_dist: reward -= time_penalty_reward   # always deduct time penalty
                if (orig_raw_dist == 0) and (raw_dist > 0): reward -= new_leave_penalty  # big penalty when leave target room

        # object seen reward
        if self.current_episode_step > self.reward_silence:
            if (raw_dist == 0) and (self.success_measure == 'see'):  # inside target room and success measure is <see>
                if not done:
                    object_reward = np.clip((self._object_cnt - n_pixel_for_object_sense) / L_pixel_reward_range, 0., 1.) * self.pixelRew
                    reward += object_reward

        if self.depth_signal:
            dep_sig = self.env.render(mode='depth')
            if dep_sig.shape[-1] > 1:
                dep_sig = dep_sig[..., 0:1]
            obs = np.concatenate([obs, dep_sig], axis=-1)
        self.last_info = cur_info
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
        ret['optsteps'] = int(dist / (self.move_sensitivity / self.house.grid_det) + 0.5)
        ret['collision'] = int(self.collision_flag)
        ret['target_room'] = self.house.targetRoomTp
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
    """
    def reset_hardness(self, hardness=None, max_birthplace_steps=None):
        self.hardness = hardness
        self.max_birthplace_steps = max_birthplace_steps
        if (hardness is None) and (max_birthplace_steps is None):
            self.availCoorsSize = self.house.getConnectedLocationSize()
        else:
            allowed_dist = self.house.maxConnDist
            if hardness is not None:
                allowed_dist = min(int(self.house.maxConnDist * hardness + 1e-10), allowed_dist)
            if max_birthplace_steps is not None:
                allowed_dist = min(self.house.getAllowedGridDist(max_birthplace_steps * self.move_sensitivity), allowed_dist)
            self.availCoorsSize = self.house.getConnectedLocationSize(max_allowed_dist=allowed_dist)
        n_house = self.env.num_house
        self._availCoorsSizeDict = [dict() for _ in range(n_house)]
        self._availCoorsSizeDict[self.house._id][self.house.targetRoomTp] = self.availCoorsSize

    """
    recover the state (location) of the agent from the info dictionary
    """
    def set_state(self, info):
        self.env.reset(x=info['pos'][0], y=info['pos'][1], yaw=info['yaw'])

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
