#!/usr/bin/env python3
# Author: Sawas
import random
import sys
sys.path.insert(0, '..')
import os
import time
import math
import  cv2 as cv
import numpy as np
import mujoco_py
from gym.envs.mujoco import mujoco_env
from gym import utils, spaces
from gym_ur5.controller.MujocoController import MJ_Controller
import gym
from gym import spaces
import time
from termcolor import colored

class UR5Env(gym.Env):
    def __init__(self):
        super(UR5Env, self).__init__()
        self.score = 0
        self.faild = 0
        self.start_time = 0
        self.end_time = 0
        self.td = 0
        self.total_time = 0

        self.depth_axis,self.elvation_dis,self.lateral_dist= 0,0,0

        self.f_t =[0,0,0,0,0,0]
        self.privous_coord = [0, 0, 0]
        self.init = [0.005, -0.438, 1.112]
        self.coord = [0.005, -0.438, 1.112]
        self.target = [0.0, -0.46, 1.115]
        self.object_index=1


        self.rotations = {0: 0, 1: 30, 2: 60, 3: 90, 4: -30, 5: -60}
        self.controller = MJ_Controller()
        self.done = False
        self.degree = 0
        self.target_reached = 3
        self.correction_value = 0.025
        self.reward = 0
        self.epoch = 0
        self.wait = 5
        self.step_size = 0.001
        self.controller.show_model_info()
        self.action_space = spaces.Discrete(17)
        self.observation_space = spaces.Box(low=-100, high=100, shape=(6,), dtype=np.float32)

    def calculate_euclidean_distances(self):

        
        self.dist_T_O = [np.linalg.norm(self.target[i]-self.privous_coord[i]) for i in range(3)]
        self.dist_T_N = [np.linalg.norm(self.target[i]-self.coord[i]) for i in range(3)]
        self.lateral_dist = np.linalg.norm(self.target[self.geoAxes[0]]-self.coord[self.geoAxes[0]])
        self.depth_axis = self.coord[self.geoAxes[1]]
        self.elvation_dis = np.linalg.norm(self.target[self.geoAxes[2]]-self.coord[self.geoAxes[2]])
        
    def step(self, action):

        self.reward = 0
        self.privous_coord = self.coord.copy()
        self.move(action)
        self.calculate_euclidean_distances()
        self.controller.stay(self.wait)
        self.f_t = self.controller.get_ft()
        self.check_collision()
        self.check_approching_target()
        self.check_target()
        
        
        state = [0,0,0,0,0,0]        
        state =  self.f_t 
        self.reward = round(self.reward, 2)

        
         
        if(sum(self.dist_T_N) < 0.022):
            self.step_size = 0.0005
            self.correction_value = 0.013
        else:
            self.step_size = 0.001
            self.correction_value = 0.025

        return np.array(state, dtype=np.float32), self.reward, self.done, {}
    
    def reset(self):
        
        #print("*"*30)
        self.done = False
        self.reward = 0
        self.epoch = 0
        self.start_time = time.time()

        self.geoAxes=[0,1,2]
        self.depth_target= 0.452
        self.controller.change_object_palace(2, self.object_index, 0.95,"platt"+str(self.object_index)) 
        self.object_index = random.choice([1,2])
        self.object_index = 5
        self.target[0]=random.uniform(-0.1,0.1)
        self.init[0] = random.uniform(self.target[0] - 0.006, self.target[0] + 0.006)
        self.init[1] = random.uniform(-0.436, -0.433)
        self.init[2] = random.uniform(1.1, 1.12)
        self.controller.change_object_shape(self.object_index)
        self.controller.change_object_palace(self.target[0], -.75, 0.95,"platt"+str(self.object_index))
        self.coord = self.init.copy()

        self.degree = ((self.coord[0]+0.0001)/0.0001) * 0.01348
        
        self.controller.move_ee([0.005, -0.432, 1.11], marker=True)
        self.controller.stay(self.wait)
        self.controller.move_ee([self.coord[0],self.coord[1],self.coord[2]], marker=True)
        #self.controller.stay(100)
        self.f_t = self.controller.get_ft()
        
        return np.array( self.f_t, dtype=np.float32)
    
    def check_collision(self):
        collision = sum([True  if self.f_t[i] > 100 \
             else True if self.f_t[i] < -100 \
                 else False for i in range(6)])
        if collision > 0 :
            self.done= True
            self.faild += 1
            self.reward -= 20
            print(colored("****** COLLISION ******", color='red', attrs=['bold']))
            print(colored("Success rate = "+ str(self.score)+"/"+str(self.score+self.faild), color='yellow', attrs=['bold']))
        
        self.reward += sum([-1  if self.f_t[i] > 30 \
             else -1 if self.f_t[i] < -30 \
                 else 1 for i in range(3)])
        self.reward += sum ([1  if self.f_t[i] < 30  and  self.f_t[i] > -30 \
            and sum(self.dist_T_N) <   0.01 and sum(self.dist_T_N) < sum(self.dist_T_O) \
            else -1 for i in range(3)])
       
    def check_insertion(self):
        if self.elvation_dis > 0.02  or self.lateral_dist  > 0.02:
            self.done = True
            self.faild += 1
            self.reward -= 100
            print(colored("****** Went out of Boundaries ******", color='red', attrs=['bold']))
            print(colored("Success Socre = "+ str(self.score)+"/"+str(self.score+self.faild), color='yellow', attrs=['bold']))

        if np.abs(self.depth_axis) >= np.abs(self.depth_target) and self.elvation_dis < 0.02  and self.lateral_dist  < 0.02 :
            self.done = True
            self.reward += 100 
            self.target_reached += 1
            self.score += 1
            self.end_time = time.time()
            self.td= self.end_time -self.start_time
            self.total_time = self.total_time+ self.td
            print(colored("Success Socre = "+ str(self.score)+"/"+str(self.score+self.faild), color='yellow', attrs=['bold']))
            print(colored("Avarage Time = "+str(self.total_time/self.score), color='red', attrs=['bold']))
            print(colored("****** TASK ACCOMPLISHED ******", color='green', attrs=['bold']))

    def Check_EF_approaching_target(self):
        
        self.reward += sum([ 2 if  self.dist_T_N[i] < self.dist_T_O[i] \
            else 0 if  self.dist_T_N[i] == self.dist_T_O[i] else -2 for i in range(3)])

        self.reward += (-2 if np.abs(self.coord[self.geoAxes[1]]) < np.abs(self.privous_coord[self.geoAxes[1]]) and sum(self.dist_T_N) <= 0.01 \
                         else 4 if sum(self.dist_T_N) <= 0.01 else 0 )


    def move(self, action):
        
        
        if (action == 0):#Movingonxaxis X
            self.coord[0] += self.step_size
            self.degree += self.correction_value

        elif (action == 1):#Movingonx axis X
             self.coord[0] -= self.step_size
             self.degree -= self.correction_value
 
        #elif (action == 2):##Movingony axis Y
         #    self.coord[1] += self.step_size
 
        elif (action == 2):##Movingony axis Y
            self.coord[1] -= self.step_size
 
        elif (action == 3):##Movingonz axis Z
             self.coord[2] += self.step_size
 
        elif (action == 4):##Movingonz axis Z
             self.coord[2] -= self.step_size
 
        #elif (action == 6):##Movingonz axis XY
         #   self.coord[0] += self.step_size
          #  self.coord[1] += self.step_size
          #  self.degree += self.correction_value

        elif (action == 5):##Movingonz axis XY
            self.coord[0] -= self.step_size
            self.coord[1] -= self.step_size
            self.degree -= self.correction_value
     
        #elif (action == 8):##Movingonz axis XY
           # self.coord[0] -= self.step_size  
           # self.coord[1] += self.step_size
            #self.degree -= self.correction_value
         
        elif (action == 6):##Movingonz axis XY
            self.coord[0] += self.step_size  
            self.coord[1] -= self.step_size
            self.degree += self.correction_value
         
        elif (action == 7):##Movingonz axis XZ
            self.coord[0] += self.step_size
            self.coord[2] += self.step_size
            self.degree += self.correction_value
         
        elif (action == 8):##Movingonz axis XZ
            self.coord[0] -= self.step_size
            self.coord[2] -= self.step_size
            self.degree -= self.correction_value
 
        elif (action == 9):##Movingonz axis XZ
            self.coord[0] += self.step_size  
            self.coord[2] -= self.step_size
            self.degree += self.correction_value
         
        elif (action == 10):##Movingonz axis XZ
            self.coord[0] -= self.step_size
            self.coord[2] += self.step_size
            self.degree -= self.correction_value
         
        #elif (action == 14):##Movingonz axis YZ
           # self.coord[1] += self.step_size
           # self.coord[2] += self.step_size
         
        elif (action == 11):##Movingonz axis YZ
            self.coord[1] -= self.step_size
            self.coord[2] -= self.step_size
         
        #elif action == 16 :##Movingonz axis YZ
          #  self.coord[1] += self.step_size
          #  self.coord[2] -= self.step_size
         
        elif action == 12 :##Movingonz axis YZ
            self.coord[1] -= self.step_size
            self.coord[2] += self.step_size
         
        #elif action == 15:##Movingonz axis XYZ
          #  self.coord[0] += self.step_size
          #  self.coord[1] += self.step_size
          #  self.coord[2] += self.step_size
           # self.degree += self.correction_value
             
        #elif action == 9 :##Movingonz axis XYZ
         #   self.coord[0] += self.step_size
          #  self.coord[1] += self.step_size
         #   self.coord[2] -= self.step_size
           # self.degree += self.correction_value
 
        elif action == 13 :##Movingonz axis XYZ
            self.coord[0] += self.step_size
            self.coord[1] -= self.step_size
            self.coord[2] -= self.step_size
            self.degree += self.correction_value
         
        #elif action == 7 :##Movingonz axis XYZ
        #    self.coord[0] -= self.step_size
        #    self.coord[1] += self.step_size
        #    self.coord[2] += self.step_size
        #    self.degree -= self.correction_value
 
        elif action == 14 :##Movingonz axis XYZ
            self.coord[0] -= self.step_size
            self.coord[1] -= self.step_size
            self.coord[2] += self.step_size
            self.degree -= self.correction_value
         
        elif action == 15 :##Movingonz axis XYZ
            self.coord[0] -= self.step_size
            self.coord[1] -= self.step_size
            self.coord[2] -= self.step_size
            self.degree -= self.correction_value

        #elif action == 3 :##Movingonz axis XYZ
         #   self.coord[0] -= self.step_size
         #   self.coord[1] += self.step_size
         #   self.coord[2] -= self.step_size
          #  self.degree -= self.correction_value
         
        elif action >= 16 :##Movingonz axis XYZ
           self.coord[0] += self.step_size
           self.coord[1] -= self.step_size
           self.coord[2] += self.step_size
           self.degree += self.correction_value

        #print(self.coord[0],self.coord[1],self.coord[2])


        self.controller.move_ee([self.coord[0], self.coord[1], self.coord[2]], marker=True)
        self.controller.current_target_joint_values[5] = math.radians(self.degree)
       
