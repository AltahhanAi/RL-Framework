'''
   Author: Abdulrahman Altahhan,  2025.
   version: 3.5

    This library of functionality in RL that aims for simplicity and general
    insight into how algorithms work, these libraries are written from scratch
    using standard Python libraries (numpy, matplotlib etc.).
    Please note that you will need permission from the author to use the code
    for research, commercially or otherwise.
'''

import rclpy as ros
from rclpy.node import Node

from geometry_msgs.msg import Twist
from nav_msgs.msg  import Odometry
from sensor_msgs.msg import LaserScan
from std_srvs.srv import Empty
from gazebo_msgs.srv import SpawnModel
from gazebo_msgs.msg import ModelStates

import numpy as np
from random import randint
from math import atan2, atan, pi, cos, hypot
import matplotlib.pyplot as plt

from rl.rlln import *
import xml.etree.ElementTree as ET
import subprocess
import time
# ====================================================================================================

# if not using foxy replace the default path by a suitable one
sdf_path = "/opt/ros/foxy/share/turtlebot3_gazebo/models/turtlebot3_burger/model.sdf"
model_path = "/opt/ros/foxy/share/turtlebot3_gazebo/worlds/turtlebot3_assessment2/burger.model"


def name(): return 'node'+str(randint(1, 1000))


def load_gazebo():
    subprocess.Popen(["ros2", "launch", "turtlebot3_gazebo", "turtlebot3_assessment2.launch.py"])
    print("ROS2 launch file has completed.")


def get_nscans_LiDAR(path=sdf_path):
    return int(ET.parse(path).find(".//sensor[@type='ray']//horizontal/samples").text)


def set_nscans_LiDAR(nscans, path=sdf_path):
    # Parse the .sdf file
    tree = ET.parse(path)
    root = tree.getroot()

    # Find the ray sensor and update the samples
    root.find(".//sensor[@type='ray']//horizontal/samples").text = str(nscans)

    # Write the modified tree back to the file
    tree.write(path)

def kill_sim_processes():
    processes = ['gzclient', 'gzserver', 'ros2']
    for proc in processes:
        try:
            subprocess.run(['killall', '-9', proc], check=True)
            print(f'Killed {proc}')
        except subprocess.CalledProcessError:
            print(f'No running process found for {proc}')


def set_real_time_rate(time_rate=1000, step_rate=0.001, path=model_path):
    # this function set the rate for the simulater for acceleration
    # usually step_rate = 1/time_rate, but you can set them to be different
    tree = ET.parse(path)
    root = tree.getroot()
    root.find('.//real_time_update_rate').text = str(time_rate)
    root.find('.//max_step_size').text = str(step_rate)
    tree.write(path, encoding="utf-8", xml_declaration=True)


def accelerate_sim(speed):
    set_real_time_rate(step_rate=0.001*speed)
# ====================================================================================================


class RobEnv(Node):
    # initialisation--------------------------------------------------------------
    # n sets hw=ow frequent the publisher is publishing to the /cmd_vel
    def __init__(self, name=name(),
                 n=8, speed=2.0, θspeed=pi/2,
                 nscans=get_nscans_LiDAR(), nscansρ=.15,
                 tol=.9,
                 sleep=False,
                 verbose=False):
        super().__init__(name)

        self.n = n
        self.speed = speed
        self.sleep = sleep
         
        self.θspeed = round(θspeed, 2)
        # percentage of laser scans used to specify if there is an obstical in front of the robot
        self.nscansρ = nscansρ 

        self.robot = Twist()

        self.verbose = verbose

        # do not change----------------------------------------------------
        self.x = 0   # initial x position
        self.y = 0   # initial y position
        self.θ = 0   # initial θ angle

        # gets how many laser beams burger is using in its LiDAR sensor
        self.scans = np.zeros(nscans)
        self.t = 0

        self.tol = tol  # meter from goal as per the requirement (tolerance)
        self.goals = [[2.0, 2.0], [-2.0, -2.0]]
        # -----------------------------------------------------------------

        self.controller = self.create_publisher(Twist, '/cmd_vel', 1)

        self.scanner = self.create_subscription(LaserScan, '/scan', self.scan, 1)
        self.odometr = self.create_subscription(Odometry,  '/odom', self.odom, 1)

        self.max_range = 3.5
        self.min_range = .35

        # establish a reset client
        self.reset_world = self.create_client(Empty, '/reset_world')
        while not self.reset_world.wait_for_service(timeout_sec=4.0):
            print('world client service...')

        # compatibility----------------------------------------------
        nturns = 16         # number of turns robot takes to complete a full circle
        resol = speed/2
        
        θresol = 2*pi/nturns
        dims = [4, 4]
        self.xdim = dims[0]  # realted to the size of the environment
        self.ydim = dims[1]  # realted to the size of the environment
        
        self.resol = round(resol, 2)
        self.θresol = round(θresol, 2)
        
        self.cols = int(self.xdim//self.resol) + 1   # number of grid columns, related to linear speed
        self.rows = int(self.ydim//self.resol) + 1   # number of grid rows,    related to linear speed
        self.orts = int(2*pi//self.θresol)     + 1   # number of angles,       related to angular speed
        
        self.goal_dist = self.distgoal(self.goals[0])
        self.θgoal_dist = self.θdistgoal(self.goals[0])

        self.nC = self.rows*self.cols              # Grid size
        self.nS = self.rows*self.cols*self.orts    # State space size
        self.nA = 3

        self.Vstar = None        # for compatibility
        self.figsize0 = (12, 2)  # for compatibility
        # --------------------------------------------------------------- 
        # self.rate = self.create_rate(30)
        self.reset()

        print('speed  = ', self.speed)
        print('θspeed = ', self.θspeed)

# Convert a quaternion to an Euler yaw angle ([0, 2π])----
    def yaw(self, orient):
        # orient: A quaternion object with attributes x, y, z, w.
        # yaw: angle in radians within the range [0, 2π].
        x, y, z, w = orient.x, orient.y, orient.z, orient.w
        yaw = atan2(2.0*(x*y + w*z), w*w + x*x - y*y - z*z)
        return yaw if yaw>0 else yaw + 2*pi # in radians, [0, 2pi]
    
# ---------------------------------------------low level sensing---------------------------------------------   
    # odometry (position and orientation) readings
    def odom(self, odoms):
        self.x = round(odoms.pose.pose.position.x, 1)
        self.y = round(odoms.pose.pose.position.y, 1)
        self.θ = round(self.yaw(odoms.pose.pose.orientation),2) 
        self.odom = np.array([self.x, self.y, self.θ])
        # if self.verbose: print('odom = ',  self.odom )

    # laser scanners readings
    def scan(self, scans):
        self.scans = np.array(scans.ranges)
        self.scans = np.clip(self.scans, self.min_range, self.max_range)
        self.scans[np.isnan(self.scans)] = 0
        # if self.verbose: print('scan = ', self.scans[:10].round(2))
        # if self.verbose: print('scan = ', np.r_[self.scans[-5:], self.scans[:5]].round(2))

# ---------------------------------------------low level control---------------------------------------------   
    def spin_n(self, n):
        for _ in range(n):
            self.controller.publish(self.robot)
            ros.spin_once(self)
            if self.sleep: time.sleep(1.0 / 30)

    # move then stop to get a defined action
    def step(self, a=1, speed=None, θspeed=None):
        if speed is None: speed = self.speed
        if θspeed is None: θspeed = self.θspeed

        self.t += 1
        # if self.verbose: print('step = ', self.t)

        if   a ==-1: self.robot.linear.x = -speed    # backwards not used
        elif a == 1: self.robot.linear.x = speed     # forwards
        elif a == 0: self.robot.angular.z = θspeed   # turn left
        elif a == 2: self.robot.angular.z = -θspeed  # turn right
        
        # try:
        # Now move and stop so that we can have a well defined actions  
        self.spin_n(self.n if a==1 else self.n-1)
        self.stop()
        # except KeyboardInterrupt:
        #     print("Execution interrupted by user. Cleaning up...")

        reward, done = self.reward(a)
        return self.s_(), reward, done, {}

    def stop(self):
        self.robot.linear.x = .0
        self.robot.angular.z = .0
        self.spin_n(self.n-1)
        if self.sleep: time.sleep(1.0 / 30)

    # reseting--------------------------------------------------------------
    def reset(self):
        # if self.verbose: print('resetting world..........................................')
        for _ in range(2):
            # to ensure earlier queued actions are flushed, there are better ways to do this
            self.reset_world.call_async(Empty.Request())
            # do not add a line to move forward one step as it will upset the reward logic

            future = self.reset_world.call_async(Empty.Request())
            ros.spin_until_future_complete(self, future, timeout_sec=1.0)

            if not future.done():
                print("Reset not completed within timeout.")

        return self.s_()

    # for compatibility, do not delete
    def render(self, **kw):
        pass

# -----------------------------------learning related (reward and state)---------------------------------------------   

    # robot reached goal ...............................................................
    def atgoal(self, goal_dist):
        if goal_dist <= self.tol:
            print('Goal has been reached woohoooooooooooooooooooooooooooooo!!')
            return True
        return False

    # robot hits a wall...................................................................
    def atwall(self):
        # check only 2*rng front scans for collision, given the robot does not move backward
        rng = int(len(self.scans)*self.nscansρ//2)  # nscansρ//2 left and nscansρ//2 right 
        return np.r_[self.scans[-rng:], self.scans[:rng]].min() <= self.min_range 
        # return self.scans.min() <= self.min_range

    # angular distance of robot to a specific goal......................................
    def θdistgoal(self, goal):
        xgoal, ygoal = goal
        θgoal = atan2(ygoal - self.y, xgoal - self.x)
        θgoal = θgoal if θgoal >= 0 else θgoal + 2 * pi  # map to [0, 2π]

        θdiff = abs(θgoal - self.θ)
        return round(min(θdiff, 2*pi - θdiff), 2)

    # Eucleadian distance of robot to a goal.............................................
    def distgoal(self, goal):
        xgoal, ygoal = goal
        return round(hypot(self.x - xgoal, self.y - ygoal), 2)
        # return round(((self.x - xgoal)**2 + (self.y - ygoal)**2)**.5, 2)

    # generic: returns the goal with the nearest orientaiton or nearest distance........
    def nearest_(self, compare='distgoal'):
        '''
        passing 'θdistgoal' returns the angular distance of the goal with the nearest orientation
        passing 'distgoal' returns the eucleadian distance of the goal with the nearest distance
        '''
        # obtain the distance or angular distance to goals (depending on func)
        dists = [getattr(self, compare)(goal) for goal in self.goals]
        # return the goal robot is heading towards or nearest to
        goal_idx, goal_dist = min(enumerate(dists), key=lambda x: x[1])
        return goal_dist, self.goals[goal_idx]

    # returns the eucleadian distance to the nearest goal
    def nearest(self):
        goal_dist, goal = self.nearest_('distgoal')
        θgoal_dist      = self.θdistgoal(goal)
        return goal_dist, θgoal_dist  #, goal

    # returns angular distance to the goal in sight
    def θnearest(self):
        θgoal_dist, goal = self.nearest_('θdistgoal')
        goal_dist        = self.distgoal(goal)
        return goal_dist, θgoal_dist  #, goal

    # calculates distances to nearest goal and to walls..................................
    def goal_seeking(self):

        # store last step distance to goal
        old_goal_dist = self.goal_dist
        old_θgoal_dist = self.θgoal_dist

        self.goal_dist, self.θgoal_dist = self.nearest()
        # self.goal_dist, self.θgoal_dist = self.θnearest()

        self.Δgoal_dist = old_goal_dist - self.goal_dist
        self.Δθgoal_dist = old_θgoal_dist - self.θgoal_dist

        self.at_wall = self.atwall()
        self.at_goal = self.atgoal(self.goal_dist)  # at_goal is the same as done

        # reset without restarting an episode if the robot hits a wall and not atgoal
        if self.at_wall and not self.at_goal: self.reset()

        return self.at_goal

    # set reward based on agent's status to produce a stabel policy...................
    # do not override this function
    def reward(self, a):
        # keep the order as is to benefit from distances calculation
        done = self.goal_seeking()
        reward = self.reward_(a)
        return reward, done

# ---------------------------------------------override---------------------------------------------   

    # override this function, 
    # you may use goal_dist, Δgoal_dist, θgoal_dist, Δθgoal_dist or at_wall and at_goal
    def reward_(self, a):
        return -1

    # simple state representation similar to grid world, it returns an index. 
    # This is not suitable for assignment **override with a suitable state representation**
    def s_(self):

        self.xi = int((self.x+self.xdim/2)//self.resol)     # x index = col, assuming the grid middle is (0,0)
        self.yi = int((self.y+self.ydim/2)//self.resol)     # y index = row, assuming the grid middle is (0,0)

        # pi/2 to be superficially resilient to slight angle variation to keep θi unchanged
        self.θi = int((self.θ+pi/2)%(2*pi)//self.θresol)

        self.si = self.xi + self.yi*self.cols     # position state in the grid
        self.s = self.nC*(self.θi) + self.si      # position state with orientation
        # if self.verbose: print('grid cell= ', self.si, 'state = ', self.s)

        return self.s
