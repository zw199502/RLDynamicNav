import logging
# import matplotlib.pyplot as plt
# import matplotlib.animation as ani
import numpy as np
# from matplotlib import collections as mc
from numpy.linalg import norm
from utils.human import Human
from utils.robot import Robot
from utils.state import *
from policy.policy_factory import policy_factory
from info import *
from math import atan2, hypot, sqrt, cos, sin, fabs, inf, ceil
from time import sleep, time
from C_library.motion_plan_lib import *

# laser scan parameters
# number of all laser beams
n_laser = 1800
laser_angle_resolute = 0.003490659
laser_min_range = 0.27
laser_max_range = 6.0

# environment size
square_width = 10.0
# environment size


class CrowdSim:
    def __init__(self):
        self.human_policy_name = 'orca' # human policy is fixed orca policy
        self.time_limit = None
        self.time_step = None
        self.robot = None
        self.humans = None
        self.global_time = None

        # reward function
        self.success_reward = None
        self.collision_penalty = None
        self.discomfort_dist = None
        self.discomfort_penalty_factor = None
        self.goal_distance_factor = None

        # last-time distance from the robot to the goal
        self.goal_distance_last = None

        self.case_capacity = None
        self.case_size = None
        self.case_counter = None
        self.randomize_attributes = None
        self.lines = None # environment margin
        self.circles = None # human margin
        self.circle_radius = None
        self.human_num = None

        # scan_intersection, each line connects the robot and the end of each laser beam
        self.scan_intersection = None # used for visualization

        # laser state
        self.scan_current = np.zeros(n_laser, dtype=np.float32)
        self.scan_last_1 = np.zeros(n_laser, dtype=np.float32)
        self.scan_last_2 = np.zeros(n_laser, dtype=np.float32)
        self.scan_last_3 = np.zeros(n_laser, dtype=np.float32)
        self.scan_last_4 = np.zeros(n_laser, dtype=np.float32)
        self.scan_last_5 = np.zeros(n_laser, dtype=np.float32)
        self.scan_end_current = np.zeros((n_laser, 2), dtype=np.float32)
        self.scan_end_last_1 = np.zeros((n_laser, 2), dtype=np.float32)
        self.scan_end_last_2 = np.zeros((n_laser, 2), dtype=np.float32)
        self.scan_end_last_3 = np.zeros((n_laser, 2), dtype=np.float32)
        self.scan_end_last_4 = np.zeros((n_laser, 2), dtype=np.float32)
        self.scan_end_last_5 = np.zeros((n_laser, 2), dtype=np.float32)

        # plt.ion()
        # plt.show()
        # self.fig, self.ax = plt.subplots(figsize=(10, 10))
        self.human1_position = []
        self.human2_position = []
        self.human3_position = []
        self.human4_position = []
        self.human5_position = []
        self.human6_position = []
        self.human7_position = []
        self.human8_position = []
        self.robot_position = []
        self.count = 0
        
    def configure(self, reward_simple=False):
        self.reward_simple = reward_simple
        self.time_limit = 25.0
        self.time_step = 0.25
        self.randomize_attributes = False
        self.success_reward = 1.0
        self.collision_penalty = -1.0
        self.discomfort_dist = 0.2
        self.discomfort_penalty_factor = 0.5
        self.goal_distance_factor = 0.01
       
        self.case_capacity = {'train': np.iinfo(np.uint32).max - 2000, 'val': 1000, 'test': 1000}
        self.case_size = {'train': np.iinfo(np.uint32).max - 2000, 'val': 100, 'test': 500}

        # margin = square_width / 2.0
        margin = 35.0  
        # here, more lines can be added to simulate obstacles
        self.lines = [[(-margin, -margin), (-margin,  margin)], \
                        [(-margin,  margin), ( margin,  margin)], \
                        [( margin,  margin), ( margin, -margin)], \
                        [( margin, -margin), (-margin, -margin)]]
        self.circle_radius = 4.0
        self.human_num = 5

        self.robot = Robot()
        self.robot.time_step = self.time_step

        self.case_counter = {'train': 0, 'test': 0, 'val': 0}

        logging.info('human number: {}'.format(self.human_num))
        if self.randomize_attributes:
            logging.info("Randomize human's radius and preferred speed")
        else:
            logging.info("Not randomize human's radius and preferred speed")
        logging.info('Square width: {}, circle width: {}'.format(square_width, self.circle_radius))
        
    def generate_random_human_position(self):
        # initial min separation distance to avoid danger penalty at beginning
        self.humans = []
        for i in range(self.human_num):
            self.humans.append(self.generate_circle_crossing_human())

        for i in range(len(self.humans)):
            human_policy = policy_factory[self.human_policy_name]()
            human_policy.time_step = self.time_step
            self.humans[i].set_policy(human_policy)

    def generate_circle_crossing_human(self):
        human = Human()
        human.time_step = self.time_step

        if self.randomize_attributes:
            # Sample agent radius and v_pref attribute from certain distribution
            human.sample_random_attributes()
        else:
            human.radius = 0.3
            human.v_pref = 1.0
        while True:
            angle = np.random.random() * np.pi * 2
            # add some noise to simulate all the possible cases robot could meet with human
            px_noise = (np.random.random() - 0.5) * human.v_pref
            py_noise = (np.random.random() - 0.5) * human.v_pref
            px = self.circle_radius * np.cos(angle) + px_noise
            py = self.circle_radius * np.sin(angle) + py_noise
            collide = False
            for agent in [self.robot] + self.humans:
                min_dist = human.radius + agent.radius + self.discomfort_dist
                if norm((px - agent.px, py - agent.py)) < min_dist or \
                        norm((px - agent.gx, py - agent.gy)) < min_dist:
                    collide = True
                    break
            if not collide:
                break
        # px, py, gx, gy, vx, vy, theta
        human.set(px, py, -px, -py, 0, 0, 0)
        return human

    def get_lidar(self, isReset=False):
        scan = np.zeros(n_laser, dtype=np.float32)
        scan_end = np.zeros((n_laser, 2), dtype=np.float32)
        self.circles = np.zeros((self.human_num, 3), dtype=np.float32)
        # here, more circles can be added to simulate obstacles
        for i in range(self.human_num):
            self.circles[i, :] = np.array([self.humans[i].px, self.humans[i].py, self.humans[i].radius])
        robot_pose = np.array([self.robot.px, self.robot.py, self.robot.theta])
        num_line = len(self.lines)
        num_circle = self.human_num
        InitializeEnv(num_line, num_circle, n_laser, laser_angle_resolute)
        for i in range (num_line):
            set_lines(4 * i    , self.lines[i][0][0])
            set_lines(4 * i + 1, self.lines[i][0][1])
            set_lines(4 * i + 2, self.lines[i][1][0])
            set_lines(4 * i + 3, self.lines[i][1][1])
        for i in range (num_circle):
            set_circles(3 * i    , self.humans[i].px)
            set_circles(3 * i + 1, self.humans[i].py)
            set_circles(3 * i + 2, self.humans[i].radius)
        set_robot_pose(robot_pose[0], robot_pose[1], robot_pose[2])
        # t1 = time()
        cal_laser()
        # t2 = time()
        # print(t2 - t1)
        self.scan_intersection = []
        for i in range(n_laser):
            scan[i] = get_scan(i)
            scan_end[i, :] = np.array([get_scan_line(4 * i + 2), get_scan_line(4 * i + 3)])
            ### used for visualization
            self.scan_intersection.append([(get_scan_line(4 * i + 0), get_scan_line(4 * i + 1)), \
                                           (get_scan_line(4 * i + 2), get_scan_line(4 * i + 3))])
            ### used for visualization
        
        #### proposed method ####
        if isReset:
            self.scan_current = np.clip(scan, laser_min_range, laser_max_range) / laser_max_range
            self.scan_end_current = np.copy(scan_end)
            self.scan_last_1 = np.copy(self.scan_current)
            self.scan_last_2 = np.copy(self.scan_current)
            self.scan_last_3 = np.copy(self.scan_current)
            self.scan_last_4 = np.copy(self.scan_current)
            # self.scan_last_5 = np.copy(self.scan_current)
            self.scan_end_last_1 = np.copy(self.scan_end_current)
            self.scan_end_last_2 = np.copy(self.scan_end_current)
            self.scan_end_last_3 = np.copy(self.scan_end_current)
            self.scan_end_last_4 = np.copy(self.scan_end_current)
            # self.scan_end_last_5 = np.copy(self.scan_end_current)
        else:
            for i in range(n_laser):
                set_scan_end_last(self.scan_end_current[i, 0], self.scan_end_current[i, 1], i, 1)
                set_scan_end_last(self.scan_end_last_1[i, 0], self.scan_end_last_1[i, 1], i, 2)
                set_scan_end_last(self.scan_end_last_2[i, 0], self.scan_end_last_2[i, 1], i, 3)
                set_scan_end_last(self.scan_end_last_3[i, 0], self.scan_end_last_3[i, 1], i, 4)
                # set_scan_end_last(self.scan_end_last_4[i, 0], self.scan_end_last_4[i, 1], i, 5)
            # t1 = time()
            transform_scan_last()    
            # print('time: ', time() - t1)
            for i in range(n_laser):
                self.scan_last_1[i] = get_last_scan(i, 1)
                self.scan_last_2[i] = get_last_scan(i, 2)
                self.scan_last_3[i] = get_last_scan(i, 3)
                self.scan_last_4[i] = get_last_scan(i, 4)
                # self.scan_last_5[i] = get_last_scan(i, 5)
            # self.scan_last_5 = np.clip(self.scan_last_5, laser_min_range, laser_max_range) / laser_max_range
            self.scan_last_4 = np.clip(self.scan_last_4, laser_min_range, laser_max_range) / laser_max_range
            self.scan_last_3 = np.clip(self.scan_last_3, laser_min_range, laser_max_range) / laser_max_range
            self.scan_last_2 = np.clip(self.scan_last_2, laser_min_range, laser_max_range) / laser_max_range
            self.scan_last_1 = np.clip(self.scan_last_1, laser_min_range, laser_max_range) / laser_max_range            
            self.scan_current = np.clip(scan, laser_min_range, laser_max_range) / laser_max_range

            # self.scan_end_last_5 = np.copy(self.scan_end_last_4)
            self.scan_end_last_4 = np.copy(self.scan_end_last_3)
            self.scan_end_last_3 = np.copy(self.scan_end_last_2)
            self.scan_end_last_2 = np.copy(self.scan_end_last_1)
            self.scan_end_last_1 = np.copy(self.scan_end_current)
            self.scan_end_current = np.copy(scan_end)
        #### proposed method ####

        #### no center transformation ####
        # if isReset:
        #     self.scan_current = np.clip(scan, laser_min_range, laser_max_range) / laser_max_range
        #     self.scan_end_current = np.copy(scan_end)
        #     self.scan_last_1 = np.copy(self.scan_current)
        #     self.scan_last_2 = np.copy(self.scan_current)
        #     self.scan_last_3 = np.copy(self.scan_current)
        #     self.scan_last_4 = np.copy(self.scan_current)
        #     # self.scan_last_5 = np.copy(self.scan_current)
        # else:
        #     # self.scan_last_5 = self.scan_last_4
        #     self.scan_last_4 = self.scan_last_3
        #     self.scan_last_3 = self.scan_last_2
        #     self.scan_last_2 = self.scan_last_1
        #     self.scan_last_1 = self.scan_current            
        #     self.scan_current = np.clip(scan, laser_min_range, laser_max_range) / laser_max_range
        #### no center transformation ####

        ReleaseEnv()

    def reset(self, phase='test'):
        assert phase in ['train', 'val', 'test']
        self.global_time = 0

        counter_offset = {'train': self.case_capacity['val'] + self.case_capacity['test'],
                            'val': 0, 'test': self.case_capacity['val']}
        # px, py, gx, gy, vx, vy, theta
        self.robot.set(0, -self.circle_radius, 0, self.circle_radius, 0, 0, np.pi / 2)
        self.goal_distance_last = self.robot.get_goal_distance()
        
        if self.case_counter[phase] >= 0:
            # for every training/valuation/test, generate same initial human states
            # np.random.seed(counter_offset[phase] + self.case_counter[phase])
            self.generate_random_human_position()
    
            # case_counter is always between 0 and case_size[phase]
            self.case_counter[phase] = (self.case_counter[phase] + 1) % self.case_size[phase]
        else:
            raise NotImplementedError

        self.get_lidar(isReset=True)

        # get the observation
        ob_lidar = np.hstack((self.scan_current, self.scan_last_1, self.scan_last_2, self.scan_last_3, self.scan_last_4))
        dx = self.robot.gx - self.robot.px
        dy = self.robot.gy - self.robot.py
        theta = self.robot.theta
        y_rel = dy * cos(theta) - dx * sin(theta)
        x_rel = dy * sin(theta) + dx * cos(theta)
        r = hypot(x_rel, y_rel) / square_width
        t = atan2(y_rel, x_rel) / np.pi
        ob_position = np.array([r, t], dtype=np.float32)
        self_state = FullState(self.robot.px, self.robot.py, self.robot.vx, self.robot.vy, self.robot.radius, \
                               self.robot.gx, self.robot.gy, self.robot.v_pref, self.robot.theta)
        ob_state = [human.get_observable_state() for human in self.humans]
        ob_coordinate = JointState(self_state, ob_state)

        # self.human1_position.append([self.humans[0].px, self.humans[0].py])
        # self.human2_position.append([self.humans[1].px, self.humans[1].py])
        # self.human3_position.append([self.humans[2].px, self.humans[2].py])
        # self.human4_position.append([self.humans[3].px, self.humans[3].py])
        # self.human5_position.append([self.humans[4].px, self.humans[4].py])
        # self.human6_position.append([self.humans[5].px, self.humans[5].py])
        # self.human7_position.append([self.humans[6].px, self.humans[6].py])
        # self.human8_position.append([self.humans[7].px, self.humans[7].py])
        # self.robot_position.append([self.robot.px, self.robot.py])

        return ob_lidar, ob_position, ob_coordinate

    def step(self, action):
        human_actions = []
        for human in self.humans:
            # observation for humans is always coordinates
            ob = [other_human.get_observable_state() for other_human in self.humans if other_human != human]
            human_actions.append(human.act(ob))

        # uodate states
        robot_x, robot_y, robot_theta = self.robot.compute_pose(action)
        self.robot.update_states(robot_x, robot_y, robot_theta, action)
        for i, human_action in enumerate(human_actions):
            self.humans[i].update_states(human_action)

        # t1 = time()
        # get new laser scan and grid map
        self.get_lidar()  
        # t2 = time()
        # print(t2 - t1)
        self.global_time += self.time_step
        
        # if reaching goal
        goal_dist = hypot(robot_x - self.robot.gx, robot_y - self.robot.gy)
        reaching_goal = goal_dist < self.robot.radius

        # collision detection between humans
        for i in range(self.human_num):
            for j in range(i + 1, self.human_num):
                dx = self.humans[i].px - self.humans[j].px
                dy = self.humans[i].py - self.humans[j].py
                dist = (dx ** 2 + dy ** 2) ** (1 / 2) - self.humans[i].radius - self.humans[j].radius
                if dist < 0:
                    # detect collision but don't take humans' collision into account
                    logging.debug('Collision happens between humans in step()')

        # self.human1_position.append([self.humans[0].px, self.humans[0].py])
        # self.human2_position.append([self.humans[1].px, self.humans[1].py])
        # self.human3_position.append([self.humans[2].px, self.humans[2].py])
        # self.human4_position.append([self.humans[3].px, self.humans[3].py])
        # self.human5_position.append([self.humans[4].px, self.humans[4].py])
        # self.human6_position.append([self.humans[5].px, self.humans[5].py])
        # self.human7_position.append([self.humans[6].px, self.humans[6].py])
        # self.human8_position.append([self.humans[7].px, self.humans[7].py])
        # self.robot_position.append([self.robot.px, self.robot.py])

        # collision detection between the robot and humans
        collision = False
        dmin = (self.scan_current * laser_max_range).min()
        if dmin <= self.robot.radius:
            collision = True

        reward = 0
        if self.global_time >= self.time_limit - 1:
            reward = 0
            done = True
            info = Timeout()
        elif collision:
            reward = self.collision_penalty
            done = True
            info = Collision()
        elif ((dmin - self.robot.radius) < self.discomfort_dist):
            # penalize agent for getting too close 
            reward = (dmin - self.robot.radius - self.discomfort_dist) * self.discomfort_penalty_factor * self.time_step
            done = False
            info = Danger(dmin)
        else:
            reward = 0
            done = False
            info = Nothing()

        if reaching_goal:
            reward = reward + self.success_reward
            done = True
            info = ReachGoal()
            # np.savetxt('test_data/eight_h1.txt', np.array(self.human1_position))
            # np.savetxt('test_data/eight_h2.txt', np.array(self.human2_position))
            # np.savetxt('test_data/eight_h3.txt', np.array(self.human3_position))
            # np.savetxt('test_data/eight_h4.txt', np.array(self.human4_position))
            # np.savetxt('test_data/eight_h5.txt', np.array(self.human5_position))
            # np.savetxt('test_data/eight_h6.txt', np.array(self.human6_position))
            # np.savetxt('test_data/eight_h7.txt', np.array(self.human7_position))
            # np.savetxt('test_data/eight_h8.txt', np.array(self.human8_position))
            # np.savetxt('test_data/eight_r.txt', np.array(self.robot_position))
            # self.human1_position = []
            # self.human2_position = []
            # self.human3_position = []
            # self.human4_position = []
            # self.human5_position = []
            # self.human6_position = []
            # self.human7_position = []
            # self.human8_position = []
            # self.robot_position = []
        else:
            if self.reward_simple:
                reward = reward
            else:
                reward = reward + self.goal_distance_factor * (self.goal_distance_last - goal_dist)
        self.goal_distance_last = goal_dist
  
        for i, human in enumerate(self.humans):
            # let humans move circularly from two points
            if human.reached_destination():
                self.humans[i].gx = -self.humans[i].gx
                self.humans[i].gy = -self.humans[i].gy

        # get the observation
        ob_lidar = np.hstack((self.scan_current, self.scan_last_1, self.scan_last_2, self.scan_last_3, self.scan_last_4))
        dx = self.robot.gx - self.robot.px
        dy = self.robot.gy - self.robot.py
        theta = self.robot.theta
        y_rel = dy * cos(theta) - dx * sin(theta)
        x_rel = dy * sin(theta) + dx * cos(theta)
        r = hypot(x_rel, y_rel) / square_width
        t = atan2(y_rel, x_rel) / np.pi
        ob_position = np.array([r, t], dtype=np.float32)
        self_state = FullState(self.robot.px, self.robot.py, self.robot.vx, self.robot.vy, self.robot.radius, \
                               self.robot.gx, self.robot.gy, self.robot.v_pref, self.robot.theta)
        ob_state = [human.get_observable_state() for human in self.humans]
        ob_coordinate = JointState(self_state, ob_state)
        return ob_lidar, ob_position, ob_coordinate, reward, done, info

    def render(self, mode='laser'):
        if mode == 'laser':
            self.ax.set_xlim(-5.0, 5.0)
            self.ax.set_ylim(-5.0, 5.0)
            self.ax.set_xlabel('x/m', fontproperties = 'Times New Roman', fontsize=16)
            self.ax.set_ylabel('y/m', fontproperties = 'Times New Roman', fontsize=16)
            for human in self.humans:
                human_circle = plt.Circle(human.get_position(), human.radius, fill=False, color='b')
                self.ax.add_artist(human_circle)
            self.ax.add_artist(plt.Circle(self.robot.get_position(), self.robot.radius, fill=True, color='r'))
            # x, y, theta = self.robot.px, self.robot.py, self.robot.theta
            # dx = cos(theta)
            # dy = sin(theta)
            # self.ax.arrow(x, y, dx, dy,
            #     width=0.01,
            #     length_includes_head=True, 
            #     head_width=0.15,
            #     head_length=1,
            #     fc='r',
            #     ec='r')
            ii = 0
            lines = []
            while ii < n_laser:
                lines.append(self.scan_intersection[ii])
                ii = ii + 36
            lc = mc.LineCollection(lines)
            self.ax.add_collection(lc)
            plt.xticks(fontname = "Times New Roman", fontsize=16)
            plt.yticks(fontname = "Times New Roman", fontsize=16)

            plt.savefig('test_data/' + str(self.count) + '.pdf')  
            self.count = self.count + 1
            plt.draw()
            plt.pause(0.001)
            plt.cla()

            
