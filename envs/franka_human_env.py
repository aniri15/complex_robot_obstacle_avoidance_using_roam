#from mujoco_py import load_model_from_xml
import mujoco
import mujoco.viewer
#import gymnasium as gym
#from gymnasium import spaces
#from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer
#from gymnasium.utils.ezpickle import EzPickle
from gymnasium_robotics.utils import mujoco_utils
#from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer
from typing import Optional
import mediapy as media

import time
import numpy as np
from envs.config import register_custom_envs
from envs.custom_scenarios import scenarios
#from envs.ik_controller import LevenbegMarquardtIK
import matplotlib.pyplot as plt
from franka_gym_test.franka_human_avoider import MayaviAnimator
from franka_gym_test.pandaIk import PandaIK
import os
from dynamic_obstacle_avoidance.obstacles import CuboidXd as Cuboid
from dynamic_obstacle_avoidance.obstacles import EllipseWithAxes as Ellipse
import pybullet

DEFAULT_CAMERA_CONFIG = {
    "distance": 1.4,
    "azimuth": 150.0,
    "elevation": -25.0,
    "lookat": np.array([1.5, 0, 0.75]),
}
DEFAULT_SIZE = 480


class LevenbegMarquardtIK:
    
    def __init__(self, model, data, step_size, tol, alpha, jacp, jacr, damping):
        self.model = model
        self.data = data
        self.step_size = step_size
        self.tol = tol
        self.alpha = alpha
        self.jacp = jacp
        self.jacr = jacr
        self.damping = damping
    
    def check_joint_limits(self, q):
        """Check if the joints is under or above its limits"""
        for i in range(len(q)):
            q[i] = max(self.model.jnt_range[i][0], min(q[i], self.model.jnt_range[i][1]))

    #Levenberg-Marquardt pseudocode implementation
    def calculate(self, goal, init_q, body_id):
        """Calculate the desire joints angles for goal"""
        self.data.qpos = init_q
        mujoco.mj_forward(self.model, self.data)
        current_pose = self.data.body(body_id).xpos
        error = np.subtract(goal, current_pose)
        delta_q_norm = 0
        while (np.linalg.norm(error) >= self.tol):
            #calculate jacobian
            mujoco.mj_jac(self.model, self.data, self.jacp, self.jacr, goal, body_id)
            #calculate delta of joint q
            n = self.jacp.shape[1]
            I = np.identity(n)
            product = self.jacp.T @ self.jacp + self.damping * I
            
            if np.isclose(np.linalg.det(product), 0):
                j_inv = np.linalg.pinv(product) @ self.jacp.T
            else:
                j_inv = np.linalg.inv(product) @ self.jacp.T
            
            delta_q = j_inv @ error
            #compute next step
            self.data.qpos += self.step_size * delta_q
            #check limits
            self.check_joint_limits(self.data.qpos)
            #compute forward kinematics
            mujoco.mj_forward(self.model, self.data) 
            #calculate new error
            error = np.subtract(goal, self.data.body(body_id).xpos)
            error_norm = np.linalg.norm(error)
            delta_q_norm1 = np.linalg.norm(delta_q)
            # if the 
            # if abs(delta_q_norm1 - delta_q_norm) < 0.001:
            #     break
            delta_q_norm = delta_q_norm1
        return self.data  
    
class FrankaHumanEnv():
    def __init__(
            self,
            scene_path,
            control_mode='position',
            obj_range=0.1,
            target_range=0.1,
            num_obst=1,
            obj_goal_dist_threshold=0.03,
            obj_gripper_dist_threshold=0.02,
            max_vel=0.1,
            obj_lost_reward=-0.2,
            collision_reward=-1.,
            scenario=None,
            dynamic_human=False,
            **kwargs
    ):
        self._mujoco = mujoco
        self._utils = mujoco_utils
        self.control_mode = control_mode
        #self.human_shape = HumanShape(pos=[0.5, 0.5, 0])  # Position human obstacle
        #self.fullpath = os.path.join(os.path.dirname(__file__), "assets", "franka_test.xml")
        self.fullpath = scene_path
        self.FRAMERATE = 60 #(Hz)
        self.height = 300
        self.width = 300
        self.n_frames = 60
        self.frames = []
        self.dynamic_human = dynamic_human
        self.pandaik = PandaIK()

        # Initialize MuJoCo environment with the combined model
        self.spec = mujoco.MjSpec.from_file(self.fullpath)
        self.model = self.spec.compile()
        
        #self.model = mujoco.MjModel.from_xml_path(self.fullpath)
        self.data = mujoco.MjData(self.model)
        #self.model = load_model_from_xml(full_model_xml)
        self.render_mode = 'human'
        self.init_qpos = self.data.qpos.copy()
        
        # Todo: two render methods are used, need to be unified
        #self.renderer = mujoco.Renderer(self.model, self.height, self.width)
        #self.mujoco_renderer = MujocoRenderer(
        #    self.model, self.data, DEFAULT_CAMERA_CONFIG
        #)
        #Make a new camera, move it to a closer distance.
        self.camera = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(self.model, self.camera)
        self.camera.distance = 4
        self.it_max = 2000              #horizon

        # update step size
        self.step_size_position = 0.01
        self.step_size_ik = 0.1 #0.5
        self.delta_time = 0.01      # 100hz

        # initialize for ik controller
        self.tol = 0.05   # unit: m, tolerance is 5cm
        self.damping = 1e-3
        self.jacp = np.zeros((3, self.model.nv)) #translation jacobian
        self.jacr = np.zeros((3, self.model.nv)) #rotational jacobian
        self.trajectory_qpos = []
        self.data_storage = []
        self.model_storage = []
        self.point_storage = []
        #self.obstacle_number = 7
        self.initial_qpos = {
            #'object0:joint': [1.25, 0.53, 0.4, 1., 0., 0., 0.],#[1.25, 0.53, 0.4, 1., 0., 0., 0.]
            'joint1': 0.0254,
            'joint2': -0.2188,
            'joint3': -0.0265,
            'joint4': -2.6851,
            'joint5': -0.0092,
            'joint6': 2.4664,
            'joint7': 0.0068,
        }
        self.singularities_number = 0
        self.initial_simulation()
        
    def initial_simulation(self):
        for name, value in self.initial_qpos.items():
            self._utils.set_joint_qpos(self.model, self.data, name, value)
        mujoco.mj_forward(self.model, self.data)
    
    def bulid_goal_object(self,goal):
        name = "goal"
        position = goal
        size = [0.02, 0.02, 0.02]
        self.spec.worldbody.add_geom(name= name +'_geom',
                type=mujoco.mjtGeom.mjGEOM_BOX,
                rgba=[0, 1, 0, 1],
                size= size,
                pos= position)
        self.model = self.spec.compile()
        self.data = mujoco.MjData(self.model)

    def render(self):
        self.mujoco_renderer.render(self.render_mode)
        #self.renderer.update_scene(self.data)

    def check_joint_limits(self, q):
        """Check if the joints is under or above its limits"""
        for i in range(len(q)):
            q[i] = max(self.model.jnt_range[i][0], min(q[i], self.model.jnt_range[i][1]))

    #Levenberg-Marquardt pseudocode implementation
    def calculate(self, goal, init_q, body_id):
        """Calculate the desire joints angles for goal"""
        self.data.qpos = init_q
        #mujoco.mj_forward(self.model, self.data)
        mujoco.mj_step(self.model, self.data)
        current_pose = self.data.body(body_id).xpos
        error = np.subtract(goal, current_pose)
        error_norm = np.linalg.norm(error)
        delta_q_norm = 0

        self.trajectory_qpos.append(self.data.qpos.copy())
        #self.model_storage.append((self.model.geom_pos, self.model.geom_quat))
        #self.model_storage.append((self.data.xpos, self.data.xquat))
        self.store_data()
        while (np.linalg.norm(error) >= self.tol):
            #calculate jacobian
            mujoco.mj_jac(self.model, self.data, self.jacp, self.jacr, goal, body_id)
            #calculate delta of joint q
            n = self.jacp.shape[1]
            I = np.identity(n)
            product = self.jacp.T @ self.jacp + self.damping * I
            jac = np.zeros((6, self.model.nv))
            jac[:3] = self.jacp
            jac[3:] = self.jacr
            if np.linalg.matrix_rank(jac) < 6:
                print("jacobian is singular")
                self.singularities_number += 1
            
            if np.isclose(np.linalg.det(product), 0):
                j_inv = np.linalg.pinv(product) @ self.jacp.T
            else:
                j_inv = np.linalg.inv(product) @ self.jacp.T
            
            delta_q = j_inv @ error
            #compute next step
            self.data.qpos += self.step_size_ik * delta_q
            #error = np.subtract(goal, self.data.body(body_id).xpos)
            #error_norm2 = np.linalg.norm(error)
            #self.data.qpos += error_norm2 * delta_q
            #check limits
            self.check_joint_limits(self.data.qpos)

            #self.trajectory_qpos.append(self.data.qpos.copy())
            mujoco.mj_step(self.model, self.data)
            error = np.subtract(goal, self.data.body(body_id).xpos)
            error_norm2 = np.linalg.norm(error)
            delta_q_norm1 = np.linalg.norm(delta_q)
            # if abs(delta_q_norm1 - delta_q_norm) < 0.001:
            #     break
            #delta_q_norm = delta_q_norm1
            self.trajectory_qpos.append(self.data.qpos.copy())
            #self.model_storage.append((self.data.xpos, self.data.xquat))
            self.store_data()
            #self.data_storage.append(self.data)
            #self.model_storage.append(self.model)
        return self.data 
     
    def RRMC(self, goal, velocity, init_q, body_id):
        self.data.qpos = init_q
        #mujoco.mj_forward(self.model, self.data)
        mujoco.mj_step(self.model, self.data)
        current_pose = self.data.body(body_id).xpos
        error = np.subtract(goal, current_pose)
        #error_norm = np.linalg.norm(error)
        #delta_q_norm = 0
        self.trajectory_qpos.append(self.data.qpos.copy())
        self.store_data()
        jac = np.zeros((6, self.model.nv))
        # while (np.linalg.norm(error) >= self.tol):
        #     #calculate jacobian
        #     mujoco.mj_jac(self.model, self.data, self.jacp, self.jacr, goal, body_id)
        #     j = np.concatenate((self.jacp, self.jacr), axis=0)
        #     #calculate delta of joint q
        #     current_pose = self.data.body(body_id).xpos
        #     dv = self.desired_velocity(error)
        #     j_inv = np.linalg.pinv(j)
        #     #print("jacp: ", self.jacp.shape)
        #     #print("jacr: ", self.jacr.shape)  
        #     #print("j_inv: ", j_inv.shape)  
        #     #print("j",j.shape)
        #     delta_q = j_inv @ dv

        #     #compute next step
        #     self.data.qpos += self.step_size_ik * delta_q

        #     #check limits
        #     self.check_joint_limits(self.data.qpos)

        #     mujoco.mj_step(self.model, self.data)
        #     error = np.subtract(goal, self.data.body(body_id).xpos)
  
        #     # if abs(delta_q_norm1 - delta_q_norm) < 0.001:
        #     #     break
        #     #delta_q_norm = delta_q_norm1
        #     self.trajectory_qpos.append(self.data.qpos.copy())
        #     self.store_data()
        time_start = self.data.time
        time_end = 0
        time_diff = 0
        while (np.linalg.norm(time_diff)<= self.delta_time):
            #calculate jacobian
            #mujoco.mj_jac(self.model, self.data, self.jacp, self.jacr, goal, body_id)
            mujoco.mj_jac(self.model, self.data, jac[:3], jac[3:], current_pose, body_id)
            j = jac
            #print("rank of jacobian: ", np.linalg.matrix_rank(j))
            if np.linalg.matrix_rank(j) < 6:
                print("jacobian is singular")
                self.singularities_number += 1
                breakpoint()
            #calculate delta of joint q
            current_pose = self.data.body(body_id).xpos
            #print("current_pose: ", current_pose)
            dv = self.desired_velocity(velocity)
            n = j.shape[1]
            I = np.identity(n)
            product = j.T @ j + self.damping * I
            
            if np.isclose(np.linalg.det(product), 0):
                j_inv = np.linalg.pinv(product) @ j.T
            else:
                j_inv = np.linalg.inv(product) @ j.T

            #j_inv = np.linalg.pinv(j)
            delta_q = j_inv @ dv
          
            if np.isclose(np.linalg.norm(delta_q), 0):
                shape = delta_q.shape
                delta_q = np.eye(shape)*0.01

            #compute next step
            #self.data.qpos += self.step_size_ik * delta_q
            #self.data.qpos = self.step_size_ik * delta_q
            self.data.qvel = delta_q

            #check limits
            self.check_joint_limits(self.data.qpos)

            mujoco.mj_step(self.model, self.data)
            error = np.subtract(goal, self.data.body(body_id).xpos)

            # if abs(delta_q_norm1 - delta_q_norm) < 0.001:
            #     break
            #delta_q_norm = delta_q_norm1
            self.trajectory_qpos.append(self.data.qpos.copy())
            self.store_data()
            time_end = self.data.time
            time_diff = time_end - time_start
            #print("time_diff: ", time_diff)
        return self.data

    def desired_velocity(self, velocity):
        #velocity = np.linalg.norm(error)
        #velocity = self.step_size_position * velocity
        # set the desired velocity to end effector
        dv = np.array([velocity[0],velocity[1],velocity[2], 0, 0, 0])
        return dv
    
    def use_ik_controller(self, goal):
        #Init variables.
        init_q = self.data.qpos.copy()
        body_id = self.model.body('robot:gripper_link').id
        #body_id = self.model.site('robot0:grip').id
        goal = goal[0]
        print("current goal: ", goal)
        print("current init_q: ", init_q)
        print("current_pose: ", self.data.body(body_id).xpos)
        self.data = self.calculate(goal, init_q, body_id) #calculate the qpos
        print("time: ", self.data.time)

    def panda_ik_controller(self, goal):
        #Init variables.
        init_q = self.data.qpos.copy()
        body_id = self.model.body('robot:gripper_link').id
        #body_id = self.model.site('robot0:grip').id
        goal_position = goal[0]
        goal_orientation = [0,0, 0.0, 0.0]
        goal = np.concatenate((goal_position, goal_orientation))
        
        print("current goal: ", goal)
        print("current init_q: ", init_q)
        print("current_pose: ", self.data.body(body_id).xpos)
        #self.data = self.calculate(goal, init_q, body_id) #calculate the qpos
        self.trajectory_qpos.append(init_q)
        result_q = self.pandaik.compute_inverse_kinematics(goal, init_q)
        self.data.qpos = result_q
        mujoco.mj_step(self.model, self.data)
        self.store_data()
        print("time: ", self.data.time)

    def resolved_rate_motion_control(self, goal, velocity):
        #Init variables.
        init_q = self.data.qpos.copy()
        body_id = self.model.body('robot:gripper_link').id
        #body_id = self.model.site('robot0:grip').id
        goal = goal[0]
        
        print("current goal: ", goal)
        print("current init_q: ", init_q)
        print("current_pose: ", self.data.body(body_id).xpos)

        #self.data = self.RRMC(goal, init_q, body_id) #calculate the qpos
        self.data = self.RRMC(goal, velocity, init_q, body_id) #calculate the qpos
        print("time: ", self.data.time)

    def compare_frames(self, target):
        #Plot results
        print("Results")
        result = self.data.qpos.copy()

        self.data.qpos = self.init_qpos
        mujoco.mj_forward(self.model, self.data)
        self.renderer.update_scene(self.data, self.camera)
        target_plot = self.renderer.render()

        self.data.qpos = result
        mujoco.mj_forward(self.model, self.data)
        #result_point = data.body('wrist_3_link').xpos
        result_point = self.data.body('robot:gripper_link').xpos
        self.renderer.update_scene(self.data, self.camera)
        result_plot = self.renderer.render()

        print("testing point =>", target)
        print("Levenberg-Marquardt result =>", result_point, "\n")

        images = {
            'Testing point': target_plot,
            'Levenberg-Marquardt result': result_plot,
        }

        #media.show_images(images)
        return images
    
    # Function to display images using matplotlib
    def show_images(self,images):
        fig, axes = plt.subplots(1, len(images), figsize=(10, 5))
        if len(images) == 1:
            axes = [axes]  # Ensure axes is iterable if there's only one image
        
        for ax, (title, img) in zip(axes, images.items()):
            ax.imshow(img)
            ax.set_title(title)
            ax.axis('off')
        
        plt.show()
    
    def real_time_render(self):
        m = self.model
        d = self.data
        with mujoco.viewer.launch_passive(m, d) as viewer:
            # Close the viewer automatically after 30 wall-seconds.
            start = time.time()
            while viewer.is_running() and time.time() - start < 80:
                step_start = time.time()

                # mj_step can be replaced with code that also evaluates
                # a policy and applies a control signal before stepping the physics.
                mujoco.mj_step(m, d)

                # Example modification of a viewer option: toggle contact points every two seconds.
                with viewer.lock():
                    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(d.time % 2)

                # Pick up changes to the physics state, apply perturbations, update options from GUI.
                viewer.sync()

                # Rudimentary time keeping, will drift relative to wall clock.
                time_until_next_step = m.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)

    def real_time_movement_render(self, init_position, goal):
        m = self.model
        d = self.data
        with mujoco.viewer.launch_passive(m, d) as viewer:
            # Close the viewer automatically after 30 wall-seconds.
            start = time.time()
            while viewer.is_running() and time.time() - start < 80:
                step_start = time.time()

                self.move_robot(init_position, goal)
                # mj_step can be replaced with code that also evaluates
                # a policy and applies a control signal before stepping the physics.
                #mujoco.mj_step(m, d)

                # Example modification of a viewer option: toggle contact points every two seconds.
                with viewer.lock():
                    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(d.time % 2)

                # Pick up changes to the physics state, apply perturbations, update options from GUI.
                viewer.sync()

                # Rudimentary time keeping, will drift relative to wall clock.
                time_until_next_step = m.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)

    def replay(self):
        # use the mujoco viewer to replay the simulation
        m = self.model
        d = self.data
        
        with mujoco.viewer.launch_passive(m, d) as viewer:
                # Close the viewer automatically after 30 wall-seconds.
                start = time.time()
                while viewer.is_running() and time.time() - start < 80:
                    step_start = time.time()
                    for i in range(len(self.trajectory_qpos)):
                        d.qpos = self.trajectory_qpos[i]
                        gem_pos = np.zeros((89,3))
                        #d.xpos, d.xquat = self.model_storage[i]
                        m.geom_pos, m.geom_quat = self.model_storage[i]

                        mujoco.mj_step(m, d)
                        viewer.sync()

                    # Rudimentary time keeping, will drift relative to wall clock.
                    #time_until_next_step = m.opt.timestep - (time.time() - step_start)
                    #if time_until_next_step > 0:
                    #    time.sleep(time_until_next_step)
    
    def render2(self):
        # use the mujoco viewer to replay the simulation
        m = self.model
        d = self.data

        with mujoco.viewer.launch_passive(m, d) as viewer:
            # Close the viewer automatically after 30 wall-seconds.
            start = time.time()
            while viewer.is_running() and time.time() - start < 80:
                step_start = time.time()
                #current_pose = self.data.body('robot:gripper_link').xpos
                #print("current_pose: ", current_pose)
                mujoco.mj_step(m, d)

                    # Example modification of a viewer option: toggle contact points every two seconds.
                    #with viewer.lock():
                    #    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(d.time % 2)

                    # Pick up changes to the physics state, apply perturbations, update options from GUI.
                viewer.sync()

                # Rudimentary time keeping, will drift relative to wall clock.
                time_until_next_step = m.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)

    def move_robot(self,init_position, goal):
        #self.mujoco_renderer.render(self.render_mode)
        it_max = self.it_max
        animator = MayaviAnimator(it_max=it_max, end_effector_position=init_position, 
                                  attractor_position=goal, dynamic_human=self.dynamic_human)    
        animator.setup()
        self.human_obstacle = animator.human_obstacle_3d
        self.adjust_obstacle_init()
        #animator.dynamic_human = self.dynamic_human


        for ii in range(it_max):
            #update trajectory
            velocity = animator.run(ii=ii)
            if not ii % 10:
                print(f"it={ii}")
            print('iteration: ', ii)
            print("velocity: ", velocity)
            next_desired_pos = init_position + velocity*self.delta_time  #self.step_size_position
            #self.use_ik_controller(next_desired_pos)
            #self.panda_ik_controller(next_desired_pos)
            self.resolved_rate_motion_control(next_desired_pos, velocity)

            # update end effector position
            init_position = self.get_ee_position()
            init_position = np.array([[init_position[0], init_position[1], init_position[2]]])
            
            animator.update_trajectories(init_position, ii)
            self.human_obstacle = animator.human_obstacle_3d
            self.adjust_obstacle()

            if self.check_goal_reach(init_position, goal):
                print("goal reached")
                break
            
    def move_point(self,init_position, goal):
         #self.mujoco_renderer.render(self.render_mode)
        it_max = self.it_max
        animator = MayaviAnimator(it_max=it_max, end_effector_position=init_position, attractor_position=goal)    
        animator.setup()
        self.human_obstacle = animator.human_obstacle_3d
        self.adjust_obstacle_init()
        animator.dynamic_human = self.dynamic_human

        for ii in range(it_max):
            #update trajectory
            velocity = animator.run(ii=ii)
            #if not ii % 10:
            #    print(f"it={ii}")
            print('iteration: ', ii)
            print("velocity: ", velocity)
            next_desired_pos = init_position + velocity*self.step_size_position
            # update end effector position
            init_position = next_desired_pos
            #init_position = np.array([[init_position[0], init_position[1], init_position[2]]])
            animator.update_trajectories(init_position, ii)
            print('init_position: ', init_position)
            self.point_storage.append(init_position)
            self.store_data()
            self.human_obstacle = animator.human_obstacle_3d
            self.adjust_obstacle()
            if self.check_goal_reach(init_position, goal):
                print("goal reached")
                break
            
    def render_point_trajectory(self):
        # use the mujoco viewer to replay the simulation
        m = self.model
        d = self.data
        radius = 0.1
        
        with mujoco.viewer.launch_passive(m, d) as viewer:
                # Close the viewer automatically after 30 wall-seconds.
                start = time.time()
                #geom = mujoco._structs.MjvGeom()
                viewer.user_scn.ngeom += 1
                mujoco.mjv_initGeom(viewer.user_scn.geoms[viewer.user_scn.ngeom-1], mujoco.mjtGeom.mjGEOM_SPHERE, 
                                            np.array([0.1,0.1,0.1]),np.zeros(3), np.zeros(9),np.array([1,0,1,1]))
                while viewer.is_running() and time.time() - start < 80:
                    step_start = time.time()
                    for i in range(len(self.point_storage)-1):
                        # show the point trajectory
                        #mujoco.mjv_initGeom(viewer.user_scn.geoms[viewer.user_scn.ngeom-1], mujoco.mjtGeom.mjGEOM_SPHERE, 
                        #                    np.array([0.1,0.1,0.1]),np.zeros(3), np.zeros(9),np.array([1,0,1,1]))
                       
                        from_ = self.point_storage[i].flatten()
                        to_ = self.point_storage[i+1].flatten()
                        mujoco.mjv_connector(viewer.user_scn.geoms[viewer.user_scn.ngeom-1], 
                                              mujoco.mjtGeom.mjGEOM_LINE, radius, from_, to_)
                
                        #m.geom_pos, m.geom_quat = self.model_storage[i]

                        mujoco.mj_step(m, d)
                        viewer.sync()

        
    def check_goal_reach(self, position, goal):
        if np.linalg.norm(position - goal) < self.tol:
            return True
        return False

        #env.render()
    def transform_degree_to_quat(self, degree, direction):
        """Next try is to calculate the values according to the tutorials formula:
        Form:
        quat = [x, y, z, w]
        degree = [x_degree, y_degree, z_degree]
        direction = [x_dir, y_dir, z_dir]
        Example: rotate 45 degrees around x, y, z axis(direction: -x, -y, z)
        x = RotationAxis.x * sin(RotationAngle / 2) => -1 * sin( (1/4 * pi) / 2 ) = -0,382
        y = RotationAxis.y * sin(RotationAngle / 2) => -1 * sin( (1/4 * pi) / 2 ) = -0,382
        z = RotationAxis.z * sin(RotationAngle / 2) => 1 * sin( (1/4 * pi) / 2 ) = 0,382
        w = cos( (1/4 * pi) / 2) = 0,923"""
        quat = np.zeros(4)
        # transform degree to radian
        radian = np.radians(degree)
        # calculate the values
        quat[0] = direction[0] * np.sin(radian[0] / 2)
        quat[1] = direction[1] * np.sin(radian[1] / 2)
        quat[2] = direction[2] * np.sin(radian[2] / 2)
        quat[3] = np.cos(radian[0] / 2)
        return quat
    
    def store_data(self):
        # store the data of the simulation
        self.model_storage.append((self.model.geom_pos.copy(), self.model.geom_quat.copy()))

    def adjust_obstacle(self):
        # set the initial position, size and orientation of the obstacles
        print('------------------------------------------------------')
        obstacles = self.human_obstacle
        for ii, obs in enumerate(obstacles):
            if isinstance(obs, Ellipse):
                orientation = [0,0,0]
                #if obs.pose.orientation is not None:
                    # orientation is degree??
                quat = obs.pose.orientation.as_quat()
                if quat[0] < 0:
                    quat = quat * (-1)
                    obs.pose.orientation.from_quat(quat)
                orientation = obs.pose.orientation.as_euler("xyz", degrees=True)

                if np.isclose(abs(orientation[0]), 180) and ii in [7, 8]:
                    orientation[0] = 0
                    orientation[1] = orientation[1] - 180
                    orientation[2] = orientation[2]

                direction = [1,1,1]
                xml_quat = self.transform_degree_to_quat(orientation,direction)
                name_body = obs.name 
                name_geom = name_body + '_geom'

                #print('orientation', orientation)
                #print('position', obs.center_position)

                self.set_geom_quat(name_geom,xml_quat)
                self.set_geom_xpos(name_geom, obs.center_position)
                #self.set_geom_size(name_geom, obs.axes_length * 0.5)
                #self.set_body_xpos(name_body, obs.center_position)
                #self.set_body_xquat(name_body, xml_quat)
                #self.set_body_size(name_body,obs.axes_length*0.5)
            if isinstance(obs, Cuboid):
                name_body = obs.name
                name_geom = name_body + '_geom'
              
                #print('position', obs.center_position)
                self.set_geom_xpos(name_geom, obs.center_position)
                #self.set_geom_size(name_geom, obs.axes_length * 0.5)
                #self.set_body_xpos(name_body, obs.center_position)
                #self.set_body_size(name_body,obs.axes_length*0.5)

        #self.data = mujoco.MjData(self.model)
        #self.data.qpos = self.trajectory_qpos[-1]
        mujoco.mj_forward(self.model, self.data)
        
        #self.model = self.spec.compile()

    def adjust_obstacle_init(self):
        # set the initial position, size and orientation of the obstacles
        
        obstacles = self.human_obstacle
        for ii, obs in enumerate(obstacles):
            if isinstance(obs, Ellipse):
                orientation = [0,0,0]
                #if obs.pose.orientation is not None:
                    # orientation is degree??
                quat = obs.pose.orientation.as_quat()
                if quat[0] < 0:
                    quat = quat * (-1)
                    obs.pose.orientation.from_quat(quat)
                orientation = obs.pose.orientation.as_euler("xyz", degrees=True)

                if np.isclose(abs(orientation[0]), 180) and ii in [7, 8]:
                    orientation[0] = 0
                    orientation[1] = orientation[1] - 180
                    orientation[2] = orientation[2]
                #print('orientation', orientation)
                direction = [1,1,1]
                xml_quat = self.transform_degree_to_quat(orientation,direction)
                name = obs.name
                position = obs.center_position
                size = obs.axes_length * 0.5
                color = ii*0.1
                print('position', position)
                print('orientation', orientation)
                #self.spec.worldbody.add_geom(name= name,
                # body_ellipse = self.spec.worldbody.add_body(name= name,quat= xml_quat,  pos= position,)
                # body_ellipse.add_geom(name= name +'_geom',
                #     type=mujoco.mjtGeom.mjGEOM_ELLIPSOID,
                #     # change the color every iteration
                #     rgba= [color, color+0.3, 1, 1],
                #     size= size,
                #     )
                self.spec.worldbody.add_geom(name= name +'_geom',
                        type=mujoco.mjtGeom.mjGEOM_ELLIPSOID,
                        # change the color every iteration
                        rgba= [color, color+0.3, 1, 1],
                        size= size,
                        pos= position,
                        quat= xml_quat)
            if isinstance(obs, Cuboid):
                name = obs.name
                position = obs.center_position
                size = obs.axes_length * 0.5
                print('position', position)
                #self.spec.worldbody.add_geom(name= name,
                # body_cube = self.spec.worldbody.add_body(name= name,pos= position,)
                # body_cube.add_geom(name= name +'_geom',
                #         type=mujoco.mjtGeom.mjGEOM_BOX,
                #         rgba=[1, 0, 0, 1],
                #         size= size)
                self.spec.worldbody.add_geom(name= name +'_geom',
                        type=mujoco.mjtGeom.mjGEOM_BOX,
                        rgba=[1, 0, 0, 1],
                        size= size,
                        pos= position)
        self.model = self.spec.compile()
        self.data = mujoco.MjData(self.model)
        self.initial_simulation()
        #mujoco.mj_forward(self.model,self.data)
        #print(self.spec.to_xml())


        
    # ------------------read from the xml file ----------------------------------------
    def get_ee_position(self):
        # gripper
        grip_pos = self._utils.get_site_xpos(self.model, self.data, "robot0:grip")
        return grip_pos
        
    def get_goal_position(self):
        goal_pos = self._utils.get_site_xpos(self.model, self.data, "target_pos")
        return goal_pos

    def get_obstacle_info(self):
        obstacles = []
        for n in range(self.total_obst):
            obstacles.append({
                'pos': self._utils.get_joint_qpos(self.model, self.data, f'obstacle{n}' + ':joint')[:3],
                'size': self.get_geom_size(f'obstacle{n}')
            })
        return obstacles
#---------------------------body-------------------------------------------------
    def get_body_xpos(self, name):
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        return self.model.body_xpos[body_id]
        #return self.data.xpos[body_id]

    def set_body_xpos(self, name, pos):
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        self.model.body_xpos[body_id] = pos
        #self.data.xpos[body_id] = pos

    def get_body_xquat(self, name):
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        return self.model.body_quat[body_id]
        #return self.data.xquat[body_id]
    
    def set_body_xquat(self, name, quat):
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        self.model.body_quat[body_id] = quat
        #self.data.xquat[body_id] = quat

    def set_body_size(self, name, size):
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        self.model.body_size[body_id] = size
        #self.data.body_size[body_id] = size

    def get_body_size(self, name):
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        return self.model.body_size[body_id]
        #return self.data.body_size[body_id]
    
#---------------------------geom----------------------------------------------------------
    def get_geom_size(self, name):
        geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
        return self.model.geom_size[geom_id]
        #return self.data.geom_size[geom_id]

    def set_geom_size(self, name, size):
        geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
        self.model.geom_size[geom_id] = size
        #self.data.geom_size[geom_id] = size
        #self.data.size[geom_id] = size
        #adjust rbound (radius of bounding sphere for which collisions are not checked)
        #self.model.geom_rbound[geom_id] = np.sqrt(np.sum(np.square(self.model.geom_size[geom_id])))

    def get_geom_xpos(self, name):
        geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
        return self.model.geom_pos[geom_id]
        r#eturn self.data.geom_xpos[geom_id]
    
    def set_geom_xpos(self, name, pos):
        geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
        self.model.geom_pos[geom_id] = pos
        #self.data.geom_xpos[geom_id] = pos

    def get_geom_quat(self, name):
        geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
        return self.model.geom_quat[geom_id]
        #return self.data.geom_xmat[geom_id]
    
    def set_geom_quat(self, name, quat):
        geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
        self.model.geom_quat[geom_id] = quat
        #self.data.geom_xmat[geom_id] = quat

#---------------------------site---------------------------------------------
    def get_site_size(self, name):
        site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name)
        return self.data.site_size[site_id]

    def set_site_size(self, name, size):
        site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name)
        self.data.site_size[site_id] = size

    def get_site_pos(self, name):
        site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name)
        return self.data.site_pos[site_id]

    def set_site_pos(self, name, pos):
        site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name)
        self.data.site_pos[site_id] = pos

    def set_site_quat(self, name, quat):
        site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name)
        self.data.site_quat[site_id] = quat

    #def create_obstacle_ellipse(self, name, pos, quat, size):


    # Small bug in gymnasium robotics returns wrong values, use local implementation for now.
    @staticmethod
    def custom_get_joint_qvel(model, data, name):
        """Return the joints linear and angular velocities (qvel) of the model."""
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        joint_type = model.jnt_type[joint_id]
        joint_addr = model.jnt_dofadr[joint_id]

        if joint_type == mujoco.mjtJoint.mjJNT_FREE:
            ndim = 6
        elif joint_type == mujoco.mjtJoint.mjJNT_BALL:
            ndim = 4
        else:
            assert joint_type in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE)
            ndim = 1

        start_idx = joint_addr
        end_idx = joint_addr + ndim

        return data.qvel[start_idx:end_idx]

    @property
    def dt(self):
        """Return the timestep of each Gymanisum step."""
        return self.model.opt.timestep * self.n_substeps

# main function
#if __name__ == '__main__':
    #test_env()
    #env = FrankaHumanEnv()
    #obs = env.get_ee_pos()
    #obs = env.render()
    