from __future__ import annotations  # To be removed in future python versions
from pathlib import Path
from dataclasses import dataclass, field
from typing import Callable
import math

import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

# from mayavi.api import Engine
# from mayavi.sources.api import ParametricSurface
# from mayavi.modules.api import Surface
# from mayavi import mlab

# (!!!) Somehow cv2 has to be imported after mayavi (?!) 
import cv2

from vartools.states import Pose
from vartools.dynamics import ConstantValue
from vartools.dynamics import LinearSystem, CircularStable
from vartools.colors import hex_to_rgba, hex_to_rgba_float
from vartools.linalg import get_orthogonal_basis

from dynamic_obstacle_avoidance.obstacles import Obstacle
from dynamic_obstacle_avoidance.obstacles import CuboidXd as Cuboid
from dynamic_obstacle_avoidance.obstacles import EllipseWithAxes as Ellipse

from nonlinear_avoidance.multi_body_franka_obs import create_3d_franka_obs
from nonlinear_avoidance.multi_obs_env import create_3d_human
#from nonlinear_avoidance.multi_obs_env import create_3d_franka_obs2
from nonlinear_avoidance.multi_body_franka_obs import (
    transform_from_multibodyobstacle_to_multiobstacle,
)
from nonlinear_avoidance.multi_obstacle_avoider import MultiObstacleAvoider
from nonlinear_avoidance.multi_obstacle_avoider import MultiObstacleContainer
from nonlinear_avoidance.dynamics.spiral_dynamics import SpiralingDynamics3D
from nonlinear_avoidance.dynamics.spiral_dynamics import SpiralingAttractorDynamics3D

from nonlinear_avoidance.nonlinear_rotation_avoider import (
    ConvergenceDynamicsWithoutSingularity,
)

#from scripts.three_dimensional.visualizer3d import CubeVisualizer

import matplotlib.pyplot as plt

import gymnasium as gym
import numpy as np
from pynput import keyboard
from envs.custom_scenarios import scenarios

import random
import time

from envs.config import register_custom_envs


Vector = np.ndarray

global ctrl
global rot_ctrl


#from mujoco_py import load_model_from_xml, MjSim, MjViewer
#import gym
#import mujoco_py

    
class MayaviAnimator:
    def __init__(
        self, it_max: int = 300, delta_time: float = 0.005, filename: str = "animation", 
        end_effector_position: np.ndarray = np.zeros(3),
        attractor_position: np.ndarray = np.zeros(3),
        dynamic_human = False) -> None:
        self.it_max = it_max
        self.delta_time = delta_time

        self.filename = filename
        self.figuretype = ".png"

        self.save_to_file = True

        self.main_folder = Path("figures")
        self.image_folder = Path("animation")

        # self.leading_zeros = math.ceil(math.log10(self.it_max + 1))
        self.leading_zeros = 4
 
        self.end_effector_position = end_effector_position
        self.attractor_position = attractor_position
        self.dynamic_human = dynamic_human  # the obstacle is dynamic

    def run(self,ii):    
        velocity = self.update_step(ii)
        if self.dynamic_human:
            self.update_human(ii)
        return velocity


    def setup(self, n_grid=5):
        dimension = 3

        # Trajectory integration
        start_positions = self.end_effector_position
        print("start_positions: ", start_positions.shape[1])
        self.n_traj = start_positions.shape[1]
        # self.trajectories shape is (3,301,25), 25 trajectories, 301 time steps and 3 dimensions
        self.trajectories = np.zeros((dimension, self.it_max + 1, self.n_traj))
        self.trajectories[:, 0, :] = start_positions
    

        cm = plt.get_cmap("gist_rainbow")
        self.color_list = [cm(1.0 * cc / self.n_traj) for cc in range(self.n_traj)]


#-----------------------main steps -------------------------------------------------------------------
        # step1 create tree of obstacles
        self.human_obstacle_3d = create_3d_human()
        #self.human_obstacle_3d = create_3d_franka_obs2()
        dynamics = LinearSystem(attractor_position=self.attractor_position)
        


        # step2 transform tree of obstacles to multiobstacle????
        transformed_human = transform_from_multibodyobstacle_to_multiobstacle(
            self.human_obstacle_3d
        )

        # step3 create container of obstacles
        self.container = MultiObstacleContainer()
        if self.dynamic_human:
            self.container.append(transformed_human)
        
        #self.container.append(transformed_human)

        # step4 create avoider
        self.avoider = MultiObstacleAvoider.create_with_convergence_dynamics(
            obstacle_container=self.container,
            initial_dynamics=dynamics,
            # reference_dynamics=linearsystem(attractor_position=dynamics.attractor_position),
            create_convergence_dynamics=True,
            convergence_radius=0.55 * math.pi,
            smooth_continuation_power=0.7,
        )
#----------------------------------------------------------------------------------------------------------
        #self.visualizer = Visualization3D()

        #self.dynamic_human = False

    def update_human(self, ii: int) -> None:
        # amplitude_leg1 = 0.12
        # frequency_leg1 = 0.2
        # idx = self.human_obstacle_3d.get_obstacle_id_from_name("leg1")
        # obstacle = self.human_obstacle_3d[idx]
        # rotation = Rotation.from_euler(
        #     "y", amplitude_leg1 * np.sin(ii * frequency_leg1)
        # )
        # obstacle.orientation = obstacle.orientation * rotation

        # amplitude_leg1 = -0.12
        # frequency_leg1 = 0.2
        # idx = self.human_obstacle_3d.get_obstacle_id_from_name("leg2")
        # obstacle = self.human_obstacle_3d[idx]
        # rotation = Rotation.from_euler(
        #     "y", amplitude_leg1 * np.sin(ii * frequency_leg1)
        # )
        # obstacle.orientation = obstacle.orientation * rotation

        amplitude_leg1 = -0.08
        frequency_leg1 = 0.2
        idx = self.human_obstacle_3d.get_obstacle_id_from_name("upperarm1")
        obstacle = self.human_obstacle_3d[idx]
        rotation = Rotation.from_euler(
            "y", amplitude_leg1 * np.sin(ii * frequency_leg1)
        )
        obstacle.orientation = obstacle.orientation * rotation

        amplitude_leg1 = -0.12
        frequency_leg1 = 0.2
        idx = self.human_obstacle_3d.get_obstacle_id_from_name("lowerarm1")
        obstacle = self.human_obstacle_3d[idx]
        rotation = Rotation.from_euler(
            "y", amplitude_leg1 * np.sin(ii * frequency_leg1)
        )
        obstacle.orientation = obstacle.orientation * rotation

        amplitude_leg1 = 0.05
        frequency_leg1 = 0.2
        idx = self.human_obstacle_3d.get_obstacle_id_from_name("upperarm2")
        obstacle = self.human_obstacle_3d[idx]
        rotation = Rotation.from_euler(
            "x", amplitude_leg1 * np.sin(ii * frequency_leg1)
        )
        obstacle.orientation = rotation * obstacle.orientation

        amplitude_leg1 = -0.05
        frequency_leg1 = 0.2
        idx = self.human_obstacle_3d.get_obstacle_id_from_name("lowerarm2")
        obstacle = self.human_obstacle_3d[idx]
        rotation = Rotation.from_euler(
            "x", amplitude_leg1 * np.sin(ii * frequency_leg1)
        )
        obstacle.orientation = obstacle.orientation * rotation

        reference_point_updated = obstacle.get_reference_point(in_global_frame=True)

        self.human_obstacle_3d.align_obstacle_tree()


    def update_step(self, ii: int) -> None:
        # from mayavi import mlab
        for it_traj in range(self.n_traj):
            velocity = self.avoider.evaluate_sequence(self.trajectories[:, ii, it_traj])
            # self.trajectories[:, ii + 1, it_traj] = (
            #     velocity * self.delta_time + self.trajectories[:, ii, it_traj]
            # )
        return velocity
    
    def update_trajectories(self,position,ii:int):
        for it_traj in range(self.n_traj):
            self.trajectories[:, ii + 1, it_traj] = position


# def main_animation_dynamic():
#     #animator = MayaviAnimator(filename="avoidance_around_dynamic_human", it_max=20)
#     animator = MayaviAnimator(filename="avoidance_around_dynamic_human")

#     # animator.setup(n_grid=1)
#     animator.setup()

#     animator.dynamic_human = True
#     animator.run()
