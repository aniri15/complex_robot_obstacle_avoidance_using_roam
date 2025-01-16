from __future__ import annotations  # To be removed in future python versions
from pathlib import Path
from dataclasses import dataclass, field
from typing import Callable
import math

import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

from mayavi.api import Engine
from mayavi.sources.api import ParametricSurface
from mayavi.modules.api import Surface
from mayavi import mlab

# (!!!) Somehow cv2 has to be imported after mayavi (?!) 
import cv2

from vartools.states import Pose
from vartools.dynamics import ConstantValue
from vartools.dynamics import LinearSystem
from vartools.colors import hex_to_rgba, hex_to_rgba_float
from vartools.linalg import get_orthogonal_basis

from dynamic_obstacle_avoidance.obstacles import Obstacle
from dynamic_obstacle_avoidance.obstacles import CuboidXd as Cuboid
from dynamic_obstacle_avoidance.obstacles import EllipseWithAxes as Ellipse

#from nonlinear_avoidance.multi_body_franka_obs import create_3d_franka_obs2
#from nonlinear_avoidance.multi_body_human import create_3d_human
from nonlinear_avoidance.multi_obs_env import create_3d_human
from nonlinear_avoidance.multi_body_human import (
    transform_from_multibodyobstacle_to_multiobstacle,
)
from nonlinear_avoidance.multi_obstacle_avoider import MultiObstacleAvoider
from nonlinear_avoidance.multi_obstacle_avoider import MultiObstacleContainer
from nonlinear_avoidance.dynamics.spiral_dynamics import SpiralingDynamics3D
from nonlinear_avoidance.dynamics.spiral_dynamics import SpiralingAttractorDynamics3D

from nonlinear_avoidance.nonlinear_rotation_avoider import (
    ConvergenceDynamicsWithoutSingularity,
)

from scripts.three_dimensional.visualizer3d import CubeVisualizer

class Visualization3D:
    dimension = 3
    obstacle_color = hex_to_rgba_float("724545ff")

    figsize = (1920, 1080)

    def __init__(self):
        self.engine = Engine()
        self.engine.start()
        # self.scene = self.engine.new_scene(
        #     size=self.figsize,
        #     bgcolor=(1, 1, 1),
        #     fgcolor=(0.5, 0.5, 0.5),
        # )
        # self.scene.scene.disable_render = False  # for speed
        # self.scene.background = (255, 255, 255)
        # self.scene.background = (1, 1, 1)

        # self.obstacle_color = np.array(self.obstacle_color)
        # self.obstacle_color[-1] = 0.5
        self.scene = mlab.figure(
            size=self.figsize, bgcolor=(1, 1, 1), fgcolor=(0.5, 0.5, 0.5)
        )

    def plot_obstacles(self, obstacles):
        for ii, obs in enumerate(obstacles):
            if isinstance(obs, Ellipse):
                # plot_ellipse_3d(
                #     scene=scene, center=obs.center_position, axes_length=obs.axes_length
                # )

                source = ParametricSurface()
                source.function = "ellipsoid"
                self.engine.add_source(source)
                surface = Surface()
                source.add_module(surface)

                actor = surface.actor  # mayavi actor, actor.actor is tvtk actor
                # defaults to 0 for some reason, ah don't need it, turn off scalar visibility instead
                # actor.property.ambient = 1
                actor.property.opacity = self.obstacle_color[-1]
                actor.property.color = tuple(self.obstacle_color[:3])

                # Colour ellipses by their scalar indices into colour map
                actor.mapper.scalar_visibility = False

                # gets rid of weird rendering artifact when opacity is < 1
                actor.property.backface_culling = True
                actor.property.specular = 0.1

                # actor.property.frontface_culling = True
                if obs.pose.orientation is not None:
                    # orientation = obs.pose.orientation.as_euler("xyz")
                    # orientation = obs.pose.orientation.as_euler("xyz", degrees=True)
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
                    actor.actor.orientation = orientation
                    # breakpoint()

                # actor.actor.origin = obs.center_position
                # actor.actor.position = np.zeros(self.dimension)
                actor.actor.origin = np.zeros(self.dimension)
                actor.actor.position = obs.center_position
                #print('3d position', actor.actor.position)
                actor.actor.scale = obs.axes_length * 0.5
                #print('3d scale', actor.actor.scale)
                actor.enable_texture = True

            if isinstance(obs, Cuboid):
                visualizer = CubeVisualizer(obs)
                visualizer.draw_cube()

def set_view():
    mlab.view(
        150.15633889829527,
        68.76031172885509,
        4.135728793575641,
        #(100,100,100),  
        #distance=5.004231840226419,
        #focalpoint=(-0.32913308, 0.38534346, -0.14484502),
    )
    # mlab.view(
    #     1,1,1,
    #     (0.66062227, 0.6689306, -0.00697224),
    # )
    mlab.background = (255, 255, 255)


print("Import main")


class MayaviAnimator:
    def __init__(
        self, it_max: int = 300, delta_time: float = 0.005, filename: str = "animation"
    ) -> None:
        self.it_max = it_max
        self.delta_time = delta_time

        self.filename = filename
        self.figuretype = ".png"

        self.save_to_file = True

        self.main_folder = Path("figures")
        self.image_folder = Path("animation")

        # self.leading_zeros = math.ceil(math.log10(self.it_max + 1))
        self.leading_zeros = 4

    def run(self):
        # mlab.clf(figure=None)
        # self.visualizer.plot_obstacles(self.human_obstacle_3d)

        for ii in range(self.it_max):
            self.update_step(ii)
            self.update_human(ii)
            #self.update_view(ii)

            if self.save_to_file:
                mlab.savefig(
                    str(
                        self.main_folder
                        / self.image_folder
                        / (
                            self.filename
                            + str(ii).zfill(self.leading_zeros)
                            + self.figuretype
                        )
                    ),
                    magnification=2,
                )

            if not ii % 10:
                print(f"it={ii}")

        if self.save_to_file:
            self.save_animation()

    def update_view(self, ii):
        posangle_range = [-2.31, 1.0]
        posdist = 5.58
        z_value = 0.3

        # posangle_range = [-2.30, -2.30]
        # posdist = 10

        azimuth_range = [-133.0, 42.0]

        elevation = 80
        distance = 4.5

        # azimuth_range = [-133, 133.0]
        # distance = 10.5

        delta_posangle = (posangle_range[1] - posangle_range[0]) / self.it_max
        delta_azimuth = (azimuth_range[1] - azimuth_range[0]) / self.it_max

        new_angle = posangle_range[0] + ii * delta_posangle
        pos_cam = np.array(
            [np.cos(new_angle) * posdist, np.sin(new_angle) * posdist, z_value]
        )
        mlab.move(pos_cam)

        new_azimuth = azimuth_range[0] + ii * delta_azimuth
        mlab.view(azimuth=new_azimuth, elevation=elevation, distance=distance)

    def save_animation(self):
        folder = str(self.main_folder / self.image_folder)
        os.system(
            f"ffmpeg "
            # + f"-framerate {1 / self.delta_time} "
            + "-framerate 20 "
            # + "-pattern_type glob "
            + f"-i ./{folder}/{self.filename}%{self.leading_zeros}d.png -vcodec mpeg4 "
            + f"-y ./figures/{self.filename}.mp4"
        )

    def setup(self, n_grid=5):
        dimension = 3

        # Trajectory integration
        x_range = [-0.8, 0.8]
        y_value = -3.5
        # z_range = [-0.6, 0.6]
        z_range = [-0.7, 0.3]

        yv = y_value * np.ones(n_grid * n_grid)
        xv, zv = np.meshgrid(
            np.linspace(x_range[0], x_range[1], n_grid),
            np.linspace(z_range[0], z_range[1], n_grid),
        )
        start_positions = np.vstack((xv.flatten(), yv.flatten(), zv.flatten()))
        self.n_traj = start_positions.shape[1]
        # self.trajectories shape is (3,301,25), 25 trajectories, 301 time steps and 3 dimensions
        self.trajectories = np.zeros((dimension, self.it_max + 1, self.n_traj))
        self.trajectories[:, 0, :] = start_positions

        cm = plt.get_cmap("gist_rainbow")
        self.color_list = [cm(1.0 * cc / self.n_traj) for cc in range(self.n_traj)]

############################################################################################################
        # Create Scene
        self.human_obstacle_3d = create_3d_human()
        #self.human_obstacle_3d = create_3d_franka_obs2()
        dynamics = LinearSystem(attractor_position=np.array([1, -2, 0.0]))

        transformed_human = transform_from_multibodyobstacle_to_multiobstacle(
            self.human_obstacle_3d
        )
        self.container = MultiObstacleContainer()
        self.container.append(transformed_human)

        self.avoider = MultiObstacleAvoider.create_with_convergence_dynamics(
            obstacle_container=self.container,
            initial_dynamics=dynamics,
            # reference_dynamics=linearsystem(attractor_position=dynamics.attractor_position),
            create_convergence_dynamics=True,
            convergence_radius=0.55 * math.pi,
            smooth_continuation_power=0.7,
        )
##########################################################################################################################
        self.visualizer = Visualization3D()

        self.dynamic_human = False

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

        amplitude_leg1 = 0.08
        frequency_leg1 = 0.2
        idx = self.human_obstacle_3d.get_obstacle_id_from_name("upperarm1")
        obstacle = self.human_obstacle_3d[idx]
        rotation = Rotation.from_euler(
            "y", amplitude_leg1 * np.sin(ii * frequency_leg1)
        )
        obstacle.orientation = obstacle.orientation * rotation

        amplitude_leg1 = 0.12
        frequency_leg1 = 0.2
        idx = self.human_obstacle_3d.get_obstacle_id_from_name("lowerarm1")
        obstacle = self.human_obstacle_3d[idx]
        rotation = Rotation.from_euler(
            "y", amplitude_leg1 * np.sin(ii * frequency_leg1)
        )
        obstacle.orientation = obstacle.orientation * rotation

        amplitude_leg1 = -0.05
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

        # delta_position = reference_point_updated - reference_point_original
        # obstacle.center_position = obstacle.center_position + delta_position
        # print(delta_position)
        # if np.linalg.norm(delta_position):
        #     breakpoint()

        # obstacle.center_position = obstacle.center_position - delta_position

        # Update in 'parallel' obstacle -> this might not actually be needed as they're all
        #
        # obstacle_parallel = self.container.get_obstacle_tree(0).get_component(idx)
        # obstacle_parallel.orientation = obstacle.orientation
        # obstacle_parallel.position = obstacle.orientation

        # self.delta_time
        # pass

    def update_step(self, ii: int) -> None:
        for it_traj in range(self.n_traj):
            velocity = self.avoider.evaluate_sequence(self.trajectories[:, ii, it_traj])
            self.trajectories[:, ii + 1, it_traj] = (
                velocity * self.delta_time + self.trajectories[:, ii, it_traj]
            )
            print('position', self.trajectories[:, ii + 1, it_traj])
        

        # Clear
        mlab.clf(figure=None)
        self.visualizer.plot_obstacles(self.human_obstacle_3d)

        # Plot
        for it_traj in range(self.n_traj):
            mlab.plot3d(
                self.trajectories[0, : ii + 2, it_traj],
                self.trajectories[1, : ii + 2, it_traj],
                self.trajectories[2, : ii + 2, it_traj],
                color=self.color_list[it_traj][:3],
                tube_radius=0.01,
            )

            mlab.points3d(
                self.trajectories[0, ii + 1, it_traj],
                self.trajectories[1, ii + 1, it_traj],
                self.trajectories[2, ii + 1, it_traj],
                color=(0, 0, 0),
                scale_factor=0.06,
            )

        #if self.dynamic_human:
        #    self.update_human(ii)

        set_view()


def main_animation():
    animator = MayaviAnimator()

    animator.setup()
    # animator.run()
    animator.save_animation()


def main_animation_dynamic():
    #animator = MayaviAnimator(filename="avoidance_around_dynamic_human", it_max=20)
    animator = MayaviAnimator(filename="avoidance_around_dynamic_human")

    # animator.setup(n_grid=1)
    animator.setup()

    animator.dynamic_human = True
    animator.run()

    animator.save_animation()
    #animator.save_animation_using_cv2()


if (__name__) == "__main__":
    figtype = ".jpeg"
    mlab.close(all=True)

    # main(savefig=True)
    # main_animation()

    main_animation_dynamic()