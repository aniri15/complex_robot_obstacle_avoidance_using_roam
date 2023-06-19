"""
Library for the Rotation (Modulation Imitation) of Linear Systems
"""
# Author: Lukas Huber
# GitHub: hubernikus
# Created: 2022-01-20

import json
from dataclasses import dataclass

import math
from typing import Optional

import numpy as np
from numpy import linalg as LA

import matplotlib.pyplot as plt

from vartools.states import Pose
from nonlinear_avoidance.dynamics.circular_dynamics import (
    # CircularRotationDynamics,
    SimpleCircularDynamics,
)


# IDEAS: Metrics to use
# - Distance to circle (desired path)
# - Average acceleration
# - Std acceleration
# - # of switching
# - # of stuck in local minima
# - Deviation / error to desired velocity


def normalize_velocities(velocities):
    vel_norm = LA.norm(velocities, axis=0)

    ind_nonzero = vel_norm > 0
    if not any(ind_nonzero):
        raise ValueError()

    velocities = velocities[:, ind_nonzero] / np.tile(
        vel_norm[ind_nonzero], (velocities.shape[0], 1)
    )
    return velocities


def mean_squared_error_to_path(trajectory, center_position, radius):
    # Assumption of circular dynamics
    traj_centered = trajectory - np.tile(center_position, (trajectory.shape[1], 1)).T
    err_squared = abs(np.sum(traj_centered**2, axis=0) - radius**2)

    return np.mean(err_squared)


def mean_squared_acceleration(trajectory, delta_time):
    velocities = (trajectory[:, 1:] - trajectory[:, :-1]) / delta_time
    acceleration = (velocities[:, 1:] - velocities[:, :-1]) / delta_time

    acceleration_squared = np.sum(acceleration**2, axis=0)
    return np.mean(acceleration_squared)


def dot_product_acceleration(trajectory, delta_time):
    velocities = (trajectory[:, 1:] - trajectory[:, :-1]) / delta_time
    velocities = normalize_velocities(velocities)

    velocity_dotprod = np.sum(velocities[:, 1:] * velocities[:, :-1], axis=0)
    # breakpoint()
    return np.mean(velocity_dotprod)
    # velocity_dot_prod_fact = (1.0 - velocity_dotprod) / 2.0
    # return np.mean(velocity_dot_prod_fact**2)


def mean_squared_velocity_deviation(trajectory, dynamic_functor, delta_time):
    velocities = (trajectory[:, 1:] - trajectory[:, :-1]) / delta_time
    velocities = normalize_velocities(velocities)

    init_velocities = np.zeros_like(velocities)
    for ii in range(init_velocities.shape[1]):
        init_velocities[:, ii] = dynamic_functor(trajectory[:, ii])

    delta_vel = velocities - init_velocities
    deviations = np.sum(delta_vel**2, axis=0)

    return np.mean(deviations)


def dot_product_velocity_deviation(trajectory, dynamic_functor, delta_time):
    velocities = (trajectory[:, 1:] - trajectory[:, :-1]) / delta_time
    velocities = normalize_velocities(velocities)

    init_velocities = np.zeros_like(velocities)
    for ii in range(init_velocities.shape[1]):
        init_vel = dynamic_functor(trajectory[:, ii])
        if not (init_norm := LA.norm(init_vel)):
            continue
        init_velocities[:, ii] = init_vel / init_norm

    velocity_dotprod = np.sum(velocities * init_velocities, axis=0)
    return np.mean(velocity_dotprod)
    # velocity_dotprod_factor = (1.0 - velocity_dotprod) / 2.0
    # return np.mean(velocity_dotprod_factor**2)


@dataclass
class TrajectoryEvaluator:
    data_folder: "str"
    # data_path: "str" = "/home/lukas/Code/nonlinear_avoidance/comparison/data"
    data_path: "str" = "/home/lukas/Code/nonlinear_obstacle_avoidance/nonlinear_avoidance/comparison/data"

    n_runs: int = 0

    dist_to_path: float = 0

    squared_acceleration: float = 0
    squared_acceleration_std: float = 0
    squared_error_velocity: float = 0
    squared_error_velocity_std: float = 0

    dotprod_err_velocity: float = 0
    dotprod_err_velocity_std: float = 0
    dotprod_acceleration: float = 0
    dotprod_acceleration_std: float = 0

    n_local_minima: int = 0
    n_converged: int = 0

    def run(self):
        with open(
            os.path.join(self.data_path, "..", "comparison_parameters.json")
        ) as user_file:
            simulation_parameters = json.load(user_file)

        delta_time = simulation_parameters["delta_time"]
        it_max = simulation_parameters["it_max"]

        initial_dynamics = SimpleCircularDynamics(
            dimension=2, pose=Pose.create_trivial(dimension=2)
        )

        datafolder_path = os.path.join(self.data_path, self.data_folder)
        files_list = os.listdir(datafolder_path)

        if self.n_runs <= 0:
            self.n_runs = len(files_list)
        else:
            files_list = files_list[: self.n_runs]

        print(f"Evaluating #{self.n_runs} runs.")

        self.dist_to_path_list = np.zeros(self.n_runs)
        self.squared_acceleration_list = np.zeros(self.n_runs)
        self.squared_error_velocity_list = np.zeros(self.n_runs)

        self.dotprod_err_velocity_list = np.zeros(self.n_runs)
        self.dotprod_acceleration_list = np.zeros(self.n_runs)

        for ii, filename in enumerate(files_list):
            trajectory = np.loadtxt(
                os.path.join(datafolder_path, filename),
                delimiter=",",
                dtype=float,
                skiprows=0,
            )

            if not len(trajectory):
                warnings.warn("Empty trajectory file.")
                continue

            trajectory = trajectory.T

            if trajectory.shape[1] < it_max:
                self.n_local_minima += 1
            else:
                self.n_converged += 1

            self.dist_to_path_list[ii] = mean_squared_error_to_path(
                trajectory, center_position=np.zeros(2), radius=2.0
            )

            self.squared_error_velocity_list[ii] = mean_squared_velocity_deviation(
                trajectory, initial_dynamics.evaluate, delta_time
            )

            self.dotprod_err_velocity_list[ii] = dot_product_velocity_deviation(
                trajectory, initial_dynamics.evaluate, delta_time
            )

            self.squared_acceleration_list[ii] = mean_squared_acceleration(
                trajectory, delta_time
            )

            self.dotprod_acceleration_list[ii] = dot_product_acceleration(
                trajectory, delta_time
            )

        self.dist_to_path = np.mean(self.dist_to_path_list)
        self.dist_to_path_std = np.std(self.dist_to_path_list)

        self.squared_acceleration = np.mean(self.squared_acceleration_list)
        self.squared_acceleration_std = np.std(self.squared_acceleration_list)

        self.squared_error_velocity = np.mean(self.squared_error_velocity_list)
        self.squared_error_velocity_std = np.std(self.squared_error_velocity_list)

        self.dotprod_err_velocity = np.mean(self.dotprod_err_velocity_list)
        self.dotprod_err_velocity_std = np.std(self.dotprod_err_velocity_list)

        self.nics_err_velocity_list = (1.0 - self.dotprod_err_velocity_list) * 0.5
        self.nics_err_velocity = np.mean(self.nics_err_velocity_list)
        self.nics_err_velocity_std = np.std(self.nics_err_velocity_list)

        self.nics_err_velocity_list = self.dotprod_err_velocity_list
        self.nics_err_velocity = np.mean(self.dotprod_err_velocity_list)

        self.dotprod_acceleration = np.mean(self.dotprod_acceleration_list)
        self.dotprod_acceleration_std = np.std(self.dotprod_acceleration_list)

        self.nics_acceleration_list = (1.0 - self.dotprod_acceleration_list) * 0.5
        self.nics_acceleration = np.mean(self.nics_acceleration_list)
        self.nics_acceleration_std = np.std(self.nics_acceleration_list)


def print_table(evaluation_list):
    value = [ee.data_folder for ee in evaluation_list]
    print(" & ".join(["Name"] + value) + " \\\\ \hline")

    value = [
        f"{ee.n_converged / ee.n_runs * 100:.0f}" + "\\%" for ee in evaluation_list
    ]
    print(" & ".join(["$N^c$"] + value) + " \\\\ \hline")

    value = [
        f"{ee.n_local_minima / ee.n_runs * 100:.0f}" + "\\%" for ee in evaluation_list
    ]
    print(" & ".join(["$N^m$"] + value) + " \\\\ \hline")

    value = [
        f"{ee.dist_to_path:.2f}" + " $\\pm$ " + f"{ee.dist_to_path_std:.2f}"
        for ee in evaluation_list
    ]
    # std = [f"{ee.dist_to_path_std:.2f}" for ee in evaluation_list]
    print(" & ".join(["$\\Delta R^2$"] + value) + " \\\\ \hline")

    value = [
        f"{ee.squared_error_velocity:.2f}"
        + " $\\pm$ "
        + f"{ee.squared_error_velocity_std:.2f}"
        for ee in evaluation_list
    ]
    print(" & ".join(["$\Delta v$"] + value) + " \\\\ \hline")

    # value = [f"{(1.0 - ee.dotprod_err_velocity) * 0.5:.2f}" for ee in evaluation_list]
    value = [
        f"{ee.nics_err_velocity:.2f}" + " $\\pm$ " + f"{ee.nics_err_velocity_std:.2f}"
        for ee in evaluation_list
    ]
    print(" & ".join(["$\\langle v \\rangle $"] + value) + " \\\\ \hline")

    value = [
        f"{ee.squared_acceleration:.2f}"
        + " $\\pm$ "
        + f"{ee.squared_acceleration_std:.2f}"
        for ee in evaluation_list
    ]
    print(" & ".join(["$a$"] + value) + " \\\\ \hline")

    # value = [
    #     f"{(1.0 - ee.dotprod_acceleration) * 0.5 * 1e4:.2f}" for ee in evaluation_list
    # ]
    value = [
        f"{ee.nics_acceleration*1e4:.2f}"
        + " $\\pm$ "
        + f"{ee.nics_acceleration_std*1e4:.2f}"
        for ee in evaluation_list
    ]
    print(" & ".join(["$\\langle a \\rangle [1e-4 m/s]$"] + value) + " \\\\ \hline")


if (__name__) == "__main__":
    # if False:
    if True:
        # n_runs = 5
        n_runs = -1  # All runs...

        nonlinear_evaluation = TrajectoryEvaluator(
            n_runs=n_runs, data_folder="nonlinear_avoidance"
        )
        nonlinear_evaluation.run()

        modulation_evaluation = TrajectoryEvaluator(
            n_runs=n_runs, data_folder="modulation_avoidance"
        )
        modulation_evaluation.run()

        gfield_evaluation = TrajectoryEvaluator(
            n_runs=n_runs, data_folder="guiding_field"
        )
        gfield_evaluation.run()

        # gfield_evaluation = TrajectoryEvaluator(
        #     n_runs=n_runs, data_folder="guiding_field"
        # )
        # gfield_evaluation.run()

        original_evaluation = TrajectoryEvaluator(
            n_runs=n_runs, data_folder="original_trajectories"
        )
        original_evaluation.run()
    else:
        print("[INFO] We are currently using the evaluator from the memory.")

    print_table(
        [
            nonlinear_evaluation,
            modulation_evaluation,
            gfield_evaluation,
            original_evaluation,
        ]
    )
