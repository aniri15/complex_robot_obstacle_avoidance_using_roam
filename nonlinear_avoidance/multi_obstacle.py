import copy
from typing import Optional
from dataclasses import dataclass, field

import networkx as nx

import numpy as np
from numpy import typing as npt

from vartools.states import Pose, Twist
from dynamic_obstacle_avoidance.obstacles import Obstacle
from nonlinear_avoidance.geometry import get_intersection_of_obstacles

"""function summary for MultiObstacle:
1. get pose
2. update pose
3. update deformation
4. is collision free
5. get gamma
6. get gamma except components
7. get parent idx
8. get component
9. get root
10. set root
11. add component
"""
@dataclass
class MultiObstacle:
    _pose: Pose
    margin_absolut: float = 0

    _graph: nx.DiGraph = field(default_factory=lambda: nx.DiGraph())
    _local_poses: list[Pose] = field(default_factory=list)
    _obstacle_list: list[Obstacle] = field(default_factory=list)

    _root_idx: int = 0

    twist: Optional[Twist] = None

    @property
    def dimension(self) -> int:
        return self._pose.dimension

    @property
    def n_components(self) -> int:
        return len(self._obstacle_list)

    @property
    def root_idx(self) -> int:
        return self._root_idx

    @property
    def pose(self) -> Pose:
        """For compatibalitiy with Obstacle"""
        return self._pose

    @property
    def linear_velocity(self) -> npt.ArrayLike:
        if self.twist is None:
            return np.zeros(self.dimension)
        else:
            return self.twist.linear_velocity

    @linear_velocity.setter
    def linear_velocity(self, value: npt.ArrayLike) -> None:
        if self.twist is None:
            self.twist = Twist(value)
        else:
            self.twist.linear = value

    def __iter__(self):
        return iter(self._obstacle_list)

    def __getitem__(self, idx: int) -> Obstacle:
        return self._obstacle_list[idx]

    def __len__(self) -> int:
        return len(self._obstacle_list)

    def get_pose(self) -> Pose:
        """Returns a (copy) of the pose."""
        return copy.deepcopy(self._pose)

    def update_pose(self, new_pose: Pose) -> None:
        self._pose = new_pose
        for pose, obs in zip(self._local_poses, self._obstacle_list):
            obs.pose = self._pose.transform_pose_from_relative(pose)

    def update_deformation(self, step_size: float) -> None:
        if not hasattr(self, "deformation_rate"):
            return

        factor = 1 + self.deformation_rate * step_size
        for pose, obs in zip(self._local_poses, self._obstacle_list):
            pose.position = factor * pose.position
            obs.axes_length = factor * obs.axes_length

            obs.pose = self._pose.transform_pose_from_relative(pose)

    def is_collision_free(self, position, in_global_frame: bool = True) -> bool:
        return self.get_gamma(position, in_global_frame) > 1.0

    def get_gamma(self, position, in_global_frame: bool = True) -> float:
        if not in_global_frame:
            position = self._pose.transform_pose_from_relative(position)

        gammas = [
            obs.get_gamma(position, in_global_frame=True) for obs in self._obstacle_list
        ]
        return min(gammas)

    def get_gamma_except_components(
        self,
        position: np.ndarray,
        excluded_components: list[int],
        in_global_frame: bool = True,
    ) -> float:
        if not in_global_frame:
            position = self._pose.transform_pose_from_relative(position)

        gammas = []
        for ii, obs in enumerate(self._obstacle_list):
            if ii in excluded_components:
                continue

            gammas.append(obs.get_gamma(position, in_global_frame=True))

        if not len(gammas):
            raise ValueError("No components left to evaluate gamma.")

        return min(gammas)

    def get_parent_idx(self, idx_obs: int) -> Optional[int]:
        if idx_obs == self.root_idx:
            return None
        else:
            return list(self._graph.predecessors(idx_obs))[0]

    def get_component(self, idx_obs: int) -> Obstacle:
        return self._obstacle_list[idx_obs]

    def get_root(self) -> Obstacle:
        return self._obstacle_list[self._root_idx]

    def set_root(self, obstacle: Obstacle) -> int:
        self._local_poses.append(obstacle.pose)
        obstacle.pose = self._pose.transform_pose_from_relative(self._local_poses[-1])

        self._obstacle_list.append(obstacle)
        self._root_idx = 0  # Obstacle ID
        self._graph.add_node(
            self._root_idx, references_children=[], indeces_children=[]
        )
        return self._root_idx

    def add_component(
        self,
        obstacle: Obstacle,
        parent_ind: int,
        reference_position: Optional[npt.ArrayLike] = None,
    ) -> int:
        """Create and add an obstacle container in the local frame of reference.
        Returns component id"""
        if reference_position is None:
            global_reference = get_intersection_of_obstacles(
                obstacle, self.get_component(parent_ind)
            )
            if global_reference is None:
                raise ValueError("No intersection found.")

            reference_position = obstacle.pose.transform_position_to_relative(
                global_reference
            )
        else:
            reference_position = np.array(reference_position)

        obstacle.set_reference_point(reference_position, in_global_frame=False)

        new_id = len(self._obstacle_list)
        # Put obstacle to 'global' frame, but store local pose
        self._local_poses.append(obstacle.pose)
        obstacle.pose = self._pose.transform_pose_from_relative(self._local_poses[-1])
        self._obstacle_list.append(obstacle)

        self._graph.add_node(
            new_id,
            local_reference=reference_position,
            indeces_children=[],
            references_children=[],
        )

        self._graph.nodes[parent_ind]["indeces_children"].append(new_id)
        self._graph.add_edge(parent_ind, new_id)

        return new_id

    # def update_obstacles(self, delta_time):
    #     # Update all positions of the moved obstacles
    #     for pose, obs in zip(self._local_poses, self._obstacle_list):
    #         obs.shape = self._pose.transform_pose_from_relative(pose)
