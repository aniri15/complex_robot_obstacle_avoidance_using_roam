#!/USSR/bin/python3
""" Create the rotation space which is so much needed. ... """
# Author: Lukas Huber
# Github: hubernikus
# Created: 2022-07-07

# Use python 3.10 [annotations / typematching]
from __future__ import annotations  # Not needed from python 3.10 onwards

import copy
import warnings
import math
from typing import Optional, Hashable
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

import numpy.typing as npt
from numpy import linalg as LA

import networkx as nx
import matplotlib.pyplot as plt

from vartools.linalg import get_orthogonal_basis

from nonlinear_avoidance.datatypes import Vector, VectorArray

NodeType = Hashable


def directional_vector_addition(
    vector1: Vector, vector2: Vector, weight: float
) -> Vector:
    if not (norm1 := np.linalg.norm(vector1)):
        return vector2 * weight

    if not (norm2 := np.linalg.norm(vector2)):
        return vector1 * (1 - weight)

    vector1 = vector1 / norm1
    vector2 = vector2 / norm2

    vector = VectorRotationXd.from_directions(vector1, vector2).rotate(vector1, weight)

    return vector * ((1 - weight) * norm1 + weight * norm2)


def rotate_direction(
    direction: Vector, base: VectorArray, rotation_angle: float
) -> Vector:
    """Returns the rotated of the input vector with respect to the base and rotation angle."""
    if not (dir_norm := LA.norm(direction)):
        # Zero vector can not be rotated
        return direction

    direction = direction / dir_norm

    dot_prods = np.dot(base.T, direction)
    angle = math.atan2(dot_prods[1], dot_prods[0]) + rotation_angle

    # Convert angle to the two basis-axis
    out_direction = math.cos(angle) * base[:, 0] + math.sin(angle) * base[:, 1]
    out_direction *= math.sqrt(sum(dot_prods**2))

    # Finally, add the orthogonal part (no effect in 2D, but important for higher dimensions)
    out_direction += direction - np.sum(dot_prods * base, axis=1)
    return out_direction * dir_norm


def rotate_array(
    directions: VectorArray,
    base: VectorArray,
    rotation_angle: float,
) -> VectorArray:
    """Rotate upper level base with respect to total."""
    dimension, n_dirs = directions.shape

    directions = directions / LA.norm(directions, axis=0)

    # Matrix dimensions: [2 x n_dirs ] <- [dimension x 2 ].T @ [dimension x n_dirs]
    dot_prods = np.dot(base.T, directions)
    angles = np.arctan2(dot_prods[1, :], dot_prods[0, :]) + rotation_angle

    # Compute output from rotation
    out_vectors = np.tile(base[:, 0], (n_dirs, 1)).T * np.tile(
        np.cos(angles), (dimension, 1)
    ) + np.tile(base[:, 1], (n_dirs, 1)).T * np.tile(np.sin(angles), (dimension, 1))
    out_vectors *= np.tile(np.sqrt(np.sum(dot_prods**2, axis=0)), (dimension, 1))

    # Finally, add the orthogonal part (no effect in 2D, but important for higher dimensions)
    out_vectors += directions - (base @ dot_prods)

    return out_vectors


@dataclass
class VectorRotationXd:
    """This approach allows successive modulation which can be added up.

    Attributes
    ----------
    base array of size [dimension x 2]: The (orthonormal) base constructed from the to
        input directions
    rotation_angle (float): The rotation angle resulting from the two input directions
    """

    base: VectorArray
    rotation_angle: float

    @classmethod
    def from_directions(cls, vec_init: Vector, vec_rot: Vector) -> Self:
        """Alternative constructor base on two input vectors which define the
        initialization."""

        # # Normalize both vectors
        vec_init = vec_init / LA.norm(vec_init)
        vec_rot = vec_rot / LA.norm(vec_rot)
        dot_prod = np.dot(vec_init, vec_rot)

        if dot_prod <= -1 - 1e-6:
            warnings.warn("(Close to) anti-parallel vectors.")

        if np.allclose(vec_init, vec_rot):
            # (Anti-)parallel vectors => calculate random perpendicular vector
            vec_perp = np.zeros(vec_init.shape)
            if not LA.norm(vec_init[:2]):
                vec_perp[0] = 1
            else:
                vec_perp[0] = vec_init[1]
                vec_perp[1] = vec_init[0] * (-1)
                vec_perp[:2] = vec_perp[:2] / LA.norm(vec_perp[:2])
        else:
            vec_perp = vec_rot - vec_init * dot_prod
            vec_perp = vec_perp / LA.norm(vec_perp)

        angle = np.arccos(min(max(dot_prod, -1), 1))
        return cls(base=np.array([vec_init, vec_perp]).T, rotation_angle=angle)

    # def __mult__(self, factor) -> VectorRotationXd:
    #     instance_copy = copy.deepcopy(self)
    #     instance_copy.rotation_angle = instance_copy.rotation_angle * factor
    #     return instance_copy

    @property
    def base0(self):
        return self.base[:, 0]

    @property
    def dimension(self):
        try:
            return self.base.shape[0]
        except AttributeError:
            warnings.warn("base has not been defined")
            return None

    def inv(self):
        """Returns the inverse of the proposed rotation."""
        new_instance = copy.deepcopy(self)
        new_instance.rotation_angle = (-1) * new_instance.rotation_angle
        return new_instance

    def get_first_vector(self) -> Vector:
        return self.base[:, 0]

    def get_second_vector(self) -> Vector:
        """Returns the second vector responsible for the rotation"""
        return rotate_direction(
            direction=self.base[:, 0],
            rotation_angle=self.rotation_angle,
            base=self.base,
        )

    def rotate(self, direction: Vector, rot_factor: float = 1) -> Vector:
        """Returns the rotated of the input vector with respect to the base and rotation angle
        rot_factor: factor gives information about extension of rotation"""
        return rotate_direction(
            direction=direction,
            rotation_angle=rot_factor * self.rotation_angle,
            base=self.base,
        )

    def rotate_sequence(
        self, sequence: VectorRotationSequence, rotation_factor: float = 1.0
    ) -> VectorRotationSequence:
        # Make sure to keep original
        sequence = copy.deepcopy(sequence)

        base_rotated = rotate_array(
            directions=sequence.basis_array.reshape(self.dimension, -1),
            base=self.base,
            rotation_angle=self.rotation_angle * rotation_factor,
        )

        sequence.basis_array = base_rotated.reshape(self.dimension, -1, 2)
        return sequence

    def rotate_vector_rotation(
        self, rotation: VectorRotationXd, rot_factor: float = 1
    ) -> VectorRotationXd:
        rotation = copy.deepcopy(rotation)
        rotation.base = rotate_array(
            directions=rotation.base,
            base=rotation.base,
            rotation_angle=rot_factor * self.rotation_angle,
        )
        return rotation

    def inverse_rotate(self, direction):
        return rotate_direction(
            direction=direction,
            rotation_angle=(-1) * self.rotation_angle,
            base=self.base,
        )


@dataclass
class VectorRotationSequence:
    """
    Vector-Rotation environment based on multiple vectors

    Attributes
    ----------
    vectors_array (np.array of shape [dimension x n_rotations + 1]):
        (storing) the inital array of vectors
    basis_array (numpy array of  shape [dimension x n_rotations x 2]):
        contains the basis of all rotations
    rotation_angles: The rotation between going from one to the next basis
    """

    # def __init__(self, vectors_array: np.ndarray) -> None:
    #     # Normalize
    #     vectors_array = vectors_array / LA.norm(vectors_array, axis=0)

    #     dot_prod = np.sum(vectors_array[:, 1:] * vectors_array[:, :-1], axis=0)

    #     if np.sum(dot_prod == (-1)):  # Any of the values
    #         raise ValueError("Antiparallel vectors.")

    #     # Evaluate basis and angles
    #     vec_perp = vectors_array[:, 1:] - vectors_array[:, :-1] * dot_prod
    #     vec_perp = vec_perp / LA.norm(vec_perp, axis=0)

    #     self.basis_array = np.stack((vectors_array[:, :-1], vec_perp), axis=2)
    #     self.rotation_angles = np.arccos(dot_prod)

    # def __init__(self, basis_array: np.ndarray, angles: np.ndarray) -> None:
    basis_array: np.ndarray
    rotation_angles: np.ndarray

    @classmethod
    def create_empty(cls, dimension: int) -> Self:
        return cls(np.zeros((dimension, 0, 2)), np.zeros(0))

    @classmethod
    def create_from_vector_array(cls, vectors_array: np.ndarray) -> Self:
        # 
        array_norm = LA.norm(vectors_array, axis=0)
        if np.any(np.isclose(array_norm, 0)):
            raise ValueError("Zero vector in sequence.")

        vectors_array = vectors_array / array_norm
        dot_prod = np.sum(vectors_array[:, 1:] * vectors_array[:, :-1], axis=0)

        if np.sum(dot_prod == (-1)):  # Any of the values
            raise ValueError("Antiparallel vectors.")

        # Evaluate basis and angles
        ind_nonzero = np.zeros(dot_prod.shape[0], dtype=bool)
        for ii in range(ind_nonzero.shape[0]):
            ind_nonzero[ii] = not np.allclose(
                vectors_array[:, ii], vectors_array[:, ii + 1]
            )
        vecs_perp = vectors_array[:, 1:] - vectors_array[:, :-1] * dot_prod
        vecs_perp[:, ind_nonzero] = vecs_perp[:, ind_nonzero] / LA.norm(
            vecs_perp[:, ind_nonzero], axis=0
        )

        for ii in np.arange(ind_nonzero.shape[0])[np.logical_not(ind_nonzero)]:
            # (Anti-)parallel vectors => calculate random perpendicular vector
            vec_init = vectors_array[:, ii]
            vec_perp = np.zeros(vec_init.shape)
            if not LA.norm(vec_init[:2]):
                vec_perp[0] = 1
            else:
                vec_perp[0] = vec_init[1]
                vec_perp[1] = vec_init[0] * (-1)
                vec_perp[:2] = vec_perp[:2] / LA.norm(vec_perp[:2])

            vecs_perp[:, ii] = vec_perp

        angles = np.zeros(ind_nonzero.shape[0])
        angles[ind_nonzero] = np.arccos(np.maximum(dot_prod[ind_nonzero], -1.0))

        return cls(np.stack((vectors_array[:, :-1], vecs_perp), axis=2), angles)

    @property
    def dimension(self):
        return self.basis_array.shape[0]

    @property
    def n_rotations(self):
        return self.basis_array.shape[1]

    def base(self) -> Vector:
        return self.basis_array[:, [0, -1]]

    def append_from_direction(self, direction: Vector) -> None:
        rotation = VectorRotationXd.from_directions(
            self.basis_array[:, -1, -1], direction
        )
        self.append_from_base_and_angle(rotation.base, rotation.rotation_angle)

    def append_from_base_and_angle(self, base0: np.ndarray, angle: float) -> None:
        self.rotation_angles = np.append(self.rotation_angles, angle)
        self.basis_array = np.append(
            self.basis_array, base0.reshape(self.dimension, 1, 2), axis=1
        )

    def push_root_from_base_and_angle(self, base0: np.ndarray, angle: float) -> None:
        self.rotation_angles = np.append(angle, self.rotation_angles)
        self.basis_array = np.append(
            base0.reshape(self.dimension, 1, 2), self.basis_array, axis=1
        )

    def append_from_rotation(self, rotation: VectorRotationXd) -> None:
        raise NotImplementedError()

    def get_end_vector(self) -> Vector:
        final_rotation = VectorRotationXd(
            base=self.basis_array[:, -1, :],
            rotation_angle=self.rotation_angles[-1],
        )
        return final_rotation.rotate(self.basis_array[:, -1, 0])

    def rotate(self, direction: Vector, rot_factor: float = 1) -> Vector:
        """Rotate over the whole length of the vector."""
        weights = np.zeros(self.n_rotations)
        weights[-1] = rot_factor
        return self.rotate_weighted(direction, weights=weights)

    def rotate_weighted(self, direction: Vector, weights: npt.ArrayLike) -> Vector:
        """
        Returns the rotated direction vector with repsect to the (rotation-)weights

        weights (list of floats (>=0) with length [self.n_rotations]): indicates fraction
        of each rotation which is applied.
        """
        if weights is None:
            raise NotImplementedError("Argument needed.")

        # Starting at the root
        cumulated_weights = np.cumsum(weights[::-1])[::-1]

        if not math.isclose(cumulated_weights[0], 1):
            warnings.warn("Weights are summing up to more than 1.")

        temp_base = np.copy(self.basis_array)
        temp_angle = self.rotation_angles * cumulated_weights

        # Update the basis of rotation weights from top-to-bottom
        # by rotating upper level base with respect to total
        for ii in reversed(range(self.n_rotations - 1)):
            temp_base[:, (ii + 1) :, :] = rotate_array(
                directions=temp_base[:, (ii + 1) :, :].reshape(self.dimension, -1),
                base=temp_base[:, ii, :],
                rotation_angle=self.rotation_angles[ii] * (1 - cumulated_weights[ii]),
            ).reshape(self.dimension, -1, 2)

        # Finally: rotate from bottom-to-top
        for ii in range(self.n_rotations):
            direction = rotate_direction(
                direction=direction,
                rotation_angle=temp_angle[ii],
                base=temp_base[:, ii, :],
            )
        return direction


class VectorRotationTree:
    """
    VectorRotation but originating structured in a tree

    Positive node number reference the corresponding reference-id
    (negative numbers are used for the vectors)
    """

    # Major refactoring for speed up needed, following things should be addressed
    # TODO: what happens if an obstacle is at angle 'pi'?
    # TODO: allow evaluating adding / removing 'sub-trees' / 'sub-sequences'
    # TODO: Allow obtaining reduced-graphs
    # TODO: reduce reliance on dicitionaries (?!)
    # TODO: reduce graph usage, it could be done with a simple linked list faster (?)

    # as it might happend at  zero-level
    # if it's not the last-branch this could probably be extended by 'jumping' a node (?)

    # Maximum level a tree can reach (this is used to limit loops)
    maximum_level: int = 100

    def __init__(
        self, root_idx: Optional[int] = None, root_direction: Optional[Vector] = None
    ) -> None:
        self._graph = nx.DiGraph()

        if root_idx is not None and root_direction is not None:
            self.set_root(root_idx, root_direction)

    def set_root_orientation(
        self, root_id: NodeType, orientation: VectorRotationXd
    ) -> None:
        self._graph.add_node(
            root_id,
            level=0,
            direction=orientation.base[:, 0],
            weight=0,
            orientation=orientation,
        )

        self._root_idx = root_id
        self._dimension = orientation.dimension

    def set_root(self, root_idx: NodeType, direction: Vector) -> None:
        # To easier find the root again (!)
        vector_orientation = VectorRotationXd.from_directions(direction, direction)
        self.set_root_orientation(root_idx, vector_orientation)

    @property
    def root(self) -> NodeType:
        return self._graph.nodes(self._root_idx)

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def graph(self):
        # rename to _G (?)
        return self._graph

    def get_parent(self, node_id: NodeType) -> NodeType:
        """Returns the (unique) parent."""
        return self._graph.predecessors(node_id)[0]

    def get_children(self, node_id: NodeType) -> list[NodeType]:
        """Returns all the children of an input node."""
        return self._graph.successors(node_id)

    def add_node_orientation(
        self, node_id: NodeType, orientation: VectorRotationXd, parent_id: NodeType
    ) -> None:
        # TODO: check consistency (!)
        direction = orientation.rotate(orientation.base[:, 0])
        self.add_node(node_id=node_id, direction=direction, parent_id=parent_id)

    def add_node(
        self,
        node_id: NodeType,
        direction: Optional[Vector] = None,
        parent_id: Optional[NodeType] = None,
        child_id: Optional[NodeType] = None,
        level: Optional[int] = None,
    ) -> None:
        if not (dir_norm := np.linalg.norm(direction)):
            raise ValueError("Zero direction cannot be interpreted.")
        direction = direction / dir_norm

        if node_id in self.graph.nodes():
            raise ValueError(f"Node with id: {node_id} already exists.")

        if parent_id is not None:
            self._graph.add_edge(
                parent_id,
                node_id,
            )
            level = self._graph.nodes[parent_id]["level"] + 1

        elif child_id is not None:
            self._graph.add_edge(node_id, child_id)
            self.set_direction(direction, node_id)

        elif level is None:
            raise ValueError(
                "Argument 'level' is needed, if no parent or child is provided"
            )

        if level >= self.maximum_level:
            raise ValueError(f"Exceeding  maximum level {level} of direction-tree.")

        self._graph.add_node(
            node_id,
            level=level,
            direction=None,
            weight=0,
            orientation=None,
        )

        if np.any(np.isnan(direction)):
            raise ValueError("Faulty input.")

        if direction is not None:
            self.set_direction(node_id, direction)

        # TODO: what happens when you overwrite a node (?)

    def draw1(self):
        # show DiGraph structure
        options = {
            'node_color': 'blue',
            'node_size': 100,
            'width': 3,
            'arrowstyle': '-|>',
            'arrowsize': 12,
        }
        #nx.draw_networkx(self._graph, arrows=True,**options)
        nx.draw(self._graph,with_labels=True)
        #plt.draw()
        plt.show()

    def draw(self):
        # Verify graph is not empty
        if not self._graph.nodes or not self._graph.edges:
            print("Graph is empty.")
            return
        
        # Create custom labels for the nodes
        node_labels = {
            node: f"Obs: {node.obstacle}, Comp: {node.component}, Level: {node.relative_level}"
            for node in self._graph.nodes
        }

        # Relabel the graph for visualization
        labeled_graph = nx.relabel_nodes(self._graph, node_labels)

        
        # Set layout and options
        pos = nx.spring_layout(labeled_graph, seed=42, k=0.5)  # Calculate positions for nodes

        # Adjust positions for better spacing
        threshold = 0.1  # Distance threshold
        scaling_factor = 1.5  # Scaling multiplier

        nodes = list(pos.keys())
        for i, node1 in enumerate(nodes):
            x1, y1 = pos[node1]
            for node2 in nodes[i + 1:]:
                x2, y2 = pos[node2]
                dist = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
                if dist < threshold:  # Check if distance is below threshold
                    pos[node2] = (x2 * scaling_factor, y2 * scaling_factor)

        # Assign colors based on component number
        components = {node.component for node in self._graph.nodes}  # Unique component numbers
        color_map = plt.cm.get_cmap('tab10', len(components))  # Generate a colormap
        component_to_color = {comp: color_map(idx) for idx, comp in enumerate(sorted(components))}
        node_colors = [component_to_color[node.component] for node in self._graph.nodes]

        # Set options for drawing the graph
        options = {
        'node_color': node_colors,
        'node_size': 700,
        'width': 2,
        'with_labels': True,
        'font_size': 10,
        'cmap': 'tab10',
    }

        # Draw the graph with labeled nodes
        plt.ioff() 
        plt.figure(figsize=(12, 12))
        nx.draw(labeled_graph, pos, **options)
        #nx.draw_networkx_labels(labeled_graph, pos, labels=node_labels, font_size=8)

        plt.title("Graph Visualization with Component Names")

        #print("Nodes: ", labeled_graph.nodes)
        #print("Edges: ", labeled_graph.edges)
        #plt.tight_layout()
        plt.savefig("graphs/graph.png",dpi=300, bbox_inches='tight')
        plt.close()

        # from pyvis.network import Network
        # net = Network(notebook=True)
        # net.from_nx(labeled_graph)
        # #net.from_nx(self._graph)
        #net.show("graph.html")




    @classmethod
    def from_sequence(
        cls, root_id: int, node_id: int, sequence: VectorRotationSequence
    ) -> Self:
        new_ = cls()
        new_.set_root(root_id, direction=sequence.basis_array[:, 0, 0])

        # Add sequence without root
        new_.add_sequence(
            node_id=node_id,
            parent_id=root_id,
            sequence=sequence,
            common_parent=True,
        )
        return new_

    def add_sequence(
        self,
        node_id: NodeType,
        sequence: VectorRotationSequence,
        parent_id: NodeType,
        common_parent: bool = True,
    ) -> None:
        # breakpoint()
        if common_parent:
            # Check if roots are really the same
            if not np.allclose(
                sequence.basis_array[:, 0, 0],
                self._graph.nodes[parent_id]["direction"],
            ):
                # breakpoint()
                raise ValueError("Parents are not aligned.")
            it_start = 1
        else:
            it_start = 0

        for ii in range(it_start, sequence.n_rotations):
            internode_id = (node_id, ii)
            self.add_node(
                node_id=internode_id,
                direction=sequence.basis_array[:, ii, 0],
                parent_id=parent_id,
            )
            parent_id = internode_id

        # Real NodeId for the last node
        final_direction = sequence.get_end_vector()
        self.add_node(
            node_id=node_id,
            direction=final_direction,
            parent_id=parent_id,
        )

    def set_direction(self, node_id: int, direction: Vector) -> None:
        self._graph.nodes[node_id]["direction"] = direction

        # TODO: Not all rotation would have to be reset (only when higher weight is changed..)
        # Do better checks to accelerate
        successors = self.get_all_childs_children(node_id)
        for succ in successors:
            succ["orientation"] = None

    def evaluate_all_orientations(
        self, sorted_list: list[NodeType] = None, pi_margin: Optional[float] = None
    ) -> None:
        """Updates all orientations of the '_graph' class.
        -> store the new direction in the graph as 'part_direction'
        """
        # TODO: only store partial_direction here (!)
        if sorted_list is None:
            sorted_list = self.get_nodes_ascending()

        # Special values for root-node
        self._graph.nodes[sorted_list[0]]["part_direction"] = self._graph.nodes[
            sorted_list[0]
        ]["direction"]

        self._graph.nodes[sorted_list[0]][
            "orientation"
        ] = VectorRotationXd.from_directions(
            self._graph.nodes[sorted_list[0]]["direction"],
            self._graph.nodes[sorted_list[0]]["direction"],
        )

        for n_id in self._graph.nodes:
            for c_id in self._graph.successors(n_id):
                self._graph.nodes[c_id][
                    "orientation"
                ] = VectorRotationXd.from_directions(
                    self._graph.nodes[n_id]["part_direction"],
                    self._graph.nodes[c_id]["direction"],
                )

                if np.any(np.isnan(self._graph.nodes[c_id]["orientation"].base)):
                    breakpoint()

                # At which angle the rotation should be reduced to obtain a continuous behavior
                if (
                    pi_margin
                    and self._graph.nodes[c_id]["orientation"].rotation_angle
                    > pi_margin
                ):
                    warnings.warn("Breakpoint deactivatet -- check 'vector_rotation'")
                    breakpoint()
                    weight = self._graph.nodes[c_id]["orientation"].rotation_angle
                    weight = (math.pi - abs(weight)) / (math.pi - pi_margin)
                    self._graph.nodes[c_id]["orientation"].rotation_angle *= weight

                    self._graph.nodes[c_id]["part_direction"] = self._graph.nodes[c_id][
                        "orientation"
                    ].get_second_vector()

                else:
                    self._graph.nodes[c_id]["part_direction"] = self._graph.nodes[c_id][
                        "direction"
                    ]

                if np.any(np.isnan(self._graph.nodes[c_id]["part_direction"])):
                    breakpoint()

    def reset_node_weights(self) -> None:
        for node_id in self._graph.nodes:
            self._graph.nodes[node_id]["weight"] = 0

    def set_node(
        self, node_id: NodeType, parent_id: NodeType, direction: Vector
    ) -> None:
        # TODO: implement such that it updates lower and higher nodes (!)
        raise NotImplementedError()

    def get_nodes_ascending(self) -> list[NodeType]:
        # Ascending sorted node-list
        level_list = [self._graph.nodes[node]["level"] for node in self._graph.nodes]
        node_unsorted = [node for node in self._graph.nodes]
        return [node_unsorted[ii] for ii in np.argsort(level_list)]

    def get_all_childs_children(self, node: NodeType) -> list[NodeType]:
        """Returns list of nodes which are in the directional line of the argument node."""
        successor_list = [ii for ii in self._graph.successors(node)]

        ii = 0
        while ii < len(successor_list):
            # Add all children elements to the list
            successor_list += [jj for jj in self._graph.successors(successor_list[ii])]
            ii += 1

        return successor_list

    def get_weighted_mean(
        self, node_list: list[NodeType], weights: list[float]
    ) -> Vector:
        """Evaluate the weighted mean of the graph."""
        # sorted_list = self.get_nodes_ascending()
        # self.update_partial_rotations(node_list, weights, sorted_list)
        # rotation_sequence =  self.evaluate_graph_summing(sorted_list)
        rotation_sequence = self.reduce_weighted_to_sequence(node_list, weights)
        return rotation_sequence.get_end_vector()

    def reduce_weighted_to_sequence(
        self, node_list: list[NodeType], weights: npt.ArrayLike
    ) -> VectorRotationSequence:
        if len(node_list) != len(weights):
            raise ValueError(
                "Number of nodes does not correspond to number of weights."
            )

        sorted_list = self.get_nodes_ascending()
        self.update_partial_rotations(node_list, weights, sorted_list)
        return self.evaluate_graph_summing(sorted_list)

    def update_partial_rotations(
        self, node_list: list[NodeType], weights: npt.ArrayLike, sorted_list
    ) -> None:
        if not math.isclose((weight_sum := np.sum(weights)), 1.0, rel_tol=1e-3):
            warnings.warn(f"Sum of weights {weight_sum} is not equal to one.")

        # TODO: Maybe this should be done at the end of the step (?)
        self.reset_node_weights()

        # Weights are stored in the predecessing nodes of the corresponding edge
        for ii, node in enumerate(node_list):
            self._graph.nodes[node]["weight"] = weights[ii]

        self.evaluate_all_orientations(sorted_list)
        for node in self._graph.nodes():
            if np.any(np.isnan(self._graph.nodes[node]["orientation"].base)):
                breakpoint()

        for node_id in reversed(sorted_list):
            # Reverse update the weights
            for pred_id in self._graph.predecessors(node_id):
                # There is only one predecessor,
                # Where are the weights stored / where are the rotations stored (?)
                self._graph.nodes[pred_id]["weight"] += self._graph.nodes[node_id][
                    "weight"
                ]

            # Update orientation and create 'partial' orientations
            self._graph.nodes[node_id]["part_orientation"] = VectorRotationXd(
                base=self._graph.nodes[node_id]["orientation"].base,
                rotation_angle=(
                    self._graph.nodes[node_id]["orientation"].rotation_angle
                    * self._graph.nodes[node_id]["weight"]
                ),
            )

            if np.any(np.isnan(self._graph.nodes[node_id]["part_orientation"].base)):
                breakpoint()

            if (
                not self._graph.successors(node_id)
                or self._graph.nodes[node_id]["weight"] <= 0
                or self._graph.nodes[node_id]["weight"] >= 1
            ):
                # No successor nodes (or successors with only zero weight !? )
                # or full rotation is being kept
                # TODO: why is the second condition not working ?!
                continue

            successors = self.get_all_childs_children(node_id)

            _succ_basis = []
            for succ in successors:
                _succ_basis.append(self._graph.nodes[succ]["part_orientation"].base)

            if not len(_succ_basis):
                continue

            if np.any(np.isnan(_succ_basis)):
                breakpoint()

            # Make sure dimension is the first axes for future array restructuring
            succ_basis = np.swapaxes(_succ_basis, 0, 1)

            # TODO: this could be directly integrated in the final rotation (as there we
            # fully rotate backwards)
            #     hence remove this - add more (inverse rotation) later => safe an operation...
            # Backwards rotate such that it's aligned with the new angle
            succ_basis = rotate_array(
                directions=succ_basis.reshape(self.dimension, -1),
                base=self._graph.nodes[node_id]["orientation"].base,
                rotation_angle=(
                    self._graph.nodes[node_id]["part_orientation"].rotation_angle
                    - self._graph.nodes[node_id]["orientation"].rotation_angle
                ),
            ).reshape(self.dimension, -1, 2)

            for ii, succ in enumerate(successors):
                self._graph.nodes[succ]["part_orientation"].base = succ_basis[:, ii, :]

            if np.any(np.isnan(self._graph.nodes[succ]["part_orientation"].base)):
                breakpoint()
                aa = 0

    def evaluate_graph_summing(self, sorted_list) -> VectorRotationSequence:
        """Graph summing under assumption of shared-basis at each level.

        => the number of calculations is $2x (n_{childrend} of node) \forall node in nodes $
        i.e. does currently not scale well
        But calculations are simple, i.e., this could be sped upt with cython / C++ / Rust
        """
        level_list = [self._graph.nodes[node_id]["level"] for node_id in sorted_list]
        # sequence_vectors = np.zeros((self.dimension, len(set(level_list))))
        vector_sequence = VectorRotationSequence.create_empty(dimension=self.dimension)

        # Bottom up calculation - from lowest level to highest level
        # at each level, take the weighted average of all rotations
        for ll, level in enumerate(set(level_list)):
            nodelevel_ids = []
            for node_id, lev in zip(sorted_list, level_list):
                if lev == level and self._graph.nodes[node_id]["weight"]:
                    nodelevel_ids.append(node_id)

            if not nodelevel_ids:
                continue

            # Each round higher levels are un-rotated to share the same basis
            shared_first_basis = self._graph.nodes[nodelevel_ids[0]][
                "part_orientation"
            ].base[:, 0]
            shared_basis = get_orthogonal_basis(shared_first_basis)

            # Get the rotation-vector (second -base vector) of all of the
            # same-level rotation-structs in the local_basis
            local_basis = np.array(
                [
                    self._graph.nodes[jj]["part_orientation"].base[:, 1]
                    for jj in nodelevel_ids
                ]
            ).T
            # local_basis = shared_basis.T @ local_basis

            # Add the rotation angles up
            local_basis *= np.array(
                [
                    self._graph.nodes[jj]["part_orientation"].rotation_angle
                    for jj in nodelevel_ids
                ]
            )
            local_mean_basis = np.sum(local_basis, axis=1)
            new_angle = LA.norm(local_mean_basis)
            if new_angle:  # Nonzero
                # local_mean_basis[0] = 0  # Really (?)
                # averaged_direction = shared_basis @ (local_mean_basis / new_angle)
                averaged_direction = local_mean_basis / new_angle
            else:
                # No rotation, hence it's the first vector
                averaged_direction = shared_basis[:, 0]

            # Rotate all following rotation-levels back
            all_successors = []
            all_basis = np.zeros((self.dimension, 0))
            for node_id in nodelevel_ids:
                # Transform all child angles to first base-direction
                successors = self.get_all_childs_children(node_id)

                if not successors:
                    # No successors
                    continue

                _succ_basis = [
                    self._graph.nodes[succ]["part_orientation"].base
                    for succ in successors
                ]
                # Make sure dimension is the first axes for future array restructuring
                succ_basis = np.swapaxes(_succ_basis, 0, 1)

                succ_basis = rotate_array(
                    directions=succ_basis.reshape(self.dimension, -1),
                    base=self._graph.nodes[node_id]["part_orientation"].base,
                    rotation_angle=(-1)
                    * self._graph.nodes[node_id]["part_orientation"].rotation_angle,
                )

                all_successors += successors
                all_basis = np.hstack((all_basis, succ_basis))

            if not all_successors:
                # Zero list -> check the next level
                continue

            if new_angle:
                # Only append if nonzero angle, i.e., actual rotation
                new_base = np.vstack((shared_first_basis, averaged_direction)).T
                # Transform to the new basis-direction
                all_basis = rotate_array(
                    # directions=all_basis.reshape(self.dimension, -1),
                    directions=all_basis,
                    base=new_base,
                    rotation_angle=new_angle,
                ).reshape(self.dimension, -1, 2)

                vector_sequence.append_from_base_and_angle(new_base, new_angle)

            else:
                # Zero transformation angle, hence just reshape
                all_basis = all_basis.reshape(self.dimension, -1, 2)

            for ii, node in enumerate(all_successors):
                self._graph.nodes[node]["part_orientation"].base = all_basis[:, ii, :]

        vector_sequence.append_from_base_and_angle(
            np.vstack((shared_first_basis, averaged_direction)).T, new_angle
        )

        if np.any(np.isnan(vector_sequence.basis_array)):
            breakpoint()  # TODO: remove debug

        return vector_sequence

    def rotate(
        self, initial_vector: Vector, node_list: list[int], weights: list[float]
    ) -> Vector:
        """Returns the rotated vector based on the mean-direction.

        Assumption that the initial"""
        rotated_dir = self.get_weighted_mean()
        temp_rotation = VectorRotationXd.from_directions(
            self.root["orientation"].base[:, 0], rotated_dir
        )

        return rotate_direction(
            direction=initial_vector,
            base=temp_rotation.base,
            rotation_angle=temp_rotation.rotation_angle,
        )

    def inverse_rotate(
        self, initial_vector: Vector, node_list: list[int], weights: list[float]
    ) -> Vector:
        """Returns the rotated vector based on the mean-direction.

        Assumption that the initial"""
        rotated_dir = self.get_weighted_mean()
        temp_rotation = VectorRotationXd.from_directions(
            rotated_dir, self.root["orientation"].base[:, 0]
        )

        return rotate_direction(
            direction=initial_vector,
            base=temp_rotation.base,
            rotation_angle=temp_rotation.rotation_angle,
        )

    def get_rotation_weights(self, parent_id: int, direction: Vector) -> float:
        pass

    def rotate_weighted(self, node_id_list: list[int], weights: list[float]):
        # For safe rotation at the back
        raise NotImplementedError()


def create_zero_vector_rotation(dimension: int) -> VectorRotationXd:
    """Creates a zero vector rotation object."""
    return VectorRotationXd(base=np.eye((dimension, 2)), rotation_angle=0.0)
