from collections.abc import Iterable
from functools import cached_property
from typing import ClassVar

import navis
import numpy as np
from tqdm import tqdm

from .neuron import POSTSYNAPSE_TYPE, PRESYNAPSE_TYPE, Neuron


class NeuronActivityStrategy:
    def __init__(self, skeletons: Iterable[Neuron]):
        self.skeletons = list(skeletons)

        self.skeleton_index = {}
        for i, skeleton in enumerate(self.skeletons):
            self.skeleton_index[skeleton.root_id] = i

        self.step_count = 0
        self.state = np.zeros(len(self.skeletons), dtype=np.float32)

    def fire(self, *root_ids: int):
        for root_id in root_ids:
            self.state[self.skeleton_index[root_id]] = 1

    def step(self):
        self.step_count += 1

    def plot(self, activity_threshold: float = 0.5):
        neurons = []
        for i in list(*np.nonzero(self.state > activity_threshold)):
            skeleton = self.skeletons[i]
            skeleton.nodes["activity"] = self.state[i]
            neurons.append(skeleton)

        if not neurons:
            return

        nl = navis.NeuronList(neurons)
        nl.plot3d(
            backend="plotly",
            color_by="activity",
            palette="flare",
            legend=False,
            title=f"{self.__class__.__name__} (step={self.step_count})",
        )


class WeightsMatrixStrategy(NeuronActivityStrategy):
    NT_EFFECTS: ClassVar = {
        "ACH": 1,    # Excitatory
        "DA": 1,     # Excitatory (can have inhibitory effects in specific brain regions)
        "GABA": -1,  # Inhibitory
        "GLUT": -1,  # Inhibitory (but excitatory in mammals!)
        "OCT": -1,   # Inhibitory
        "SER": -1,   # Inhibitory (can have excitatory effects in specific brain circuits)
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.scaling_factor = 1

    def effect(
        self,
        source_neuron: Neuron,
        target_neuron: Neuron,
        conenctor_ids: list[int],
    ) -> float:
        if source_neuron.nt_type is not None and source_neuron.nt_type in self.NT_EFFECTS:
            effect_sign = self.NT_EFFECTS[source_neuron.nt_type]
        else:
            effect_sign = 1

        return effect_sign * len(conenctor_ids)

    @cached_property
    def weights_matrix(self) -> np.ndarray:
        connectivity: dict[tuple[int, int], list[int]] = {}
        connector_input: dict[int, int] = {}
        connector_outputs: dict[int, list[int]] = {}

        for i, skeleton in tqdm(enumerate(self.skeletons)):
            for connector_id, connector_type in skeleton.connectors[["connector_id", "type"]].itertuples(index=False):
                if connector_type == PRESYNAPSE_TYPE:
                    i_input = i
                    connector_input[connector_id] = i_input

                    for i_output in connector_outputs.get(connector_id, []):
                        connectivity.setdefault((i_input, i_output), []).append(connector_id)

                elif connector_type == POSTSYNAPSE_TYPE:
                    i_output = i
                    connector_outputs.setdefault(connector_id, []).append(i_output)

                    if connector_id in connector_input:
                        i_input = connector_input[connector_id]
                        connectivity.setdefault((i_input, i_output), []).append(connector_id)

        n = len(self.skeletons)
        matrix = np.zeros((n, n), dtype=np.float32)

        print("Calculate effects")

        max_absolute_effect = -np.inf
        for (i_input, i_output), connector_ids in connectivity.items():
            e = self.effect(
                self.skeletons[i_input],
                self.skeletons[i_output],
                connector_ids,
            )

            matrix[i_input][i_output] = e
            max_absolute_effect = max(max_absolute_effect, e)

        self.scaling_factor = max_absolute_effect

        return matrix

    def step(self):
        self.step_count += 1

        self.state = (self.weights_matrix @ self.state) / self.scaling_factor


class EigendecomposedMatrixStrategy(WeightsMatrixStrategy):
    @cached_property
    def eigendecomposition(self):
        return np.linalg.eig(self.weights_matrix)

    @property
    def eigenbasis_transition_matrix(self):
        return np.linalg.inv(self.eigendecomposition.eigenvectors)

    @cached_property
    def eigenvalues_matrix(self):
        return np.diag(self.eigendecomposition.eigenvalues)

    def step(self):
        self.step_count += 1

        state_in_eigenbasis = self.eigenbasis_transition_matrix @ self.state
        self.state = (self.eigenvalues_matrix @ state_in_eigenbasis) / self.scaling_factor
