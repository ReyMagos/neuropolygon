import navis
import pandas as pd

PRESYNAPSE_TYPE = 0
POSTSYNAPSE_TYPE = 1

SWC_SEP = ' '
SWC_COMMENT = '#'
SWC_COLUMNS = ("node_id", "label", "x", "y", "z", "radius", "parent_id")


class Neuron(navis.TreeNeuron):
    root_id: int
    nt_type: str | None = None

    @staticmethod
    def from_swc(root_id: int) -> "Neuron":
        swc = pd.read_csv(
            f"data/skeletons/{root_id}.swc",
            sep=SWC_SEP,
            comment=SWC_COMMENT,
            names=SWC_COLUMNS,
        )

        neuron = Neuron(swc)
        neuron.root_id = root_id

        return neuron
