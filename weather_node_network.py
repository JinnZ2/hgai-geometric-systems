"""
Weather Node Network — Probabilistic Forecasting Pipeline

A closed-loop data -> simulation -> scenario -> feedback architecture
designed to embrace uncertainty, chaotic sensitivity, and interconnection.

Layers:
    1. Data Acquisition — multi-source sensor ingestion
    2. Preprocessing & Assimilation — bias correction, normalization
    3. Model Integration — ensemble simulation with uncertainty propagation
    4. Postprocessing — probabilistic summarization (median, percentiles)
    5. Feedback — observation comparison and adaptive learning

Architecture uses a node-based network where each stage processes inputs
and propagates outputs (including uncertainty metadata) to downstream nodes.
"""

from typing import Any, Dict, List
import numpy as np


# -----------------------------
# Base Node Class
# -----------------------------
class Node:
    """Base class for pipeline nodes in the weather forecasting network.

    Parameters
    ----------
    name : str
        Human-readable identifier for the node.
    """

    def __init__(self, name: str):
        self.name = name
        self.outputs: Dict[str, Any] = {}
        self.downstream: List[Node] = []

    def connect(self, node: 'Node'):
        """Connect a downstream node to receive this node's outputs.

        Parameters
        ----------
        node : Node
            The downstream node to connect.
        """
        self.downstream.append(node)

    def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Each node overrides this with its own computation.

        Parameters
        ----------
        inputs : dict
            Data dictionary from upstream node.

        Returns
        -------
        dict
            Processed outputs including uncertainty metadata.
        """
        raise NotImplementedError

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute this node and propagate to downstream nodes.

        Parameters
        ----------
        inputs : dict
            Data dictionary from upstream node.

        Returns
        -------
        dict
            This node's outputs.
        """
        self.outputs = self.process(inputs)
        print(f"[{self.name}] outputs: {list(self.outputs.keys())}")
        for node in self.downstream:
            node.run(self.outputs)
        return self.outputs


# -----------------------------
# 1. Data Acquisition Node
# -----------------------------
class DataAcquisition(Node):
    """Multi-source data ingestion node.

    Fetches data from satellite, radar, ground station, ocean sensor,
    and crowdsourced sources. Each source carries uncertainty estimates.
    """

    def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        data = {
            "satellite": np.random.rand(5, 5),
            "radar": np.random.rand(5, 5),
            "ground": np.random.rand(5, 5),
            "ocean": np.random.rand(5, 5),
            "uncertainty": np.random.rand(5, 5) * 0.1
        }
        return data


# -----------------------------
# 2. Preprocessing & Assimilation Node
# -----------------------------
class Preprocessing(Node):
    """Bias correction, gap-filling, and cross-source normalization.

    Normalizes each data source to zero mean and unit variance while
    preserving uncertainty estimates for downstream ensemble generation.
    """

    def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        processed = {}
        for k, v in inputs.items():
            if k != "uncertainty":
                processed[k] = (v - np.mean(v)) / (np.std(v) + 1e-6)
        processed["uncertainty"] = inputs["uncertainty"]
        return processed


# -----------------------------
# 3. Model Integration Node
# -----------------------------
class ModelIntegration(Node):
    """Ensemble simulation with uncertainty propagation.

    Runs multiple perturbed simulations combining atmospheric, oceanic,
    land-surface, and microclimate signals. Each ensemble member samples
    from the uncertainty distribution to capture chaotic sensitivity.

    Parameters
    ----------
    ensemble_size : int
        Number of ensemble members to simulate.
    """

    def __init__(self, name: str, ensemble_size: int = 5):
        super().__init__(name)
        self.ensemble_size = ensemble_size

    def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        ensemble = []
        for _ in range(self.ensemble_size):
            perturb = inputs["uncertainty"] * np.random.randn(
                *inputs["uncertainty"].shape
            )
            sim = sum(
                [v for k, v in inputs.items() if k != "uncertainty"]
            ) + perturb
            ensemble.append(sim)
        return {"ensemble": ensemble, "uncertainty": inputs["uncertainty"]}


# -----------------------------
# 4. Postprocessing & Communication Node
# -----------------------------
class Postprocessing(Node):
    """Probabilistic summarization of ensemble forecasts.

    Computes median, 10th and 90th percentile bounds from the ensemble
    to communicate forecast uncertainty as probability ranges rather
    than deterministic predictions.
    """

    def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        ensemble = np.array(inputs["ensemble"])
        return {
            "median": np.median(ensemble, axis=0),
            "p10": np.percentile(ensemble, 10, axis=0),
            "p90": np.percentile(ensemble, 90, axis=0),
            "uncertainty": inputs["uncertainty"]
        }


# -----------------------------
# 5. Feedback & Continuous Learning Node
# -----------------------------
class Feedback(Node):
    """Observation-prediction comparison and adaptive learning.

    Compares ensemble predictions against observed values to compute
    error signals. In a full implementation, these errors feed back
    to update upstream model parameters and uncertainty estimates.
    """

    def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        observations = np.random.rand(*inputs["median"].shape)
        error = observations - inputs["median"]
        print(f"[{self.name}] mean error: {np.mean(error):.4f}")
        return {"error": error}


# -----------------------------
# Build & Run Network
# -----------------------------
def build_network(ensemble_size: int = 5) -> DataAcquisition:
    """Construct the weather forecasting node network.

    Parameters
    ----------
    ensemble_size : int
        Number of ensemble members for the model integration layer.

    Returns
    -------
    DataAcquisition
        The root node of the network. Call ``.run({})`` to execute.
    """
    da = DataAcquisition("DataAcquisition")
    pp = Preprocessing("Preprocessing")
    mi = ModelIntegration("ModelIntegration", ensemble_size=ensemble_size)
    post = Postprocessing("Postprocessing")
    fb = Feedback("Feedback")

    da.connect(pp)
    pp.connect(mi)
    mi.connect(post)
    post.connect(fb)

    return da


def run_pipeline(ensemble_size: int = 5) -> Dict[str, Any]:
    """Build and execute the full weather pipeline.

    Parameters
    ----------
    ensemble_size : int
        Number of ensemble members for the model integration layer.

    Returns
    -------
    dict
        Root node outputs from the data acquisition layer.
    """
    network = build_network(ensemble_size=ensemble_size)
    return network.run({})


if __name__ == "__main__":
    print("=" * 50)
    print("Weather Node Network — Probabilistic Pipeline")
    print("=" * 50)
    final_output = run_pipeline()
    print("\nFinal median forecast (from postprocessing):")
    # The root node returns DataAcquisition outputs;
    # downstream results are stored on each node.
    print("Pipeline execution complete.")
