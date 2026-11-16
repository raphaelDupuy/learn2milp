"""Text file format of a branching model, shared with the C++ loader.

    MIS_BRANCHING_MODEL <version>
    features <count>
    feature <name> <mean> <standard deviation>      one line per feature
    layers <count>
    then, for each layer, one of:
    layer <inputs> <outputs> <relu|linear>          dense layer:
    <inputs values>                                   one line per output unit
    <outputs values>                                  one line of biases
    graph_layer <inputs> <outputs> <relu|linear>    graph layer:
    <inputs values>                                   W_self, one line per output unit
    <inputs values>                                   W_neighbours, same
    <inputs values>                                   W_node, same
    <outputs values>                                  one line of biases

The candidates of a node are scored together. Their features are first
standardized, (x - mean) / std. A dense layer then transforms each
candidate on its own, h' = activation(W h + b). A graph layer also looks
at the other candidates, through the edges of the graph between them:

    h'_v = activation(W_self h_v + W_neighbours mean(h_u, u neighbour of v)
                      + W_node mean(h_u, u candidate of the node) + b)

where the mean over no neighbour is 0. The last layer has one output, the
score; the candidate with the highest score is chosen.

Version 1 files have dense layers only, and a model without graph layer
is always written as version 1: every candidate is then scored on its own.

The feature names tie the model to the features it was trained on: a
model may use any subset of the features computed by the C code, and the
C++ loader refuses a model that names an unknown feature.
"""

from __future__ import annotations

import dataclasses
import pathlib

import numpy as np

HEADER = "MIS_BRANCHING_MODEL"
DENSE = "dense"
GRAPH = "graph"


@dataclasses.dataclass
class Layer:
    """One layer of a model.

    weights  dense: [outputs, inputs]; graph: [3, outputs, inputs], the
             matrices W_self, W_neighbours and W_node
    biases   [outputs]
    """

    kind: str
    weights: np.ndarray
    biases: np.ndarray
    activation: str


@dataclasses.dataclass
class BranchingModel:
    feature_names: list[str]
    mean: np.ndarray
    std: np.ndarray
    layers: list[Layer]

    @property
    def has_graph_layers(self) -> bool:
        return any(layer.kind == GRAPH for layer in self.layers)

    def scores(self, features: np.ndarray) -> np.ndarray:
        """Scores of candidates given raw features [..., features].

        Only for models without graph layer: each candidate is scored on
        its own.
        """
        if self.has_graph_layers:
            raise ValueError("A graph model scores the candidates of a node together")

        x = (features.astype(np.float64) - self.mean) / self.std

        for layer in self.layers:
            x = _activate(x @ layer.weights.T + layer.biases, layer.activation)

        return x[..., 0]

    def graph_scores(
        self,
        features: np.ndarray,
        sample_of_row: np.ndarray,
        number_of_samples: int,
        sources: np.ndarray,
        targets: np.ndarray,
    ) -> np.ndarray:
        """Scores of the candidates of several nodes (samples).

        features [rows, features] holds the candidates one sample after the
        other, sample_of_row gives the sample of each row, and every edge
        between two candidates appears in both directions in sources and
        targets (row sources[i] sends to row targets[i]).
        """
        rows = len(features)
        x = (features.astype(np.float64) - self.mean) / self.std

        # Neighbours of each row, contiguous: segment sums by reduceat.
        order = np.argsort(targets, kind="stable")
        senders = sources[order]
        degree = np.bincount(targets, minlength=rows)
        has_neighbours = degree > 0
        neighbour_starts = (np.cumsum(degree) - degree)[has_neighbours]

        sample_size = np.bincount(sample_of_row, minlength=number_of_samples)
        sample_starts = np.cumsum(sample_size) - sample_size

        for layer in self.layers:
            if layer.kind == DENSE:
                x = _activate(x @ layer.weights.T + layer.biases, layer.activation)
                continue

            neighbours = np.zeros_like(x)

            if len(senders):
                neighbours[has_neighbours] = np.add.reduceat(
                    x[senders], neighbour_starts, axis=0
                ) / degree[has_neighbours, None]

            node = np.add.reduceat(x, sample_starts, axis=0) / sample_size[:, None]
            w_self, w_neighbours, w_node = layer.weights
            x = _activate(
                x @ w_self.T
                + neighbours @ w_neighbours.T
                + node[sample_of_row] @ w_node.T
                + layer.biases,
                layer.activation,
            )

        return x[:, 0]


def _activate(x: np.ndarray, activation: str) -> np.ndarray:
    return np.maximum(x, 0.0) if activation == "relu" else x


def _numbers(values: np.ndarray, digits: int) -> str:
    return " ".join(f"{float(value):.{digits}g}" for value in values)


def write_model(path: str | pathlib.Path, model: BranchingModel) -> None:
    version = 2 if model.has_graph_layers else 1
    lines = [f"{HEADER} {version}", f"features {len(model.feature_names)}"]

    for name, mean, std in zip(model.feature_names, model.mean, model.std):
        lines.append(f"feature {name} {float(mean):.17g} {float(std):.17g}")

    lines.append(f"layers {len(model.layers)}")

    for layer in model.layers:
        if layer.kind == DENSE:
            outputs, inputs = layer.weights.shape
            lines.append(f"layer {inputs} {outputs} {layer.activation}")
            lines.extend(_numbers(row, 9) for row in layer.weights)
        else:
            _, outputs, inputs = layer.weights.shape
            lines.append(f"graph_layer {inputs} {outputs} {layer.activation}")

            for matrix in layer.weights:
                lines.extend(_numbers(row, 9) for row in matrix)

        lines.append(_numbers(layer.biases, 9))

    pathlib.Path(path).write_text("\n".join(lines) + "\n")


def read_model(path: str | pathlib.Path) -> BranchingModel:
    lines = pathlib.Path(path).read_text().splitlines()
    header = lines[0].split()

    if len(header) != 2 or header[0] != HEADER or header[1] not in ("1", "2"):
        raise ValueError(f"Not a branching model file: {path}")

    version = int(header[1])
    count = int(lines[1].split()[1])
    names, means, stds = [], [], []

    for line in lines[2 : 2 + count]:
        _, name, mean, std = line.split()
        names.append(name)
        means.append(float(mean))
        stds.append(float(std))

    position = 2 + count
    number_of_layers = int(lines[position].split()[1])
    position += 1
    layers = []

    def matrix(first: int, outputs: int, inputs: int) -> np.ndarray:
        rows = [[float(value) for value in lines[first + row].split()] for row in range(outputs)]
        return np.array(rows, np.float64).reshape(outputs, inputs)

    for _ in range(number_of_layers):
        keyword, inputs, outputs, activation = lines[position].split()
        inputs, outputs = int(inputs), int(outputs)

        if keyword == "layer":
            kind, matrices = DENSE, 1
        elif keyword == "graph_layer" and version >= 2:
            kind, matrices = GRAPH, 3
        else:
            raise ValueError(f"Unexpected layer '{keyword}' in {path}")

        weights = [matrix(position + 1 + m * outputs, outputs, inputs) for m in range(matrices)]
        biases_line = position + 1 + matrices * outputs
        biases = np.array([float(value) for value in lines[biases_line].split()], np.float64)
        layers.append(
            Layer(kind, weights[0] if kind == DENSE else np.stack(weights), biases, activation)
        )
        position = biases_line + 1

    return BranchingModel(names, np.array(means), np.array(stds), layers)
