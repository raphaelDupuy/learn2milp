"""Text file format of a branching model, shared with the C++ loader.

    MIS_BRANCHING_MODEL 1
    features <count>
    feature <name> <mean> <standard deviation>      one line per feature
    layers <count>
    layer <inputs> <outputs> <relu|linear>          then, for each layer:
    <inputs values>                                 one line per output unit
    <outputs values>                                one line of biases

A candidate is scored by standardizing its features, (x - mean) / std,
then applying the layers in order: y = activation(W x + b). The last
layer has one output, the score; the candidate with the highest score is
chosen. The feature names tie the model to the features it was trained
on: a model may use any subset of the features computed by the C code,
and the C++ loader refuses a model that names an unknown feature.
"""

from __future__ import annotations

import dataclasses
import pathlib

import numpy as np

HEADER = "MIS_BRANCHING_MODEL 1"


@dataclasses.dataclass
class BranchingModel:
    feature_names: list[str]
    mean: np.ndarray
    std: np.ndarray
    # (weights [outputs, inputs], biases [outputs], activation)
    layers: list[tuple[np.ndarray, np.ndarray, str]]

    def scores(self, features: np.ndarray) -> np.ndarray:
        """Scores of candidates given raw features [..., features]."""
        x = (features.astype(np.float64) - self.mean) / self.std

        for weights, biases, activation in self.layers:
            x = x @ weights.T + biases

            if activation == "relu":
                x = np.maximum(x, 0.0)

        return x[..., 0]


def _numbers(values: np.ndarray, digits: int) -> str:
    return " ".join(f"{float(value):.{digits}g}" for value in values)


def write_model(path: str | pathlib.Path, model: BranchingModel) -> None:
    lines = [HEADER, f"features {len(model.feature_names)}"]

    for name, mean, std in zip(model.feature_names, model.mean, model.std):
        lines.append(f"feature {name} {float(mean):.17g} {float(std):.17g}")

    lines.append(f"layers {len(model.layers)}")

    for weights, biases, activation in model.layers:
        outputs, inputs = weights.shape
        lines.append(f"layer {inputs} {outputs} {activation}")
        lines.extend(_numbers(row, 9) for row in weights)
        lines.append(_numbers(biases, 9))

    pathlib.Path(path).write_text("\n".join(lines) + "\n")


def read_model(path: str | pathlib.Path) -> BranchingModel:
    lines = pathlib.Path(path).read_text().splitlines()

    if lines[0] != HEADER:
        raise ValueError(f"Not a branching model file: {path}")

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

    for _ in range(number_of_layers):
        _, inputs, outputs, activation = lines[position].split()
        inputs, outputs = int(inputs), int(outputs)
        rows = [
            [float(value) for value in lines[position + 1 + row].split()]
            for row in range(outputs)
        ]
        biases = [float(value) for value in lines[position + 1 + outputs].split()]
        weights = np.array(rows, np.float64).reshape(outputs, inputs)
        layers.append((weights, np.array(biases, np.float64), activation))
        position += 2 + outputs

    return BranchingModel(names, np.array(means), np.array(stds), layers)
