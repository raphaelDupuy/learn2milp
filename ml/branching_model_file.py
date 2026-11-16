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

Version 3 files describe a bipartite model, after Gasse et al. (2019):
the free vertices and the edge constraints x_u + x_v <= 1 between two free
vertices form a bipartite graph. After the feature lines come

    vertex_inputs <count>, then vertex_input <name> <mean> <std> lines
    constraint_inputs <count>, then constraint_input <name> <mean> <std> lines
    vertex_layers <count>, then dense layer blocks
    constraint_layers <count>, then dense layer blocks
    message_layers <count>, then for each:
    message_layer <to_constraints|to_vertices> <targets> <sources> <hidden> <outputs> <relu|linear>
    <targets values>      L, one line per hidden unit
    <sources values>      R, one line per hidden unit
    <hidden values>       one line of hidden biases
    <hidden values>       P, one line per output unit
    <targets values>      Q, one line per output unit
    <outputs values>      one line of biases
    layers <count>, then dense layer blocks

A vertex starts from its standardized candidate features (0 if it is not
a candidate) followed by its standardized vertex inputs; a constraint from
its standardized constraint inputs. The vertex and constraint layers
transform each side, then each message layer updates one side from the
other:

    t' = activation(P mean(relu(L t + R s + b_hidden), s source of t) + Q t + b)

where the sources of a constraint are its two vertices and those of a
vertex its constraints (mean 0 if it has none). The final layers score
each candidate from its vertex state.

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
class MessageLayer:
    """One message layer of a bipartite model (see the module docstring).

    direction        "to_constraints" or "to_vertices"
    target_weights   L [hidden, targets]
    source_weights   R [hidden, sources]
    hidden_biases    [hidden]
    message_weights  P [outputs, hidden]
    own_weights      Q [outputs, targets]
    biases           [outputs]
    """

    direction: str
    target_weights: np.ndarray
    source_weights: np.ndarray
    hidden_biases: np.ndarray
    message_weights: np.ndarray
    own_weights: np.ndarray
    biases: np.ndarray
    activation: str = "relu"


@dataclasses.dataclass
class BipartiteEncoder:
    """The part of a bipartite model before the final layers."""

    vertex_input_names: list[str]
    vertex_input_means: np.ndarray
    vertex_input_stds: np.ndarray
    constraint_input_names: list[str]
    constraint_input_means: np.ndarray
    constraint_input_stds: np.ndarray
    vertex_layers: list[Layer]
    constraint_layers: list[Layer]
    message_layers: list[MessageLayer]


@dataclasses.dataclass
class BranchingModel:
    feature_names: list[str]
    mean: np.ndarray
    std: np.ndarray
    layers: list[Layer]
    encoder: BipartiteEncoder | None = None

    @property
    def has_graph_layers(self) -> bool:
        return any(layer.kind == GRAPH for layer in self.layers)

    @property
    def is_bipartite(self) -> bool:
        return self.encoder is not None

    def bipartite_scores(
        self,
        candidate_features: np.ndarray,
        is_candidate: np.ndarray,
        vertex_inputs: np.ndarray,
        constraint_inputs: np.ndarray,
        endpoints: np.ndarray,
        candidate_rows: np.ndarray,
    ) -> np.ndarray:
        """Scores of the candidates of a bipartite model, in double
        precision, from the arrays of a BipartiteBatch."""
        encoder = self.encoder
        assert encoder is not None
        features = (candidate_features.astype(np.float64) - self.mean) / self.std
        features[~is_candidate] = 0.0
        v = np.concatenate(
            [features, (vertex_inputs - encoder.vertex_input_means) / encoder.vertex_input_stds],
            axis=1,
        )
        c = (constraint_inputs.astype(np.float64) - encoder.constraint_input_means) / encoder.constraint_input_stds

        for layer in encoder.vertex_layers:
            v = _activate(v @ layer.weights.T + layer.biases, layer.activation)

        for layer in encoder.constraint_layers:
            c = _activate(c @ layer.weights.T + layer.biases, layer.activation)

        number_of_constraints = len(c)
        incident_vertex = endpoints.T.ravel()  # all first ends, then all second ends
        incident_constraint = np.concatenate([np.arange(number_of_constraints)] * 2)
        order = np.argsort(incident_vertex, kind="stable")
        degree = np.bincount(incident_vertex, minlength=len(v))
        has_constraints = degree > 0
        starts = (np.cumsum(degree) - degree)[has_constraints]

        for layer in encoder.message_layers:
            if layer.direction == "to_constraints":
                targets, sources = c, v
                hidden = targets @ layer.target_weights.T + layer.hidden_biases
                projected = sources @ layer.source_weights.T
                mean = (
                    np.maximum(hidden + projected[endpoints[:, 0]], 0.0)
                    + np.maximum(hidden + projected[endpoints[:, 1]], 0.0)
                ) / 2.0
            else:
                targets, sources = v, c
                hidden = targets @ layer.target_weights.T + layer.hidden_biases
                projected = sources @ layer.source_weights.T
                messages = np.maximum(
                    hidden[incident_vertex[order]] + projected[incident_constraint[order]], 0.0
                )
                mean = np.zeros((len(targets), hidden.shape[1]))

                if len(messages):
                    mean[has_constraints] = (
                        np.add.reduceat(messages, starts, axis=0) / degree[has_constraints, None]
                    )

            updated = _activate(
                mean @ layer.message_weights.T + targets @ layer.own_weights.T + layer.biases,
                layer.activation,
            )

            if layer.direction == "to_constraints":
                c = updated
            else:
                v = updated

        x = v[candidate_rows]

        for layer in self.layers:
            x = _activate(x @ layer.weights.T + layer.biases, layer.activation)

        return x[:, 0]

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


def _dense_lines(layers: list[Layer]) -> list[str]:
    lines = []

    for layer in layers:
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

    return lines


def write_model(path: str | pathlib.Path, model: BranchingModel) -> None:
    version = 3 if model.is_bipartite else 2 if model.has_graph_layers else 1
    lines = [f"{HEADER} {version}", f"features {len(model.feature_names)}"]

    for name, mean, std in zip(model.feature_names, model.mean, model.std):
        lines.append(f"feature {name} {float(mean):.17g} {float(std):.17g}")

    encoder = model.encoder

    if encoder is not None:
        for side, names, means, stds in [
            ("vertex", encoder.vertex_input_names, encoder.vertex_input_means, encoder.vertex_input_stds),
            ("constraint", encoder.constraint_input_names, encoder.constraint_input_means, encoder.constraint_input_stds),
        ]:
            lines.append(f"{side}_inputs {len(names)}")
            lines.extend(
                f"{side}_input {name} {float(mean):.17g} {float(std):.17g}"
                for name, mean, std in zip(names, means, stds)
            )

        lines.append(f"vertex_layers {len(encoder.vertex_layers)}")
        lines.extend(_dense_lines(encoder.vertex_layers))
        lines.append(f"constraint_layers {len(encoder.constraint_layers)}")
        lines.extend(_dense_lines(encoder.constraint_layers))
        lines.append(f"message_layers {len(encoder.message_layers)}")

        for layer in encoder.message_layers:
            hidden, targets = layer.target_weights.shape
            sources = layer.source_weights.shape[1]
            outputs = layer.message_weights.shape[0]
            lines.append(
                f"message_layer {layer.direction} {targets} {sources} {hidden} {outputs} {layer.activation}"
            )
            lines.extend(_numbers(row, 9) for row in layer.target_weights)
            lines.extend(_numbers(row, 9) for row in layer.source_weights)
            lines.append(_numbers(layer.hidden_biases, 9))
            lines.extend(_numbers(row, 9) for row in layer.message_weights)
            lines.extend(_numbers(row, 9) for row in layer.own_weights)
            lines.append(_numbers(layer.biases, 9))

    lines.append(f"layers {len(model.layers)}")
    lines.extend(_dense_lines(model.layers))

    pathlib.Path(path).write_text("\n".join(lines) + "\n")


class _Reader:
    """Lines of a model file, read one after the other."""

    def __init__(self, path: str | pathlib.Path):
        self.path = path
        self.lines = pathlib.Path(path).read_text().splitlines()
        self.position = 0

    def tokens(self) -> list[str]:
        line = self.lines[self.position]
        self.position += 1
        return line.split()

    def count(self, keyword: str) -> int:
        tokens = self.tokens()

        if len(tokens) != 2 or tokens[0] != keyword:
            raise ValueError(f"Expected '{keyword} <count>' in {self.path}")

        return int(tokens[1])

    def numbers(self) -> np.ndarray:
        return np.array([float(value) for value in self.tokens()], np.float64)

    def matrix(self, rows: int, columns: int) -> np.ndarray:
        return np.array([self.numbers() for _ in range(rows)], np.float64).reshape(rows, columns)

    def named_values(self, keyword: str, count: int) -> tuple[list[str], np.ndarray, np.ndarray]:
        names, means, stds = [], [], []

        for _ in range(count):
            word, name, mean, std = self.tokens()

            if word != keyword:
                raise ValueError(f"Expected '{keyword}' lines in {self.path}")

            names.append(name)
            means.append(float(mean))
            stds.append(float(std))

        return names, np.array(means), np.array(stds)

    def layers(self, count: int, graph_allowed: bool) -> list[Layer]:
        layers = []

        for _ in range(count):
            keyword, inputs, outputs, activation = self.tokens()
            inputs, outputs = int(inputs), int(outputs)

            if keyword == "layer":
                layers.append(Layer(DENSE, self.matrix(outputs, inputs), self.numbers(), activation))
            elif keyword == "graph_layer" and graph_allowed:
                weights = np.stack([self.matrix(outputs, inputs) for _ in range(3)])
                layers.append(Layer(GRAPH, weights, self.numbers(), activation))
            else:
                raise ValueError(f"Unexpected layer '{keyword}' in {self.path}")

        return layers


def read_model(path: str | pathlib.Path) -> BranchingModel:
    reader = _Reader(path)
    header = reader.tokens()

    if len(header) != 2 or header[0] != HEADER or header[1] not in ("1", "2", "3"):
        raise ValueError(f"Not a branching model file: {path}")

    version = int(header[1])
    names, means, stds = reader.named_values("feature", reader.count("features"))
    encoder = None

    if version == 3:
        vertex = reader.named_values("vertex_input", reader.count("vertex_inputs"))
        constraint = reader.named_values("constraint_input", reader.count("constraint_inputs"))
        vertex_layers = reader.layers(reader.count("vertex_layers"), False)
        constraint_layers = reader.layers(reader.count("constraint_layers"), False)
        message_layers = []

        for _ in range(reader.count("message_layers")):
            keyword, direction, targets, sources, hidden, outputs, activation = reader.tokens()
            targets, sources, hidden, outputs = int(targets), int(sources), int(hidden), int(outputs)

            if keyword != "message_layer" or direction not in ("to_constraints", "to_vertices"):
                raise ValueError(f"Unexpected message layer in {path}")

            message_layers.append(
                MessageLayer(
                    direction,
                    reader.matrix(hidden, targets),
                    reader.matrix(hidden, sources),
                    reader.numbers(),
                    reader.matrix(outputs, hidden),
                    reader.matrix(outputs, targets),
                    reader.numbers(),
                    activation,
                )
            )

        encoder = BipartiteEncoder(
            *vertex, *constraint, vertex_layers, constraint_layers, message_layers
        )

    layers = reader.layers(reader.count("layers"), version == 2)

    return BranchingModel(names, means, stds, layers, encoder)
