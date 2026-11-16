"""Trains a model that ranks the branching candidates of a node.

The model imitates the full strong branching expert: for each recorded
branching decision, it should give its highest score to the candidate
the expert chose. It is a small network that can be evaluated in C
inside SCIP in well under a millisecond per node.

Usage:
    python ml/train_branching_model.py \
        --train data/processed/features/er125_p01/train \
        --valid data/processed/features/er125_p01/valid \
        --hidden 32 32 --output experiments/er125_p01/models/mlp32.model

By default the network scores each candidate on its own (a multilayer
perceptron); use --hidden with no value for a linear model.

With --graph-layers K, the network is a graph neural network over the
candidates of the node, linked by the edges of the graph: the --hidden
layers transform each candidate, then K graph layers mix every
candidate's state with the mean state of its candidate neighbours and of
all the candidates (see branching_model_file.py). The graphs are read from
the instance files, searched for by name under --instances.

With --bipartite-rounds K, the network is a bipartite model after Gasse et
al. (2019): the free vertices and the edge constraints between two free
vertices form a bipartite graph, the --hidden layers embed the vertices
(candidate features and vertex inputs), and K rounds of messages go from
the vertices to the constraints and back before the candidates are
scored. The vertex states are read from the raw samples that match each
feature directory (see branching_data.with_vertex_states()).

--device cuda trains on the GPU; it needs a CUDA build of PyTorch, which
the project's CPU environment does not install. Training on the CPU (the
default) is reproducible bit for bit; on the GPU it is not.

--train and --valid accept several directories, which are aggregated:
this is how samples collected along the search of a previous model
(mis-collector --policy-model) are added to the expert's samples. The
validation metrics are then also reported for each directory.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import time

import numpy as np
import torch

from branching_data import (
    CONSTRAINT_INPUT_NAMES,
    VERTEX_INPUT_NAMES,
    BipartiteBatch,
    BranchingDataset,
    GraphBatch,
    load_feature_tables,
    segment_argmax,
)
from branching_model_file import (
    DENSE,
    GRAPH,
    BipartiteEncoder,
    BranchingModel,
    Layer,
    MessageLayer,
    read_model,
    write_model,
)


def decision_metrics(
    model_scores: np.ndarray, dataset: BranchingDataset
) -> dict[str, float]:
    """Quality of the decisions made by picking the highest model score.

    accuracy     share of samples where the pick is the expert's choice
    top5         share of samples where the expert's choice is among the
                 five highest model scores
    score_ratio  strong branching score of the pick divided by the best
                 score of the node, averaged over samples (1 = as good as
                 the expert, even when a different candidate is picked)
    mean_rank    average number of candidates the expert scores strictly
                 higher than the pick (0 = best)
    """
    # One model score per candidate, laid out like dataset.scores.
    starts = dataset.starts
    lengths = dataset.lengths
    pick = segment_argmax(model_scores, dataset.offsets)

    expert_scores = dataset.scores
    best = np.maximum.reduceat(expert_scores, starts)
    picked = expert_scores[starts + pick]

    target_score = model_scores[starts + dataset.target]
    better_than_target = np.add.reduceat(
        (model_scores > np.repeat(target_score, lengths)).astype(np.int64), starts
    )
    better_than_pick = np.add.reduceat(
        (expert_scores > np.repeat(picked, lengths)).astype(np.int64), starts
    )

    return {
        "accuracy": float((pick == dataset.target).mean()),
        "top5": float((better_than_target < 5).mean()),
        "score_ratio": float((picked / best).mean()),
        "mean_rank": float(better_than_pick.mean()),
    }


def reference_metrics(dataset: BranchingDataset) -> dict[str, dict[str, float]]:
    """Simple decision rules, to put the model's metrics in context."""
    rng = np.random.default_rng(0)
    references = {
        "random": decision_metrics(rng.random(dataset.number_of_candidates), dataset)
    }

    for name in ["fractionality", "neighbour_lp_sum", "fractional_degree"]:
        if name in dataset.feature_names:
            column = dataset.features[:, dataset.feature_names.index(name)]
            references[f"largest {name}"] = decision_metrics(column, dataset)

    return references


class CandidateScorer(torch.nn.Module):
    """Standardization followed by a small fully connected network."""

    def __init__(self, mean: np.ndarray, std: np.ndarray, hidden: list[int]):
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32))
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32))

        sizes = [len(mean), *hidden, 1]
        layers: list[torch.nn.Module] = []

        for index in range(len(sizes) - 1):
            layers.append(torch.nn.Linear(sizes[index], sizes[index + 1]))

            if index < len(sizes) - 2:
                layers.append(torch.nn.ReLU())

        self.network = torch.nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network((features - self.mean) / self.std).squeeze(-1)

    def export(self, feature_names: list[str]) -> BranchingModel:
        linear = [m for m in self.network if isinstance(m, torch.nn.Linear)]
        layers = []

        for index, layer in enumerate(linear):
            activation = "relu" if index < len(linear) - 1 else "linear"
            layers.append(_dense_layer(layer, activation))

        return BranchingModel(
            feature_names,
            self.mean.cpu().numpy().astype(np.float64),
            self.std.cpu().numpy().astype(np.float64),
            layers,
        )


def _dense_layer(layer: torch.nn.Linear, activation: str) -> Layer:
    return Layer(
        DENSE,
        layer.weight.detach().cpu().numpy().astype(np.float64),
        layer.bias.detach().cpu().numpy().astype(np.float64),
        activation,
    )


class GraphLayer(torch.nn.Module):
    """h'_v = relu(W_self h_v + W_neighbours mean of the neighbours of v
                   + W_node mean of all the candidates of the node + b)."""

    def __init__(self, inputs: int, outputs: int):
        super().__init__()
        self.own = torch.nn.Linear(inputs, outputs)
        self.neighbours = torch.nn.Linear(inputs, outputs, bias=False)
        self.node = torch.nn.Linear(inputs, outputs, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        sample_of_row: torch.Tensor,
        sample_size: torch.Tensor,
        sources: torch.Tensor,
        targets: torch.Tensor,
        degree: torch.Tensor,
    ) -> torch.Tensor:
        # Mean over no neighbour is 0: degree is clamped to 1.
        neighbours = torch.zeros_like(x).index_add_(0, targets, x[sources]) / degree
        node = x.new_zeros((len(sample_size), x.shape[1])).index_add_(0, sample_of_row, x)
        node = node / sample_size

        return torch.relu(
            self.own(x) + self.neighbours(neighbours) + self.node(node[sample_of_row])
        )


class GraphScorer(torch.nn.Module):
    """Standardization, dense layers on each candidate, graph layers over
    the candidates of the node, then a linear output."""

    def __init__(
        self, mean: np.ndarray, std: np.ndarray, hidden: list[int], graph_layers: int
    ):
        super().__init__()

        if not hidden:
            raise ValueError("A graph model needs at least one hidden layer")

        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32))
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32))

        sizes = [len(mean), *hidden]
        self.dense = torch.nn.ModuleList(
            torch.nn.Linear(sizes[i], sizes[i + 1]) for i in range(len(hidden))
        )
        self.graph = torch.nn.ModuleList(
            GraphLayer(hidden[-1], hidden[-1]) for _ in range(graph_layers)
        )
        self.output = torch.nn.Linear(hidden[-1], 1)

    def forward(self, batch: GraphBatch) -> torch.Tensor:
        device = self.mean.device
        x = (torch.from_numpy(batch.features).to(device) - self.mean) / self.std

        for layer in self.dense:
            x = torch.relu(layer(x))

        sample_of_row = torch.from_numpy(batch.sample_of_row).to(device)
        sources = torch.from_numpy(batch.sources).to(device)
        targets = torch.from_numpy(batch.targets).to(device)
        sample_size = torch.bincount(
            sample_of_row, minlength=batch.number_of_samples
        ).unsqueeze(1).to(x.dtype)
        degree = torch.bincount(targets, minlength=len(x)).clamp(min=1).unsqueeze(1).to(x.dtype)

        for layer in self.graph:
            x = layer(x, sample_of_row, sample_size, sources, targets, degree)

        return self.output(x).squeeze(-1)

    def export(self, feature_names: list[str]) -> BranchingModel:
        layers = [_dense_layer(layer, "relu") for layer in self.dense]

        for layer in self.graph:
            weights = np.stack(
                [
                    layer.own.weight.detach().cpu().numpy(),
                    layer.neighbours.weight.detach().cpu().numpy(),
                    layer.node.weight.detach().cpu().numpy(),
                ]
            ).astype(np.float64)
            bias = layer.own.bias.detach().cpu().numpy().astype(np.float64)
            layers.append(Layer(GRAPH, weights, bias, "relu"))

        layers.append(_dense_layer(self.output, "linear"))

        return BranchingModel(
            feature_names,
            self.mean.cpu().numpy().astype(np.float64),
            self.std.cpu().numpy().astype(np.float64),
            layers,
        )


class MessagePassing(torch.nn.Module):
    """t' = relu(P mean(relu(L t + R s + b_hidden), s source of t) + Q t + b)."""

    def __init__(self, direction: str, targets: int, sources: int, hidden: int, outputs: int):
        super().__init__()
        self.direction = direction
        self.target = torch.nn.Linear(targets, hidden)
        self.source = torch.nn.Linear(sources, hidden, bias=False)
        self.message = torch.nn.Linear(hidden, outputs)
        self.own = torch.nn.Linear(targets, outputs, bias=False)

    def forward(
        self,
        targets: torch.Tensor,
        sources: torch.Tensor,
        target_index: torch.Tensor,
        source_index: torch.Tensor,
        count: torch.Tensor | float,
    ) -> torch.Tensor:
        messages = torch.relu(self.target(targets)[target_index] + self.source(sources)[source_index])
        mean = messages.new_zeros((len(targets), messages.shape[1])).index_add_(
            0, target_index, messages
        ) / count
        return torch.relu(self.message(mean) + self.own(targets))

    def export(self) -> MessageLayer:
        def array(tensor: torch.Tensor) -> np.ndarray:
            return tensor.detach().cpu().numpy().astype(np.float64)

        return MessageLayer(
            self.direction,
            array(self.target.weight),
            array(self.source.weight),
            array(self.target.bias),
            array(self.message.weight),
            array(self.own.weight),
            array(self.message.bias),
        )


class BipartiteScorer(torch.nn.Module):
    """Bipartite model over the free vertices and the active edge
    constraints (after Gasse et al. 2019): dense layers on each side, then
    `rounds` pairs of message layers, vertices to constraints and back, and
    a linear output on the candidates."""

    def __init__(
        self,
        mean: np.ndarray,
        std: np.ndarray,
        vertex_mean: np.ndarray,
        vertex_std: np.ndarray,
        constraint_mean: np.ndarray,
        constraint_std: np.ndarray,
        hidden: list[int],
        rounds: int,
    ):
        super().__init__()

        if not hidden:
            raise ValueError("A bipartite model needs at least one hidden layer")

        for name, values in [
            ("mean", mean), ("std", std), ("vertex_mean", vertex_mean), ("vertex_std", vertex_std),
            ("constraint_mean", constraint_mean), ("constraint_std", constraint_std),
        ]:
            self.register_buffer(name, torch.tensor(values, dtype=torch.float32))

        width = hidden[-1]
        sizes = [len(mean) + len(vertex_mean), *hidden]
        self.vertex = torch.nn.ModuleList(
            torch.nn.Linear(sizes[i], sizes[i + 1]) for i in range(len(hidden))
        )
        self.constraint = torch.nn.ModuleList([torch.nn.Linear(len(constraint_mean), width)])
        self.messages = torch.nn.ModuleList()

        for _ in range(rounds):
            self.messages.append(MessagePassing("to_constraints", width, width, width, width))
            self.messages.append(MessagePassing("to_vertices", width, width, width, width))

        self.output = torch.nn.Linear(width, 1)

    def forward(self, batch: BipartiteBatch) -> torch.Tensor:
        device = self.mean.device

        def tensor(array: np.ndarray) -> torch.Tensor:
            return torch.from_numpy(array).to(device)

        features = (tensor(batch.candidate_features) - self.mean) / self.std
        features = features * tensor(batch.is_candidate).unsqueeze(1).to(features.dtype)
        v = torch.cat([features, (tensor(batch.vertex_inputs) - self.vertex_mean) / self.vertex_std], 1)
        c = (tensor(batch.constraint_inputs) - self.constraint_mean) / self.constraint_std

        for layer in self.vertex:
            v = torch.relu(layer(v))

        for layer in self.constraint:
            c = torch.relu(layer(c))

        endpoints = tensor(batch.endpoints)
        incident_vertex = endpoints.t().reshape(-1)
        incident_constraint = torch.arange(len(c), device=device).repeat(2)
        degree = torch.bincount(incident_vertex, minlength=len(v)).clamp(min=1).unsqueeze(1).to(v.dtype)

        for layer in self.messages:
            if layer.direction == "to_constraints":
                c = layer(c, v, incident_constraint, incident_vertex, 2.0)
            else:
                v = layer(v, c, incident_vertex, incident_constraint, degree)

        return self.output(v[tensor(batch.candidate_rows)]).squeeze(-1)

    def export(self, feature_names: list[str]) -> BranchingModel:
        def array(tensor: torch.Tensor) -> np.ndarray:
            return tensor.detach().cpu().numpy().astype(np.float64)

        encoder = BipartiteEncoder(
            list(VERTEX_INPUT_NAMES), array(self.vertex_mean), array(self.vertex_std),
            list(CONSTRAINT_INPUT_NAMES), array(self.constraint_mean), array(self.constraint_std),
            [_dense_layer(layer, "relu") for layer in self.vertex],
            [_dense_layer(layer, "relu") for layer in self.constraint],
            [layer.export() for layer in self.messages],
        )

        return BranchingModel(
            feature_names,
            array(self.mean),
            array(self.std),
            [_dense_layer(self.output, "linear")],
            encoder,
        )


def bipartite_input_statistics(
    dataset: BranchingDataset,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Mean and standard deviation of the vertex and constraint inputs over
    every free vertex and active constraint of the dataset."""
    sums = {"vertex": [0, 0.0, 0.0], "constraint": [0, 0.0, 0.0]}

    for start in range(0, dataset.number_of_samples, GRAPH_BLOCK):
        batch = dataset.bipartite_batch(np.arange(start, min(start + GRAPH_BLOCK, dataset.number_of_samples)))

        for side, values in [("vertex", batch.vertex_inputs), ("constraint", batch.constraint_inputs)]:
            values = values.astype(np.float64)
            sums[side][0] += len(values)
            sums[side][1] = sums[side][1] + values.sum(axis=0)
            sums[side][2] = sums[side][2] + (values * values).sum(axis=0)

    statistics = []

    for side in ["vertex", "constraint"]:
        count, total, squares = sums[side]
        mean = total / max(count, 1)
        std = np.sqrt(np.maximum(squares / max(count, 1) - mean * mean, 0.0))
        std[std < 1e-12] = 1.0
        statistics.extend([mean, std])

    return tuple(statistics)  # type: ignore[return-value]


# Samples per block when a graph model scores a whole dataset.
GRAPH_BLOCK = 1024


def predict(model: torch.nn.Module, dataset: BranchingDataset) -> np.ndarray:
    """Score of every candidate, laid out like dataset.scores."""
    model.eval()
    outputs = []
    device = model.mean.device

    with torch.no_grad():
        if isinstance(model, BipartiteScorer):
            for start in range(0, dataset.number_of_samples, GRAPH_BLOCK):
                samples = np.arange(start, min(start + GRAPH_BLOCK, dataset.number_of_samples))
                outputs.append(model(dataset.bipartite_batch(samples)).cpu().numpy())
        elif isinstance(model, GraphScorer):
            for start in range(0, dataset.number_of_samples, GRAPH_BLOCK):
                samples = np.arange(start, min(start + GRAPH_BLOCK, dataset.number_of_samples))
                outputs.append(model(dataset.graph_batch(samples)).cpu().numpy())
        else:
            for start in range(0, dataset.number_of_candidates, 65536):
                block = torch.from_numpy(dataset.features[start : start + 65536]).to(device)
                outputs.append(model(block).cpu().numpy())

    return np.concatenate(outputs) if outputs else np.zeros(0, np.float32)


def file_scores(model: BranchingModel, dataset: BranchingDataset) -> np.ndarray:
    """Score of every candidate under a model read from its file.

    Computed by blocks: the double precision activations of a whole
    validation set would take several gigabytes.
    """
    if model.is_bipartite:
        parts = []

        for start in range(0, dataset.number_of_samples, GRAPH_BLOCK // 4):
            samples = np.arange(start, min(start + GRAPH_BLOCK // 4, dataset.number_of_samples))
            batch = dataset.bipartite_batch(samples)
            parts.append(
                model.bipartite_scores(
                    batch.candidate_features,
                    batch.is_candidate,
                    batch.vertex_inputs,
                    batch.constraint_inputs,
                    batch.endpoints,
                    batch.candidate_rows,
                )
            )

        return np.concatenate(parts)

    if model.has_graph_layers:
        parts = []

        for start in range(0, dataset.number_of_samples, GRAPH_BLOCK // 2):
            samples = np.arange(start, min(start + GRAPH_BLOCK // 2, dataset.number_of_samples))
            batch = dataset.graph_batch(samples)
            parts.append(
                model.graph_scores(
                    batch.features,
                    batch.sample_of_row,
                    batch.number_of_samples,
                    batch.sources,
                    batch.targets,
                )
            )

        return np.concatenate(parts)

    return np.concatenate(
        [
            model.scores(dataset.features[start : start + 65536])
            for start in range(0, dataset.number_of_candidates, 65536)
        ]
    )


def batch_scores(
    model: torch.nn.Module, dataset: BranchingDataset, batch: np.ndarray
) -> torch.Tensor:
    """Model scores of the candidates of some samples, [samples, width],
    padded with a very low score."""
    if isinstance(model, BipartiteScorer):
        bipartite = dataset.bipartite_batch(batch)
        flat = model(bipartite)
        scores = flat.new_full((len(batch), int(bipartite.position.max()) + 1), -1e9)
        rows = torch.from_numpy(bipartite.sample_of_candidate).to(flat.device)
        columns = torch.from_numpy(bipartite.position).to(flat.device)
        scores[rows, columns] = flat
        return scores

    if isinstance(model, GraphScorer):
        graph = dataset.graph_batch(batch)
        flat = model(graph)
        scores = flat.new_full((len(batch), int(graph.position.max()) + 1), -1e9)
        rows = torch.from_numpy(graph.sample_of_row).to(flat.device)
        columns = torch.from_numpy(graph.position).to(flat.device)
        scores[rows, columns] = flat
        return scores

    # Padded to the largest number of candidates of the dataset: every
    # batch has the same shape.
    device = model.mean.device
    features, mask = dataset.padded_batch(batch)
    return model(torch.from_numpy(features).to(device)).masked_fill(
        ~torch.from_numpy(mask).to(device), -1e9
    )


def second_best_mask(dataset: BranchingDataset) -> np.ndarray:
    """Per candidate: whether it is one of the second-best candidates of
    its sample, by strong branching score: the best of the candidates other
    than the expert's choice, ties included."""
    starts, lengths = dataset.starts, dataset.lengths
    scores = dataset.scores.astype(np.float64)
    chosen = starts + dataset.target
    others = scores.copy()
    others[chosen] = -np.inf
    best_other = np.maximum.reduceat(others, starts)
    threshold = best_other - 1e-9 * np.maximum(1.0, np.abs(best_other))
    mask = (others >= np.repeat(threshold, lengths)) & np.repeat(lengths > 1, lengths)
    mask[chosen] = False
    return mask


def lookback_samples(dataset: BranchingDataset, second_best: np.ndarray) -> np.ndarray:
    """Per sample: whether the lookback condition holds (Gupta et al. 2022):
    the expert's choice is one of the second-best candidates of the parent
    node. Requires with_parents()."""
    assert dataset.parent is not None and dataset.parent_positions is not None
    lookback = np.zeros(dataset.number_of_samples, bool)
    has_parent = np.flatnonzero(dataset.parent >= 0)
    at_parent = dataset.parent_positions[dataset.starts[has_parent] + dataset.target[has_parent]]
    found = at_parent >= 0
    rows = dataset.starts[dataset.parent[has_parent[found]]] + at_parent[found]
    lookback[has_parent[found]] = second_best[rows]
    return lookback


def padded_positions(dataset: BranchingDataset, samples: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Row (sample within samples), column (candidate position) and
    dataset row of every candidate of some samples."""
    lengths = dataset.lengths[samples]
    rows = np.repeat(np.arange(len(samples)), lengths)
    columns = np.arange(int(lengths.sum())) - np.repeat(np.cumsum(lengths) - lengths, lengths)
    return rows, columns, dataset.starts[samples][rows] + columns


def lookback_loss(
    model: torch.nn.Module,
    dataset: BranchingDataset,
    batch: np.ndarray,
    scores: torch.Tensor,
    second_best: np.ndarray,
    lookback: np.ndarray | None,
    smoothing: float,
    lookback_weight: float,
) -> torch.Tensor:
    """Cross-entropy towards the smoothed target, 1 - smoothing on the
    expert's choice and smoothing shared by the second-best candidates of
    the node, plus lookback_weight times the cross-entropy towards the
    model's own distribution at the parent node, restricted to the
    candidates of the child, on the samples where the lookback condition
    holds (the Parent-as-Target term of Gupta et al. 2022; the parent's
    scores are a fixed target, without gradient)."""
    device = scores.device
    log_probabilities = torch.log_softmax(scores, dim=1)

    rows, columns, flat = padded_positions(dataset, batch)
    in_second_best = second_best[flat]
    count = np.bincount(rows[in_second_best], minlength=len(batch))
    weights = np.zeros(scores.shape, np.float32)
    weights[np.arange(len(batch)), dataset.target[batch]] = np.where(count > 0, 1.0 - smoothing, 1.0)
    weights[rows[in_second_best], columns[in_second_best]] = smoothing / count[rows[in_second_best]]
    loss = -(torch.from_numpy(weights).to(device) * log_probabilities).sum(1).mean()

    if lookback is None or lookback_weight <= 0:
        return loss

    children = np.flatnonzero(lookback[batch])

    if len(children) == 0:
        return loss

    with torch.no_grad():
        parent_scores = batch_scores(model, dataset, dataset.parent[batch[children]])

    rows, columns, flat = padded_positions(dataset, batch[children])
    at_parent = dataset.parent_positions[flat]
    known = at_parent >= 0
    parent_logits = torch.full((len(children), scores.shape[1]), float("-inf"), device=device)
    parent_logits[
        torch.from_numpy(rows[known]).to(device), torch.from_numpy(columns[known]).to(device)
    ] = parent_scores[
        torch.from_numpy(rows[known]).to(device), torch.from_numpy(at_parent[known]).to(device)
    ]
    parent_distribution = torch.softmax(parent_logits, dim=1)
    child_log_probabilities = log_probabilities[torch.from_numpy(children).to(device)]
    parent_as_target = -(
        torch.nan_to_num(parent_distribution) * child_log_probabilities
    ).sum(1).mean()

    return loss + lookback_weight * parent_as_target


def train(
    train_set: BranchingDataset,
    valid_set: BranchingDataset,
    hidden: list[int],
    graph_layers: int,
    bipartite_rounds: int,
    epochs: int,
    learning_rate: float,
    batch_size: int,
    seed: int,
    device: torch.device,
    smoothing: float = 0.0,
    lookback_weight: float = 0.0,
) -> tuple[torch.nn.Module, list[dict[str, float]]]:
    torch.manual_seed(seed)

    # Standardize with the statistics of the training candidates.
    # Accumulated in double precision, by blocks of samples, to avoid a
    # second copy of the training set.
    count = 0
    total = np.zeros(train_set.features.shape[1])
    total_of_squares = np.zeros_like(total)
    offsets = train_set.offsets

    for start in range(0, train_set.number_of_samples, 4096):
        stop = min(start + 4096, train_set.number_of_samples)
        block = train_set.features[offsets[start] : offsets[stop]].astype(np.float64)
        count += len(block)
        total += block.sum(axis=0)
        total_of_squares += (block * block).sum(axis=0)

    mean = total / count
    std = np.sqrt(np.maximum(total_of_squares / count - mean * mean, 0.0))
    std[std < 1e-12] = 1.0

    model: torch.nn.Module

    if bipartite_rounds > 0:
        model = BipartiteScorer(
            mean, std, *bipartite_input_statistics(train_set), hidden, bipartite_rounds
        )
    elif graph_layers > 0:
        model = GraphScorer(mean, std, hidden, graph_layers)
    else:
        model = CandidateScorer(mean, std, hidden)

    # Initialized on the CPU, then moved: the same seed gives the same
    # initial weights on every device.
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    target = torch.from_numpy(train_set.target).to(device)

    # Lookback (Gupta et al. 2022): smoothed targets and the Parent-as-Target
    # term. With both off, the loss is the plain cross-entropy below.
    second_best = second_best_mask(train_set) if smoothing > 0 or lookback_weight > 0 else None
    lookback = lookback_samples(train_set, second_best) if lookback_weight > 0 else None

    if lookback is not None:
        print(f"lookback condition on {int(lookback.sum())} of {train_set.number_of_samples} training samples")

    history = []
    best_ratio = -1.0
    best_state = None

    for epoch in range(1, epochs + 1):
        epoch_started = time.time()
        model.train()
        order = torch.randperm(train_set.number_of_samples)
        total_loss = 0.0

        for start in range(0, len(order), batch_size):
            batch = order[start : start + batch_size]
            scores = batch_scores(model, train_set, batch.numpy())

            if second_best is None:
                # The expert's choice should get the highest score of its node.
                loss = torch.nn.functional.cross_entropy(scores, target[batch.to(device)])
            else:
                loss = lookback_loss(
                    model, train_set, batch.numpy(), scores, second_best, lookback,
                    smoothing, lookback_weight,
                )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(batch)

        metrics = decision_metrics(predict(model, valid_set), valid_set)
        metrics["epoch"] = epoch
        metrics["train_loss"] = total_loss / len(order)
        history.append(metrics)

        print(
            f"epoch {epoch:3d}  loss {metrics['train_loss']:.4f}  "
            f"valid accuracy {metrics['accuracy']:.3f}  "
            f"top5 {metrics['top5']:.3f}  "
            f"score ratio {metrics['score_ratio']:.4f}  "
            f"({time.time() - epoch_started:.0f} s)",
            flush=True,
        )

        # Keep the epoch whose decisions are closest to the expert's.
        if metrics["score_ratio"] > best_ratio:
            best_ratio = metrics["score_ratio"]
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    assert best_state is not None
    model.load_state_dict(best_state)

    return model, history


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--train", required=True, nargs="+", help="directories of training feature tables")
    parser.add_argument("--valid", required=True, nargs="+", help="directories of validation feature tables")
    parser.add_argument("--output", required=True, help="path of the model file to write")
    parser.add_argument("--features", nargs="+", help="names of the features the model reads (default: all those of the tables)")
    parser.add_argument("--hidden", type=int, nargs="*", default=[32, 32], help="hidden layer sizes; none for a linear model")
    parser.add_argument("--graph-layers", type=int, default=0, help="graph layers after the hidden layers (0: each candidate is scored on its own)")
    parser.add_argument("--bipartite-rounds", type=int, default=0, help="bipartite model over the free vertices and the edge constraints, with that many rounds of messages (vertices to constraints and back)")
    parser.add_argument("--instances", default="data/instances", help="directory searched for the instance files of the graphs (graph models only)")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--smoothing", type=float, default=0.0, help="lookback target smoothing: share of the target given to the second-best candidates of the node (Gupta et al. 2022)")
    parser.add_argument("--lookback-weight", type=float, default=0.0, help="weight of the Parent-as-Target lookback term (Gupta et al. 2022); needs the raw samples")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu", help="where the network is trained (cuda: the GPU, needs a CUDA build of PyTorch; results are then not bit-for-bit reproducible)")
    arguments = parser.parse_args()

    if arguments.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda: no GPU usable by this PyTorch build")

    output = pathlib.Path(arguments.output)

    if output.exists():
        raise SystemExit(f"Output file already exists: {output}")

    started = time.time()
    train_set = load_feature_tables(arguments.train)
    valid_set = load_feature_tables(arguments.valid)

    if train_set.feature_names != valid_set.feature_names:
        raise SystemExit("Training and validation feature sets differ")

    if arguments.features:
        unknown = sorted(set(arguments.features) - set(train_set.feature_names))

        if unknown:
            raise SystemExit(f"Unknown features: {', '.join(unknown)}")

        train_set = train_set.with_features(arguments.features)
        valid_set = valid_set.with_features(arguments.features)

    if arguments.graph_layers < 0:
        raise SystemExit("--graph-layers must be at least 0")

    if arguments.bipartite_rounds < 0:
        raise SystemExit("--bipartite-rounds must be at least 0")

    if arguments.graph_layers > 0 and arguments.bipartite_rounds > 0:
        raise SystemExit("--graph-layers and --bipartite-rounds are exclusive")

    if (arguments.graph_layers > 0 or arguments.bipartite_rounds > 0) and not arguments.hidden:
        raise SystemExit("A graph model needs at least one hidden layer")

    if arguments.graph_layers > 0:
        train_set = train_set.with_candidate_graphs(arguments.instances)
        valid_set = valid_set.with_candidate_graphs(arguments.instances)

    if arguments.bipartite_rounds > 0:
        train_set = train_set.with_vertex_states(arguments.instances)
        valid_set = valid_set.with_vertex_states(arguments.instances)

    if not 0.0 <= arguments.smoothing < 1.0 or arguments.lookback_weight < 0.0:
        raise SystemExit("--smoothing must be in [0, 1) and --lookback-weight at least 0")

    if arguments.lookback_weight > 0:
        train_set = train_set.with_parents()

    for name, dataset in [("train", train_set), ("valid", valid_set)]:
        edges = (
            f", {len(dataset.edges)} edges between candidates"
            if dataset.edges is not None
            else ""
        )
        print(
            f"{name}: {len(dataset.instance_names)} feature tables, "
            f"{dataset.number_of_samples} usable samples of {dataset.total_samples}, "
            f"{dataset.number_of_candidates} candidates{edges}"
        )

    print(f"loaded in {time.time() - started:.0f} s\n")

    references = reference_metrics(valid_set)

    for name, metrics in references.items():
        print(
            f"{name:28s} valid accuracy {metrics['accuracy']:.3f}  "
            f"top5 {metrics['top5']:.3f}  score ratio {metrics['score_ratio']:.4f}"
        )

    print()

    model, history = train(
        train_set,
        valid_set,
        arguments.hidden,
        arguments.graph_layers,
        arguments.bipartite_rounds,
        arguments.epochs,
        arguments.learning_rate,
        arguments.batch_size,
        arguments.seed,
        torch.device(arguments.device),
        arguments.smoothing,
        arguments.lookback_weight,
    )

    exported = model.export(train_set.feature_names)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_model(output, exported)

    # The file is what SCIP will use: check that it reproduces the
    # decisions of the trained network.
    from_torch = predict(model, valid_set)
    from_file = file_scores(read_model(output), valid_set)
    file_metrics = decision_metrics(from_file, valid_set)

    same_pick = float(
        (
            segment_argmax(from_file, valid_set.offsets)
            == segment_argmax(from_torch, valid_set.offsets)
        ).mean()
    )

    print(
        f"\nexported model: valid accuracy {file_metrics['accuracy']:.3f}  "
        f"top5 {file_metrics['top5']:.3f}  "
        f"score ratio {file_metrics['score_ratio']:.4f}  "
        f"mean rank {file_metrics['mean_rank']:.2f}"
    )
    print(f"same decision as the trained network on {100 * same_pick:.2f}% of samples")

    if same_pick < 0.999:
        raise SystemExit("The exported file does not reproduce the trained network")

    # Model selection uses all validation samples together; the details
    # show where the model is good: on the nodes the expert visits, or on
    # those a learned policy leads to.
    valid_by_directory = {}

    if len(arguments.valid) > 1:
        for index, directory in enumerate(arguments.valid):
            part = valid_set.from_source(index)
            rows = slice(
                int(valid_set.offsets[np.searchsorted(valid_set.source, index)]),
                int(valid_set.offsets[np.searchsorted(valid_set.source, index, "right")]),
            )
            metrics = decision_metrics(from_file[rows], part)
            valid_by_directory[directory] = metrics
            print(
                f"  {directory}: {part.number_of_samples} samples, "
                f"accuracy {metrics['accuracy']:.3f}  top5 {metrics['top5']:.3f}  "
                f"score ratio {metrics['score_ratio']:.4f}"
            )

    summary = {
        "arguments": vars(arguments),
        "train_samples": train_set.number_of_samples,
        "valid_samples": valid_set.number_of_samples,
        "references": references,
        "history": history,
        "exported": file_metrics,
        "exported_by_valid_directory": valid_by_directory,
    }
    output.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"model: {output}")


if __name__ == "__main__":
    main()
