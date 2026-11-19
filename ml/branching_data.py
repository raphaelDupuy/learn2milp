"""Loading of the branching feature tables written by mis-featurizer.

A feature table has one row per branching candidate. This module groups
the rows by sample (one branching decision) and keeps the samples in
which the expert's choice is meaningful.

The candidates of all samples are stored one after the other, without
padding: graph families with many candidates per node (Barabasi-Albert
graphs have twice as many as Erdos-Renyi ones of the same density) would
otherwise make every sample as large as the largest one. Padded arrays
are built only for small groups of samples (padded_batch()).

Graph models also need, for every sample, the edges of the graph between
its candidates. with_candidate_graphs() reads them from the instance
files, found by name: the feature tables keep the vertex of every
candidate.

Bipartite models (variables and constraints, as in Gasse et al. 2019)
also need the state of every vertex at every node, which only the raw
samples record: with_vertex_states() reads it from the node tables of
mis-collector, in the directory that matches each feature directory
(data/raw/strong_branching/X for data/processed/features/X).
"""

from __future__ import annotations

import dataclasses
import pathlib

import numpy as np
import pandas as pd

# Columns that precede the features in a feature table.
LABEL_COLUMNS = [
    "sample_index",
    "candidate_position",
    "vertex_id",
    "node_cutoff",
    "expert_selection_valid",
    "evaluated",
    "score",
    "expert_rank",
    "expert_selected",
]

FEATURES_SUFFIX = ".sb.features.csv"


@dataclasses.dataclass
class BranchingDataset:
    """Samples whose candidates are stored one after the other.

    features  [candidates, features], float32
    scores    [candidates], strong branching score of each candidate
    vertices  [candidates], graph vertex of each candidate
    offsets   [samples + 1], the candidates of sample s are the rows
              offsets[s] to offsets[s + 1] - 1, in candidate order
    target    [samples], position, within its sample, of the candidate
              chosen by the expert
    instance  [samples], index of the instance each sample comes from
    source    [samples], index of the directory each sample comes from;
              the samples of one directory are consecutive
    lp_values [candidates], LP value of each candidate
    sample_ids
              [samples], sample_index of each sample in its tables

    After with_candidate_graphs() only:

    edges         [edges, 2], int16, the graph edges between the
                  candidates of each sample, as two positions within the
                  sample, the first one smaller
    edge_offsets  [samples + 1], the edges of sample s are the rows
                  edge_offsets[s] to edge_offsets[s + 1] - 1

    After with_vertex_states() only:

    states        uint8, the state of every vertex at every sample (codes
                  of VERTEX_STATES), one sample after the other
    state_offsets [samples + 1], the states of sample s are
                  states[state_offsets[s] : state_offsets[s + 1]]
    instance_edges
                  per instance, its edges [m, 2]

    After with_parents() only:

    parent        [samples], dataset index of the sample of the parent node
                  (the node whose branching created this one), -1 if the
                  parent is not a usable sample of the dataset
    parent_positions
                  [candidates], position of each candidate among the
                  candidates of the parent sample, -1 if it was not one
    """

    feature_names: list[str]
    features: np.ndarray
    scores: np.ndarray
    vertices: np.ndarray
    offsets: np.ndarray
    target: np.ndarray
    instance: np.ndarray
    source: np.ndarray
    instance_names: list[str]
    total_samples: int
    lp_values: np.ndarray
    sample_ids: np.ndarray
    source_directories: list[str]
    edges: np.ndarray | None = None
    edge_offsets: np.ndarray | None = None
    states: np.ndarray | None = None
    state_offsets: np.ndarray | None = None
    instance_edges: list[np.ndarray] | None = None
    parent: np.ndarray | None = None
    parent_positions: np.ndarray | None = None

    @property
    def number_of_samples(self) -> int:
        return len(self.offsets) - 1

    @property
    def number_of_candidates(self) -> int:
        return int(self.offsets[-1] - self.offsets[0])

    @property
    def starts(self) -> np.ndarray:
        """Row of the first candidate of every sample."""
        return self.offsets[:-1]

    @property
    def lengths(self) -> np.ndarray:
        """Number of candidates of every sample."""
        return np.diff(self.offsets)

    @property
    def max_candidates(self) -> int:
        return int(self.lengths.max()) if self.number_of_samples else 0

    def padded_batch(
        self, samples: np.ndarray, width: int | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Features of some samples, padded to the same number of candidates.

        Returns features [len(samples), width, features], 0 where padded,
        and mask [len(samples), width], True for real candidates. width
        defaults to the largest number of candidates of the dataset, so
        that every batch has the same shape.
        """
        width = self.max_candidates if width is None else width
        lengths = self.lengths[samples]
        rows = np.repeat(np.arange(len(samples)), lengths)
        columns = np.arange(int(lengths.sum())) - np.repeat(
            np.cumsum(lengths) - lengths, lengths
        )
        sources = np.repeat(self.starts[samples], lengths) + columns

        features = np.zeros(
            (len(samples), width, self.features.shape[1]), np.float32
        )
        mask = np.zeros((len(samples), width), bool)
        features[rows, columns] = self.features[sources]
        mask[rows, columns] = True

        return features, mask

    def graph_batch(self, samples: np.ndarray) -> GraphBatch:
        """Candidates of some samples, without padding, and their edges.

        Requires with_candidate_graphs(). The candidates are laid out one
        sample after the other, in the order of samples; every edge is
        given in both directions.
        """
        if self.edges is None or self.edge_offsets is None:
            raise ValueError("Candidate graphs are not loaded")

        samples = np.asarray(samples, np.int64)
        lengths = self.lengths[samples]
        first_row = np.cumsum(lengths) - lengths
        sample_of_row = np.repeat(np.arange(len(samples)), lengths)
        position = np.arange(int(lengths.sum())) - first_row[sample_of_row]
        rows = self.starts[samples][sample_of_row] + position

        edge_lengths = self.edge_offsets[samples + 1] - self.edge_offsets[samples]
        first_edge = np.cumsum(edge_lengths) - edge_lengths
        sample_of_edge = np.repeat(np.arange(len(samples)), edge_lengths)
        edge_rows = (
            self.edge_offsets[samples][sample_of_edge]
            + np.arange(int(edge_lengths.sum()))
            - first_edge[sample_of_edge]
        )
        ends = self.edges[edge_rows].astype(np.int64) + first_row[sample_of_edge, None]

        return GraphBatch(
            features=self.features[rows],
            sample_of_row=sample_of_row,
            position=position,
            sources=np.concatenate([ends[:, 0], ends[:, 1]]),
            targets=np.concatenate([ends[:, 1], ends[:, 0]]),
            number_of_samples=len(samples),
        )

    def with_candidate_graphs(
        self, instances_root: str | pathlib.Path
    ) -> BranchingDataset:
        """The same samples, with the graph edges between their candidates.

        The graph of every feature table is read from the instance file of
        the same name, searched for under instances_root.
        """
        # Positions within a sample are stored as int16.
        if self.max_candidates > np.iinfo(np.int16).max:
            raise ValueError("Too many candidates in a sample for int16 positions")

        paths = find_instance_files(instances_root, self.instance_names)
        counts = np.zeros(self.number_of_samples, np.int64)
        parts = []

        for index, name in enumerate(self.instance_names):
            samples = np.flatnonzero(self.instance == index)

            if len(samples) == 0:
                continue

            number_of_vertices, graph_edges = read_instance_graph(paths[name])
            adjacency = np.zeros((number_of_vertices, number_of_vertices), bool)
            adjacency[graph_edges[:, 0], graph_edges[:, 1]] = True
            adjacency[graph_edges[:, 1], graph_edges[:, 0]] = True

            for sample in samples:
                vertices = self.vertices[self.offsets[sample] : self.offsets[sample + 1]]

                if vertices.max() >= number_of_vertices:
                    raise ValueError(f"Candidate vertex out of range in {name}")

                first, second = np.nonzero(np.triu(adjacency[np.ix_(vertices, vertices)], 1))
                parts.append(np.stack([first, second], axis=1).astype(np.int16))
                counts[sample] = len(first)

        edge_offsets = np.zeros(self.number_of_samples + 1, np.int64)
        edge_offsets[1:] = np.cumsum(counts)

        return dataclasses.replace(
            self,
            edges=np.concatenate(parts) if parts else np.zeros((0, 2), np.int16),
            edge_offsets=edge_offsets,
        )

    def with_features(self, names: list[str]) -> BranchingDataset:
        """The same samples, described by the given features only."""
        columns = [self.feature_names.index(name) for name in names]

        return dataclasses.replace(
            self,
            feature_names=list(names),
            features=np.ascontiguousarray(self.features[:, columns]),
        )

    def from_source(self, source: int) -> BranchingDataset:
        """The samples that come from one of the loaded directories."""
        chosen = np.flatnonzero(self.source == source)

        if len(chosen) == 0:
            first = last = 0
        else:
            first, last = int(chosen[0]), int(chosen[-1]) + 1

            if last - first != len(chosen):
                raise ValueError("Samples of a directory are not consecutive")

        rows = slice(int(self.offsets[first]), int(self.offsets[last]))
        edges, edge_offsets = self.edges, self.edge_offsets

        if edges is not None and edge_offsets is not None:
            edges = edges[int(edge_offsets[first]) : int(edge_offsets[last])]
            edge_offsets = edge_offsets[first : last + 1] - edge_offsets[first]

        states, state_offsets = self.states, self.state_offsets

        if states is not None and state_offsets is not None:
            states = states[int(state_offsets[first]) : int(state_offsets[last])]
            state_offsets = state_offsets[first : last + 1] - state_offsets[first]

        return dataclasses.replace(
            self,
            features=self.features[rows],
            scores=self.scores[rows],
            vertices=self.vertices[rows],
            lp_values=self.lp_values[rows],
            offsets=self.offsets[first : last + 1] - self.offsets[first],
            target=self.target[first:last],
            instance=self.instance[first:last],
            source=self.source[first:last],
            sample_ids=self.sample_ids[first:last],
            edges=edges,
            edge_offsets=edge_offsets,
            states=states,
            state_offsets=state_offsets,
            parent=None if self.parent is None else np.where(
                self.parent[first:last] >= 0, self.parent[first:last] - first, -1
            ),
            parent_positions=None if self.parent_positions is None else self.parent_positions[rows],
        )

    def with_vertex_states(
        self, instances_root: str | pathlib.Path
    ) -> BranchingDataset:
        """The same samples, with the state of every vertex at each of them.

        The states are read from the node tables of the raw samples, in the
        directory matching each feature directory (raw_samples_directory()),
        and the instance graphs from the instance files of the same names,
        searched for under instances_root.
        """
        paths = find_instance_files(instances_root, self.instance_names)
        instance_edges = []
        parts = []
        counts = np.zeros(self.number_of_samples, np.int64)

        for index, name in enumerate(self.instance_names):
            number_of_vertices, graph_edges = read_instance_graph(paths[name])
            instance_edges.append(graph_edges)
            samples = np.flatnonzero(self.instance == index)

            if len(samples) == 0:
                continue

            directory = raw_samples_directory(
                self.source_directories[int(self.source[samples[0]])]
            )
            nodes = pd.read_csv(
                directory / f"{name}.sb.nodes.csv",
                usecols=["sample_index", "number_of_vertices", "vertex_states"],
                dtype={"vertex_states": str},
            ).set_index("sample_index")

            for sample in samples:
                row = nodes.loc[int(self.sample_ids[sample])]
                text = row["vertex_states"]

                if int(row["number_of_vertices"]) != number_of_vertices or len(text) != number_of_vertices:
                    raise ValueError(f"Vertex states of {name} do not match its graph")

                codes = STATE_CODES[np.frombuffer(text.encode("ascii"), np.uint8)]
                candidates = self.vertices[self.offsets[sample] : self.offsets[sample + 1]]

                if (
                    codes.max() > 5
                    or (codes[candidates] != FRACTIONAL).any()
                    or int((codes == FRACTIONAL).sum()) != len(candidates)
                ):
                    raise ValueError(
                        f"Vertex states of {name}, sample {self.sample_ids[sample]}, "
                        "do not match its candidates"
                    )

                parts.append(codes)
                counts[sample] = number_of_vertices

        state_offsets = np.zeros(self.number_of_samples + 1, np.int64)
        state_offsets[1:] = np.cumsum(counts)

        return dataclasses.replace(
            self,
            states=np.concatenate(parts) if parts else np.zeros(0, np.uint8),
            state_offsets=state_offsets,
            instance_edges=instance_edges,
        )

    def with_parents(self) -> BranchingDataset:
        """The same samples, linked to the sample of their parent node.

        The node and parent identifiers are read from the node tables of
        the raw samples (see with_vertex_states()). A parent is looked up
        among the samples of the same table: the same search.
        """
        parent = np.full(self.number_of_samples, -1, np.int64)
        parent_positions = np.full(self.number_of_candidates, -1, np.int32)

        for index, name in enumerate(self.instance_names):
            samples = np.flatnonzero(self.instance == index)

            if len(samples) == 0:
                continue

            directory = raw_samples_directory(
                self.source_directories[int(self.source[samples[0]])]
            )
            nodes = pd.read_csv(
                directory / f"{name}.sb.nodes.csv",
                usecols=["sample_index", "node_id", "parent_node_id", "parent_node_valid"],
            ).set_index("sample_index")
            rows = nodes.loc[self.sample_ids[samples]]
            sample_of_node = dict(zip(rows["node_id"].to_numpy(), samples))

            for sample, parent_id, valid in zip(
                samples, rows["parent_node_id"].to_numpy(), rows["parent_node_valid"].to_numpy()
            ):
                found = sample_of_node.get(parent_id, -1) if valid else -1

                if found < 0:
                    continue

                parent[sample] = found
                parent_vertices = self.vertices[self.offsets[found] : self.offsets[found + 1]]
                position_of_vertex = {int(v): p for p, v in enumerate(parent_vertices)}
                child = slice(int(self.offsets[sample]), int(self.offsets[sample + 1]))
                parent_positions[child] = [
                    position_of_vertex.get(int(v), -1) for v in self.vertices[child]
                ]

        return dataclasses.replace(self, parent=parent, parent_positions=parent_positions)

    def bipartite_batch(self, samples: np.ndarray) -> BipartiteBatch:
        """Free vertices and active constraints of some samples.

        Requires with_vertex_states(). The graph of a node keeps its free
        vertices and the edge constraints x_u + x_v <= 1 between two free
        vertices; the others are satisfied whatever the free variables do.
        Vertices and constraints are laid out one sample after the other;
        candidate_rows gives the vertex row of every candidate, in the
        order of the dataset's candidates.
        """
        if self.states is None or self.state_offsets is None or self.instance_edges is None:
            raise ValueError("Vertex states are not loaded")

        samples = np.asarray(samples, np.int64)
        candidate_features, vertex_inputs, constraint_inputs = [], [], []
        endpoints, candidate_rows, is_candidate = [], [], []
        vertex_count = 0

        for sample in samples:
            states = self.states[self.state_offsets[sample] : self.state_offsets[sample + 1]]
            graph_edges = self.instance_edges[int(self.instance[sample])]
            rows = slice(int(self.offsets[sample]), int(self.offsets[sample + 1]))
            candidates = self.vertices[rows]
            n = len(states)

            free = (states >= FREE_LP_ZERO) & (states <= FRACTIONAL)
            fractional = states == FRACTIONAL
            x = np.where(states == FREE_LP_ONE, 1.0, 0.0)
            x[candidates] = self.lp_values[rows]

            active = free[graph_edges[:, 0]] & free[graph_edges[:, 1]]
            ends = graph_edges[active]
            degree = np.bincount(graph_edges.ravel(), minlength=n)
            free_degree = np.bincount(ends.ravel(), minlength=n)

            row_of_vertex = np.full(n, -1, np.int64)
            free_vertices = np.flatnonzero(free)
            row_of_vertex[free_vertices] = vertex_count + np.arange(len(free_vertices))

            vertex_inputs.append(
                np.stack(
                    [
                        states[free_vertices] == FREE_LP_ZERO,
                        states[free_vertices] == FREE_LP_ONE,
                        fractional[free_vertices],
                        x[free_vertices],
                        degree[free_vertices],
                        free_degree[free_vertices],
                    ],
                    axis=1,
                ).astype(np.float32)
            )
            features = np.zeros((len(free_vertices), self.features.shape[1]), np.float32)
            features[row_of_vertex[candidates] - vertex_count] = self.features[rows]
            candidate_features.append(features)
            is_candidate.append(fractional[free_vertices])

            slack = 1.0 - x[ends[:, 0]] - x[ends[:, 1]]
            constraint_inputs.append(
                np.stack(
                    [
                        slack,
                        slack <= TIGHT_TOLERANCE,
                        fractional[ends[:, 0]].astype(np.float64) + fractional[ends[:, 1]],
                    ],
                    axis=1,
                ).astype(np.float32)
            )
            endpoints.append(row_of_vertex[ends])
            candidate_rows.append(row_of_vertex[candidates])
            vertex_count += len(free_vertices)

        lengths = self.lengths[samples]

        return BipartiteBatch(
            candidate_features=np.concatenate(candidate_features),
            is_candidate=np.concatenate(is_candidate),
            vertex_inputs=np.concatenate(vertex_inputs),
            constraint_inputs=np.concatenate(constraint_inputs),
            endpoints=np.concatenate(endpoints).reshape(-1, 2),
            candidate_rows=np.concatenate(candidate_rows),
            sample_of_candidate=np.repeat(np.arange(len(samples)), lengths),
            position=np.arange(int(lengths.sum())) - np.repeat(np.cumsum(lengths) - lengths, lengths),
            number_of_samples=len(samples),
        )


@dataclasses.dataclass
class GraphBatch:
    """Candidates of a group of samples, for a graph model.

    features       [rows, features], one row per candidate
    sample_of_row  [rows], sample (0 .. number_of_samples - 1) of each row
    position       [rows], position of each candidate within its sample
    sources, targets
                   [2 * edges], rows linked by a graph edge, in both
                   directions: row sources[i] sends to row targets[i]
    """

    features: np.ndarray
    sample_of_row: np.ndarray
    position: np.ndarray
    sources: np.ndarray
    targets: np.ndarray
    number_of_samples: int


# Vertex states, as mis-collector writes them (MISVertexState): one
# character per vertex in the node table, a code 0..5 here.
VERTEX_STATES = "ZOzofu"
FIXED_ZERO, FIXED_ONE, FREE_LP_ZERO, FREE_LP_ONE, FRACTIONAL, UNKNOWN = range(6)
STATE_CODES = np.full(256, 255, np.uint8)
STATE_CODES[np.frombuffer(VERTEX_STATES.encode("ascii"), np.uint8)] = np.arange(6)

# An edge constraint is tight below this slack (as in the C features).
TIGHT_TOLERANCE = 1e-6

# Inputs of the bipartite models, beyond the candidate features: names as
# in the model files, in order.
VERTEX_INPUT_NAMES = ["free_lp_zero", "free_lp_one", "fractional", "lp_value", "degree", "free_degree"]
CONSTRAINT_INPUT_NAMES = ["slack", "tight", "fractional_endpoints"]


@dataclasses.dataclass
class BipartiteBatch:
    """Free vertices and active constraints of a group of samples.

    candidate_features  [vertices, features], the features of the
                        candidates, 0 for the other free vertices
    is_candidate        [vertices]
    vertex_inputs       [vertices, len(VERTEX_INPUT_NAMES)]
    constraint_inputs   [constraints, len(CONSTRAINT_INPUT_NAMES)]
    endpoints           [constraints, 2], vertex rows of each constraint
    candidate_rows      [candidates], vertex row of each candidate
    sample_of_candidate, position
                        [candidates], sample and position of each candidate
    """

    candidate_features: np.ndarray
    is_candidate: np.ndarray
    vertex_inputs: np.ndarray
    constraint_inputs: np.ndarray
    endpoints: np.ndarray
    candidate_rows: np.ndarray
    sample_of_candidate: np.ndarray
    position: np.ndarray
    number_of_samples: int


def raw_samples_directory(features_directory: str | pathlib.Path) -> pathlib.Path:
    """Directory of the raw samples a feature directory was computed from:
    data/raw/strong_branching/X for data/processed/features/X."""
    parts = pathlib.Path(features_directory).resolve().parts

    for i in range(len(parts) - 1):
        if parts[i : i + 2] == ("processed", "features"):
            directory = pathlib.Path(*parts[:i], "raw", "strong_branching", *parts[i + 2 :])

            if not directory.is_dir():
                raise FileNotFoundError(f"Raw samples of {features_directory} not found in {directory}")

            return directory

    raise ValueError(f"{features_directory} is not under processed/features")


def find_instance_files(
    root: str | pathlib.Path, names: list[str]
) -> dict[str, pathlib.Path]:
    """Instance file of each name (without .mis), searched for under root."""
    found: dict[str, list[pathlib.Path]] = {}

    for path in pathlib.Path(root).rglob("*.mis"):
        found.setdefault(path.stem, []).append(path)

    paths = {}

    for name in names:
        candidates = found.get(name, [])

        if not candidates:
            raise FileNotFoundError(f"No instance file {name}.mis under {root}")

        contents = {path.read_bytes() for path in candidates}

        if len(contents) > 1:
            raise ValueError(f"Several different instance files named {name}.mis")

        paths[name] = candidates[0]

    return paths


def read_instance_graph(path: str | pathlib.Path) -> tuple[int, np.ndarray]:
    """Number of vertices and edges [m, 2] of a .mis instance file."""
    tokens = pathlib.Path(path).read_text().split()

    if tokens[:2] != ["MIS_INSTANCE", "1"] or tokens[2] != "n" or tokens[4] != "m":
        raise ValueError(f"Not a MIS instance file: {path}")

    number_of_vertices = int(tokens[3])
    number_of_edges = int(tokens[5])
    values = tokens[6 : 6 + 2 * number_of_edges]

    if len(values) != 2 * number_of_edges:
        raise ValueError(f"Truncated MIS instance file: {path}")

    edges = np.array(values, np.int64).reshape(number_of_edges, 2)

    if number_of_edges and (edges.min() < 0 or edges.max() >= number_of_vertices):
        raise ValueError(f"Edge end out of range in {path}")

    return number_of_vertices, edges


def segment_argmax(values: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """Per sample, position of the first candidate with the largest value.

    values has one entry per candidate, laid out as described by offsets;
    every sample must have at least one candidate.
    """
    starts = offsets[:-1]
    lengths = np.diff(offsets)
    largest = np.maximum.reduceat(values, starts)
    is_largest = np.flatnonzero(values == np.repeat(largest, lengths))
    sample = np.repeat(np.arange(len(starts)), lengths)[is_largest]

    # First largest candidate of each sample.
    _, first = np.unique(sample, return_index=True)

    return is_largest[first] - starts


def _usable_samples(table: pd.DataFrame) -> pd.Series:
    """Per row: whether its sample is a meaningful expert decision.

    A sample is kept when the node was branched (not pruned), the expert
    made a choice and strong branching evaluated every candidate.
    """
    by_sample = table.groupby("sample_index", sort=False)

    return (
        (table["node_cutoff"] == 0)
        & (table["expert_selection_valid"] == 1)
        & (by_sample["evaluated"].transform("min") == 1)
    )


def load_feature_tables(
    directories: str | pathlib.Path | list[str] | list[pathlib.Path],
) -> BranchingDataset:
    """Loads every feature table of one or several directories.

    Several directories are aggregated into one dataset, for example the
    samples collected along the expert's search and those collected along
    the search of a learned policy on the same instances, or the samples
    of several graph families.
    """
    if isinstance(directories, (str, pathlib.Path)):
        directories = [directories]

    paths = []
    source_of_path = []

    for source_index, directory in enumerate(directories):
        found = sorted(pathlib.Path(directory).glob("*" + FEATURES_SUFFIX))

        if not found:
            raise FileNotFoundError(f"No feature table found in {directory}")

        paths.extend(found)
        source_of_path.extend([source_index] * len(found))

    # First pass, on the label columns only: which samples are usable and
    # how large the arrays must be.
    feature_names: list[str] | None = None
    kept = []
    total_samples = 0
    samples = 0
    candidates = 0

    for path in paths:
        columns = list(pd.read_csv(path, nrows=0).columns)

        if columns[: len(LABEL_COLUMNS)] != LABEL_COLUMNS:
            raise ValueError(f"Unexpected label columns in {path}")

        names = columns[len(LABEL_COLUMNS):]

        if feature_names is None:
            feature_names = names
        elif names != feature_names:
            raise ValueError(f"Feature columns of {path} differ from the others")

        labels = pd.read_csv(path, usecols=LABEL_COLUMNS)
        usable = _usable_samples(labels).to_numpy()
        total_samples += labels["sample_index"].nunique()
        samples += labels["sample_index"][usable].nunique()
        candidates += int(usable.sum())
        kept.append(usable)

    assert feature_names is not None

    features = np.zeros((candidates, len(feature_names)), np.float32)
    scores = np.zeros(candidates, np.float32)
    vertices = np.zeros(candidates, np.int32)
    lp_values = np.zeros(candidates, np.float32)
    offsets = np.zeros(samples + 1, np.int64)
    target = np.full(samples, -1, np.int64)
    instance = np.zeros(samples, np.int64)
    sample_ids = np.zeros(samples, np.int64)
    instance_names = []

    # Second pass: one table at a time, appended to the arrays, so that
    # the tables never sit in memory together.
    first_sample = 0
    first_row = 0

    for instance_index, (path, usable) in enumerate(zip(paths, kept)):
        # Features are used in single precision.
        table = pd.read_csv(path, dtype={name: np.float32 for name in feature_names})
        table = table[usable]

        sample_index = table["sample_index"].to_numpy(np.int64)
        position = table["candidate_position"].to_numpy(np.int64)

        # Rows are grouped by sample, in candidate order 0, 1, 2, ...
        new_sample = np.ones(len(table), bool)
        new_sample[1:] = sample_index[1:] != sample_index[:-1]
        sample_of_row = np.cumsum(new_sample) - 1
        count = int(new_sample.sum())

        if len(np.unique(sample_index)) != count:
            raise ValueError(f"Rows of a sample are not consecutive in {path}")

        lengths = np.bincount(sample_of_row, minlength=count)
        expected = np.arange(len(table)) - np.repeat(
            np.cumsum(lengths) - lengths, lengths
        )

        if not np.array_equal(position, expected):
            raise ValueError(f"Candidates out of order in {path}")

        rows = slice(first_row, first_row + len(table))
        features[rows] = table[feature_names].to_numpy(np.float32)
        scores[rows] = table["score"].to_numpy(np.float32)
        vertices[rows] = table["vertex_id"].to_numpy(np.int32)
        lp_values[rows] = table["lp_value"].to_numpy(np.float32)
        offsets[first_sample + 1 : first_sample + count + 1] = (
            first_row + np.cumsum(lengths)
        )

        selected = table["expert_selected"].to_numpy() == 1

        if np.bincount(sample_of_row[selected], minlength=count).tolist() != [1] * count:
            raise ValueError("Every sample must have exactly one selected candidate")

        target[first_sample + sample_of_row[selected]] = position[selected]
        instance[first_sample : first_sample + count] = instance_index
        sample_ids[first_sample : first_sample + count] = sample_index[new_sample]
        instance_names.append(path.name[: -len(FEATURES_SUFFIX)])
        first_sample += count
        first_row += len(table)

    if first_sample != samples or first_row != candidates:
        raise ValueError("Inconsistent number of usable samples")

    source = np.asarray(source_of_path, np.int64)[instance]

    return BranchingDataset(
        feature_names=feature_names,
        features=features,
        scores=scores,
        vertices=vertices,
        offsets=offsets,
        target=target,
        instance=instance,
        source=source,
        instance_names=instance_names,
        total_samples=total_samples,
        lp_values=lp_values,
        sample_ids=sample_ids,
        source_directories=[str(directory) for directory in directories],
    )
