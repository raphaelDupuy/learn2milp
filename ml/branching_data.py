"""Loading of the branching feature tables written by mis-featurizer.

A feature table has one row per branching candidate. This module groups
the rows by sample (one branching decision) and keeps the samples in
which the expert's choice is meaningful.

The candidates of all samples are stored one after the other, without
padding: graph families with many candidates per node (Barabasi-Albert
graphs have twice as many as Erdos-Renyi ones of the same density) would
otherwise make every sample as large as the largest one. Padded arrays
are built only for small groups of samples (padded_batch()).
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
    offsets   [samples + 1], the candidates of sample s are the rows
              offsets[s] to offsets[s + 1] - 1, in candidate order
    target    [samples], position, within its sample, of the candidate
              chosen by the expert
    instance  [samples], index of the instance each sample comes from
    source    [samples], index of the directory each sample comes from;
              the samples of one directory are consecutive
    """

    feature_names: list[str]
    features: np.ndarray
    scores: np.ndarray
    offsets: np.ndarray
    target: np.ndarray
    instance: np.ndarray
    source: np.ndarray
    instance_names: list[str]
    total_samples: int

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

        return dataclasses.replace(
            self,
            features=self.features[rows],
            scores=self.scores[rows],
            offsets=self.offsets[first : last + 1] - self.offsets[first],
            target=self.target[first:last],
            instance=self.instance[first:last],
            source=self.source[first:last],
        )


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
    offsets = np.zeros(samples + 1, np.int64)
    target = np.full(samples, -1, np.int64)
    instance = np.zeros(samples, np.int64)
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
        offsets[first_sample + 1 : first_sample + count + 1] = (
            first_row + np.cumsum(lengths)
        )

        selected = table["expert_selected"].to_numpy() == 1

        if np.bincount(sample_of_row[selected], minlength=count).tolist() != [1] * count:
            raise ValueError("Every sample must have exactly one selected candidate")

        target[first_sample + sample_of_row[selected]] = position[selected]
        instance[first_sample : first_sample + count] = instance_index
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
        offsets=offsets,
        target=target,
        instance=instance,
        source=source,
        instance_names=instance_names,
        total_samples=total_samples,
    )
