"""Loading of the branching feature tables written by mis-featurizer.

A feature table has one row per branching candidate. This module groups
the rows by sample (one branching decision) and keeps the samples in
which the expert's choice is meaningful.
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
    """Samples padded to the same number of candidates.

    features  [samples, candidates, features], float32, 0 where padded
    mask      [samples, candidates], True for real candidates
    scores    [samples, candidates], strong branching score, 0 where padded
    target    [samples], position of the candidate chosen by the expert
    instance  [samples], index of the instance each sample comes from
    source    [samples], index of the directory each sample comes from
    """

    feature_names: list[str]
    features: np.ndarray
    mask: np.ndarray
    scores: np.ndarray
    target: np.ndarray
    instance: np.ndarray
    source: np.ndarray
    instance_names: list[str]
    total_samples: int

    @property
    def number_of_samples(self) -> int:
        return self.features.shape[0]

    @property
    def number_of_candidates(self) -> int:
        return int(self.mask.sum())

    def with_features(self, names: list[str]) -> BranchingDataset:
        """The same samples, described by the given features only."""
        columns = [self.feature_names.index(name) for name in names]

        return dataclasses.replace(
            self,
            feature_names=list(names),
            features=np.ascontiguousarray(self.features[:, :, columns]),
        )

    def from_source(self, source: int) -> BranchingDataset:
        """The samples that come from one of the loaded directories."""
        keep = self.source == source

        return dataclasses.replace(
            self,
            features=self.features[keep],
            mask=self.mask[keep],
            scores=self.scores[keep],
            target=self.target[keep],
            instance=self.instance[keep],
            source=self.source[keep],
        )


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
    the search of a learned policy on the same instances.
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
    # how large the padded arrays must be.
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

        if usable.any():
            candidates = max(
                candidates, int(labels["candidate_position"][usable].max()) + 1
            )

        kept.append(usable)

    assert feature_names is not None

    features = np.zeros((samples, candidates, len(feature_names)), np.float32)
    mask = np.zeros((samples, candidates), bool)
    scores = np.zeros((samples, candidates), np.float32)
    target = np.full(samples, -1, np.int64)
    instance = np.zeros(samples, np.int64)
    instance_names = []

    # Second pass: one table at a time, written straight into the padded
    # arrays, so that the tables never sit in memory together.
    first = 0

    for instance_index, (path, usable) in enumerate(zip(paths, kept)):
        # Features are used in single precision.
        table = pd.read_csv(path, dtype={name: np.float32 for name in feature_names})
        table = table[usable]

        _, sample_of_row = np.unique(
            table["sample_index"].to_numpy(np.int64), return_inverse=True
        )
        count = int(sample_of_row.max()) + 1 if len(table) else 0
        sample_of_row = sample_of_row + first
        position = table["candidate_position"].to_numpy(np.int64)

        if mask[sample_of_row, position].any():
            raise ValueError(f"Duplicate candidate in {path}")

        features[sample_of_row, position] = table[feature_names].to_numpy(np.float32)
        mask[sample_of_row, position] = True
        scores[sample_of_row, position] = table["score"].to_numpy(np.float32)

        selected = table["expert_selected"].to_numpy() == 1

        if np.bincount(sample_of_row[selected] - first, minlength=count).tolist() != [1] * count:
            raise ValueError("Every sample must have exactly one selected candidate")

        target[sample_of_row[selected]] = position[selected]
        instance[first : first + count] = instance_index
        instance_names.append(path.name[: -len(FEATURES_SUFFIX)])
        first += count

    if first != samples:
        raise ValueError("Inconsistent number of usable samples")

    source = np.asarray(source_of_path, np.int64)[instance]

    return BranchingDataset(
        feature_names=feature_names,
        features=features,
        mask=mask,
        scores=scores,
        target=target,
        instance=instance,
        source=source,
        instance_names=instance_names,
        total_samples=total_samples,
    )
