"""Trains a model that ranks the branching candidates of a node.

The model imitates the full strong branching expert: for each recorded
branching decision, it should give its highest score to the candidate
the expert chose. It is a small network applied to each candidate
independently, so that it can be evaluated in C inside SCIP in well under
a millisecond per node.

Usage:
    python ml/train_branching_model.py \
        --train data/processed/features/er125_p01/train \
        --valid data/processed/features/er125_p01/valid \
        --hidden 32 32 --output experiments/er125_p01/models/mlp32.model

Use --hidden with no value for a linear model.

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

from branching_data import BranchingDataset, load_feature_tables, segment_argmax
from branching_model_file import BranchingModel, read_model, write_model


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
            layers.append(
                (
                    layer.weight.detach().numpy().astype(np.float64),
                    layer.bias.detach().numpy().astype(np.float64),
                    activation,
                )
            )

        return BranchingModel(
            feature_names,
            self.mean.numpy().astype(np.float64),
            self.std.numpy().astype(np.float64),
            layers,
        )


def predict(model: CandidateScorer, dataset: BranchingDataset) -> np.ndarray:
    """Score of every candidate (the model scores candidates one by one)."""
    model.eval()
    outputs = []

    with torch.no_grad():
        for start in range(0, dataset.number_of_candidates, 65536):
            block = torch.from_numpy(dataset.features[start : start + 65536])
            outputs.append(model(block).numpy())

    return np.concatenate(outputs) if outputs else np.zeros(0, np.float32)


def file_scores(model: BranchingModel, dataset: BranchingDataset) -> np.ndarray:
    """Score of every candidate under a model read from its file.

    Computed by blocks: the double precision activations of a whole
    validation set would take several gigabytes.
    """
    return np.concatenate(
        [
            model.scores(dataset.features[start : start + 65536])
            for start in range(0, dataset.number_of_candidates, 65536)
        ]
    )


def train(
    train_set: BranchingDataset,
    valid_set: BranchingDataset,
    hidden: list[int],
    epochs: int,
    learning_rate: float,
    batch_size: int,
    seed: int,
) -> tuple[CandidateScorer, list[dict[str, float]]]:
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

    model = CandidateScorer(mean, std, hidden)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    target = torch.from_numpy(train_set.target)

    history = []
    best_ratio = -1.0
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(train_set.number_of_samples)
        total_loss = 0.0

        for start in range(0, len(order), batch_size):
            batch = order[start : start + batch_size]

            # Padded to the largest number of candidates of the dataset:
            # every batch has the same shape.
            features, mask = train_set.padded_batch(batch.numpy())
            scores = model(torch.from_numpy(features)).masked_fill(
                ~torch.from_numpy(mask), -1e9
            )

            # The expert's choice should get the highest score of its node.
            loss = torch.nn.functional.cross_entropy(scores, target[batch])

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
            f"score ratio {metrics['score_ratio']:.4f}",
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
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    arguments = parser.parse_args()

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

    for name, dataset in [("train", train_set), ("valid", valid_set)]:
        print(
            f"{name}: {len(dataset.instance_names)} feature tables, "
            f"{dataset.number_of_samples} usable samples of {dataset.total_samples}, "
            f"{dataset.number_of_candidates} candidates"
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
        arguments.epochs,
        arguments.learning_rate,
        arguments.batch_size,
        arguments.seed,
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
