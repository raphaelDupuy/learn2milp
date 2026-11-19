# Interactive menu

`learn2milp` is an interactive terminal menu to generate instances, try branching rules and models, and look at datasets. Run it from the repository root, since it reads and writes under `data/`:

```bash
./build/learn2milp
```

The menu is meant to try a rule or a model, or to look at a dataset; `mis-benchmark` remains the tool for measurements (see [command-line-tools.md](command-line-tools.md)).

## Entries

1. **Generate a single MIS instance**, of either graph family, into `data/instances/single`. The file is named as by `mis-generator` (`erdos_renyi_n125_p0_1_seed1.mis`).
2. **Generate a batch of MIS instances**: every combination of lists of parameters and seeds, into `data/instances/batch/NAME`, with its `manifest.csv`.
3. **Solve an existing MIS instance** with a chosen branching rule: SCIP's default, a learned model (`mislearned`, with strong branching on the model's first K candidates if K > 0, as `mis-benchmark --strong-branching`), the full strong branching expert (`misfullstrong`), the expert without its pruning (`misfullstrong_nocutoff`) or any SCIP rule, under the study or default settings (see [command-line-tools.md](command-line-tools.md#solver-settings)), optionally a SCIP settings file applied after them (as `mis-benchmark --parameters`, for instance one of `experiments/cuts/settings/`), a time limit and a SCIP seed. A settings file is checked as soon as it is given: an unknown parameter, or a value SCIP does not apply, is reported with its line. The results give the status, the size of the independent set (with the upper bound and the gap when the time limit is reached), the first selected vertices, the explored nodes, the LP iterations and the solving time.
4. **Generate and solve an MIS instance**: entries 1 and 3 in a row, the instance being written to `data/instances/generated_and_solved`.
5. **Compare branching rules and settings on an instance**: SCIP rules, `misfullstrong`, `misfullstrong_nocutoff` and model files (`relpscost, experiments/er125_p01/models/mlp32_h2.model`), each with every settings file of a list where `-` stands for none (`-, experiments/cuts/settings/no_node_gomory.set` compares the settings alone with the settings and that file), over one or several SCIP seeds. With a model, K candidates for strong branching are asked once for all the models. Each configuration is labelled after its rule (a model after its file name, its path when two models share a name), `+sbK` and `+` the name of its settings file (`mlp32_h2+sb1+no_node_gomory`). The runs fill a table as they finish; with several seeds, the shifted geometric means of `mis-benchmark` follow. A warning is printed if two runs prove different optimal sizes, which would mean that a rule is not exact.
6. **Inspect a data directory**: the instances it holds (families, numbers of vertices and edges) and its strong branching collections (samples, samples usable for training as counted by the featurizer, candidates, explored nodes, agreement of the policy model with the expert, solving time). Files that cannot be read, such as those of an older format, are listed with the reason and left out.
7. **Exit**.

An invalid answer is asked again; `Ctrl+D` leaves the menu. In lists, spaces after the commas are allowed.

## File browser

Wherever a file or a directory is asked for, `?` opens a browser, and so does an empty answer unless the file is optional (a settings file, where it means none):

- numbered subdirectories, then the files of the expected kind (`.mis`, `.model`, `.set`);
- a number opens a directory or picks a file, `..` goes up, `.` picks the current directory (when a directory is asked for), `n`/`p` change page, `q` cancels;
- a path, relative or absolute, can be typed instead.

`q` at the question itself goes back to the menu. A directory typed where a file is expected opens the browser there. In the lists of rules and of settings files to compare, each `?` picks a model or settings file the same way (`relpscost, ?, ?`).

The browser starts in `data/instances` for instances, `experiments` for models, `experiments/cuts/settings` for settings files and `data` for directories, then where it was last used.
