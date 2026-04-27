# RIFT-Mamba Experiments

These scripts use official RelBench splits through `relbench.datasets.get_dataset`
and `relbench.tasks.get_task`.

```bash
python experiments/run_rift.py --dataset rel-f1 --task driver-position --relbench-version v1
python experiments/run_rift_link.py --dataset rel-amazon --task user-item-purchase --relbench-version v2
python experiments/run_dfs_mlp.py --dataset rel-f1 --task driver-position --relbench-version v1
python experiments/run_dfs_lightgbm.py --dataset rel-f1 --task driver-position --relbench-version v1
python experiments/run_external_baseline.py --name relgt --dataset rel-f1 --task driver-position --command python /path/to/relgt/train.py --output experiments/results/relgt.json
python experiments/summarize_results.py
```

`run_rift.py` and `run_rift_link.py` default to `--basis-mode route_set`.
Use `--basis-mode cnn` only as a layout-dependent ablation, or choose one of
`perceiver`, `relattn`, `bimamba`, `multiscan_mamba`, `route_mamba`,
`deepset`, `set_transformer`, `ft_transformer`, `mixer`, `basis_graph`, `tcn`,
`mamba`, or `sum` for extractor ablations.

External baselines are dispatched rather than reimplemented. Point `--command`
at the official implementation entry point for GraphSAGE RDL, RelGNN, RelGT,
RT, or Griffin; the script writes the shared dataset/task config and expects the
external command to write a JSON result file.
