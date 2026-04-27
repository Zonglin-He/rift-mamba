# RIFT-Mamba

Reference implementation for **RIFT-Mamba: Relational Inverse Feature Transform with Route-wise Mamba**.

The package turns relational databases into leakage-safe route signals:

```text
RDB records
-> schema / atomic routes
-> per-sample causal reachable paths
-> relational coefficients alpha_b(q)
-> learnable basis synthesis
-> route-slot dense signal or basis-token Mamba
-> route-wise path event Mamba
-> prediction head
```

## Implemented Components

- `schema.py`: table, column, PK and FK metadata. PK/FK columns are connectivity metadata and are excluded from features.
- `routes.py`: bounded route enumeration plus `AtomicRouteEnumerator` for fact/bridge tables such as `customer <- transaction -> product`.
- `records.py`: per-sample causal traversal. Every timestamped row on a path must satisfy `tau <= seed_time`; windows are applied after the full path event time is known.
- `basis.py` and `coefficients.py`: basis generation and extraction for `b=(route,column,aggregation,window)`, plus path-conditional composite bases such as transaction price conditioned on product category.
- `semantic.py`: injectable schema/value semantic encoder. The default is deterministic hash text encoding; `SentenceTransformerTextEncoder` is available through the optional `text` extra.
- `layout.py`: stable `route x slot x channel` dense tensor layout for CNN/TCN-style feature extractors.
- `sequences.py`: path-aware route event tokens. Tokens include feature values from every row on the route path, categorical/text embedding vectors, Fourier time features, hop count, and route semantics.
- `sequence_preprocessing.py`: train-only event feature standardization for raw numeric and recency dimensions in the route-wise sequence branch.
- `nn.py`: learnable synthesis `m alpha Psi + (1-m) Omega`, `basis_mode="sum"`, `basis_mode="mamba"`, `basis_mode="cnn"`, route-wise sequence encoder, and fusion head.
- `trainer.py`: minimal supervised training loop with optional TVE auxiliary cosine loss over `TaskVectorHead`.
- `task.py`: `TaskSpec`, automatic target/proxy leakage exclusion helpers, and train-split proxy leakage audit. The same exclude set is applied to coefficient bases and sequence event tokens.
- `pretraining.py`: TVE-style future task vector targets, null indicators, sample reweighting, and cosine loss that handles null-heavy targets.
- `recommendation.py`: link-prediction rows, negative sampling, pair datasets, and two-tower RIFT pair scoring.
- `adapters.py`, `relbench_v1_loader.py`, `relbench_v2_loader.py`, `experiment.py`, `baselines.py`, `backends.py`: RelBench/materialized dataset adapter, backend selection, temporal split helpers, metrics, train-only experiment preparation, built-in DFS MLP baseline, and external baseline wrappers for DFS LightGBM, GraphSAGE RDL, RelGNN, RelGT, RT, and Griffin.
- `experiments/`: executable scripts for RIFT-Mamba, DFS-MLP, DFS-LightGBM, external baseline dispatch, and result summarization.

## Run

```bash
python -m pytest
python examples/ecommerce_churn_demo.py
```

Expected tests:

```text
23 passed
```

## Basis Branch Modes

- `basis_mode="sum"` computes the literal inverse-style signal `sum_b e_b(q)`.
- `basis_mode="mamba"` keeps all basis terms as ordered tokens, including missing-basis tokens.
- `basis_mode="cnn"` scatters basis tokens into `[batch, routes, slots, channels]` and applies a CNN encoder.

## Leakage Rules

1. Declare `timestamp` for every fact table with time semantics.
2. Build inputs only from rows satisfying `tau <= seed_time`.
3. Fit `CoefficientStandardizer` only on the train split, then transform validation/test.
4. Fit `EventFeatureStandardizer` only on train route sequences, then transform validation/test sequences.
5. Use `TaskSpec` and `build_exclude_columns()` to remove target and known leakage columns.
6. `prepare_experiment()` runs target-table and route-wise proxy leakage audits on the train split by default and excludes detected proxy columns from both coefficient and sequence branches.

## RelBench and Baselines

`RelBenchAdapter.from_materialized()` accepts already loaded tables, schema and task rows. `load_relbench_v1()` and `load_relbench_v2()` call the official RelBench API (`get_dataset`, `get_task`) and materialize official train/val/test splits into `bundle.split_rows`. They also accept RelBench-like materialized objects exposing `schema`, `tables`, `task_rows`, and `task`.

Backends are selected through `RelationalDatasetBundle.backend` or `make_backend()`:

```text
materialized
duckdb
polars
duckdb-polars
```

DuckDB and Polars backends push route joins and scalar aggregations down to SQL/lazy DataFrame execution for supported bases (`count`, numeric aggregations, datetime recency). Unsupported hash/text/composite bases fall back column-wise to the in-memory extractor so results stay correct.

Faithful RelGNN, RelGT, RT and Griffin comparisons should call their official repositories through `ExternalBaseline`; the registry names are provided by `default_baseline_registry()`.

External GraphSAGE RDL, RelGNN, RelGT, RT and Griffin comparisons are dispatched through `experiments/run_external_baseline.py`; point `--command` at the official implementation entry point. This keeps comparison code tied to the official repositories instead of reimplementing those models locally.

Example experiment commands:

```bash
python experiments/run_rift.py --dataset rel-f1 --task driver-position --relbench-version v1
python experiments/run_rift_link.py --dataset rel-amazon --task user-item-purchase --relbench-version v2
python experiments/run_dfs_mlp.py --dataset rel-f1 --task driver-position --relbench-version v1
python experiments/run_dfs_lightgbm.py --dataset rel-f1 --task driver-position --relbench-version v1
python experiments/summarize_results.py
```

Optional extras:

```bash
pip install -e ".[text]"
pip install -e ".[backends]"
pip install -e ".[relbench]"
pip install -e ".[mamba]"
pip install -e ".[baselines]"
```
