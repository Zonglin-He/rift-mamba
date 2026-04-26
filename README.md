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
- `nn.py`: learnable synthesis `m alpha Psi + (1-m) Omega`, `basis_mode="sum"`, `basis_mode="mamba"`, `basis_mode="cnn"`, route-wise sequence encoder, and fusion head.
- `task.py`: `TaskSpec`, automatic target/proxy leakage exclusion helpers, and train-split proxy leakage audit. The same exclude set is applied to coefficient bases and sequence event tokens.
- `pretraining.py`: TVE-style future task vector targets, null indicators, sample reweighting, and cosine loss that handles null-heavy targets.
- `adapters.py`, `relbench_v1_loader.py`, `relbench_v2_loader.py`, `experiment.py`, `baselines.py`, `backends.py`: RelBench/materialized dataset adapter, backend selection, temporal split helpers, metrics, train-only experiment preparation, built-in DFS MLP baseline, and external baseline wrappers for DFS LightGBM, GraphSAGE RDL, RelGNN, RelGT, RT, and Griffin.

## Run

```bash
python -m pytest
python examples/ecommerce_churn_demo.py
```

Expected tests:

```text
17 passed
```

## Basis Branch Modes

- `basis_mode="sum"` computes the literal inverse-style signal `sum_b e_b(q)`.
- `basis_mode="mamba"` keeps all basis terms as ordered tokens, including missing-basis tokens.
- `basis_mode="cnn"` scatters basis tokens into `[batch, routes, slots, channels]` and applies a CNN encoder.

## Leakage Rules

1. Declare `timestamp` for every fact table with time semantics.
2. Build inputs only from rows satisfying `tau <= seed_time`.
3. Fit `CoefficientStandardizer` only on the train split, then transform validation/test.
4. Use `TaskSpec` and `build_exclude_columns()` to remove target and known leakage columns.
5. `prepare_experiment()` runs `audit_proxy_leakage()` on the train split by default and excludes detected proxy columns from both coefficient and sequence branches.

## RelBench and Baselines

`RelBenchAdapter.from_materialized()` accepts already loaded tables, schema and task rows. `load_relbench_v1()` and `load_relbench_v2()` are named loader boundaries for official adapters and accept RelBench-like materialized objects exposing `schema`, `tables`, `task_rows`, and `task`.

Backends are selected through `RelationalDatasetBundle.backend` or `make_backend()`:

```text
materialized
duckdb
polars
duckdb-polars
```

The current DuckDB/Polars backends are explicit optional-dependency boundaries for large-table preprocessing and future SQL pushdown; the core model remains backend-independent.

Faithful RelGNN, RelGT, RT and Griffin comparisons should call their official repositories through `ExternalBaseline`; the registry names are provided by `default_baseline_registry()`.

Optional extras:

```bash
pip install -e ".[text]"
pip install -e ".[backends]"
pip install -e ".[relbench]"
pip install -e ".[mamba]"
```
