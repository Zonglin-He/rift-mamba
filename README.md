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
- `basis.py` and `coefficients.py`: basis generation and extraction for `b=(route,column,aggregation,window)`, with alpha values and missing masks.
- `semantic.py`: injectable schema/value semantic encoder. The default is deterministic hash text encoding; a frozen LLM/text encoder can implement the same interface.
- `layout.py`: stable `route x slot x channel` dense tensor layout for CNN/TCN-style feature extractors.
- `sequences.py`: path-aware route event tokens. Tokens include feature values from every row on the route path, categorical/text embedding vectors, Fourier time features, hop count, and route semantics.
- `nn.py`: learnable synthesis `m alpha Psi + (1-m) Omega`, `basis_mode="sum"`, `basis_mode="mamba"`, `basis_mode="cnn"`, route-wise sequence encoder, and fusion head.
- `task.py`: `TaskSpec`, automatic target/proxy leakage exclusion helpers, and train-split proxy leakage audit.
- `pretraining.py`: TVE-style future task vector targets and cosine loss.
- `adapters.py`, `experiment.py`, `baselines.py`: RelBench/materialized dataset adapter, train-only experiment preparation, built-in DFS MLP baseline, and external baseline wrappers for DFS LightGBM, GraphSAGE RDL, RelGNN, RelGT, RT, and Griffin.

## Run

```bash
python -m pytest
python examples/ecommerce_churn_demo.py
```

Expected tests:

```text
10 passed
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
5. Run `audit_proxy_leakage()` on the train split for autocomplete-style targets.

## RelBench and Baselines

`RelBenchAdapter.from_materialized()` accepts already loaded tables, schema and task rows. This keeps the core code independent of RelBench release-specific object formats. For official RelBench/RelBench v2 objects, subclass the adapter or map them into `RelationalDatasetBundle`.

Faithful RelGNN, RelGT, RT and Griffin comparisons should call their official repositories through `ExternalBaseline`; the registry names are provided by `default_baseline_registry()`.
