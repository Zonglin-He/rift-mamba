# RIFT-Mamba

Reference prototype for **Relational Inverse Feature Transform with Route-wise Mamba**.

The code implements the idea as a leakage-safe relational prediction pipeline:

```text
RDB records
-> schema routes
-> causal reachable sets N_r(q)
-> relational coefficients alpha_b(q)
-> learnable basis synthesis
-> basis Mamba + route-wise event Mamba
-> prediction head
```

Core pieces:

- `rift_mamba.schema`: table, column, PK and FK metadata. PK/FK columns are connectivity metadata and are excluded from features.
- `rift_mamba.routes`: bounded schema route enumeration with forward and backward PK-FK traversal.
- `rift_mamba.records`: per-sample causal route traversal. Timestamped rows must satisfy `tau <= seed_time`.
- `rift_mamba.basis`: basis generation with `b=(route,column,aggregation,window)`.
- `rift_mamba.coefficients`: extraction of `alpha_b(q)` and missing-value masks.
- `rift_mamba.nn`: learnable synthesis `m alpha Psi + (1-m) Omega`, literal sum mode, tokenized basis Mamba mode, route-wise temporal sequence encoder, and final fusion model.

Run tests:

```bash
python -m pytest
```

Run the ecommerce demo:

```bash
python examples/ecommerce_churn_demo.py
```

Important usage rule: every fact table with time semantics must declare its `timestamp` column in `TableSchema`. Otherwise no implementation can guarantee temporal leakage prevention for that table.

The basis branch supports two forms:

- `basis_mode="sum"` computes the literal inverse-style signal `sum_b e_b(q)`.
- `basis_mode="mamba"` keeps all basis terms as ordered tokens, including missing-basis tokens, and lets the basis Mamba learn higher-order interactions.

The temporal sequence builder encodes endpoint rows reached by each route, sorted by the route event time. Intermediate timestamped rows are still used for causal filtering and event-time propagation.
