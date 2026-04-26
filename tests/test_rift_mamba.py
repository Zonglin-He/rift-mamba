from __future__ import annotations

from datetime import timedelta

import numpy as np
import pytest
import torch

from rift_mamba import (
    BasisConfig,
    AtomicRouteEnumerator,
    BasisLayout,
    CoefficientExtractor,
    CoefficientStandardizer,
    ColumnSchema,
    DatabaseSchema,
    ForeignKey,
    RecordStore,
    RelBenchAdapter,
    RelationalDatasetBundle,
    RelationalBasis,
    RiftDataset,
    RiftMambaModel,
    RouteEnumerator,
    SchemaSemanticEncoder,
    TableSchema,
    TaskRow,
    TaskSpec,
    TaskVectorTargetBuilder,
    TemporalSequenceBuilder,
    audit_proxy_leakage,
    build_basis,
    build_exclude_columns,
    prepare_experiment,
    ExperimentConfig,
)
from rift_mamba.time import fourier_time_features


def make_schema() -> DatabaseSchema:
    customers = TableSchema(
        name="customers",
        primary_key="customer_id",
        columns=(
            ColumnSchema("customer_id", "primary_key", nullable=False),
            ColumnSchema("age", "numeric"),
            ColumnSchema("region", "categorical"),
        ),
    )
    transactions = TableSchema(
        name="transactions",
        primary_key="transaction_id",
        timestamp="timestamp",
        columns=(
            ColumnSchema("transaction_id", "primary_key", nullable=False),
            ColumnSchema("customer_id", "foreign_key"),
            ColumnSchema("product_id", "foreign_key"),
            ColumnSchema("price", "numeric"),
            ColumnSchema("timestamp", "datetime"),
        ),
    )
    products = TableSchema(
        name="products",
        primary_key="product_id",
        columns=(
            ColumnSchema("product_id", "primary_key", nullable=False),
            ColumnSchema("category", "categorical"),
        ),
    )
    return DatabaseSchema.from_tables(
        [customers, transactions, products],
        [
            ForeignKey("transactions", "customer_id", "customers", "customer_id", role="buyer"),
            ForeignKey("transactions", "product_id", "products", "product_id", role="item"),
        ],
    )


def make_store(schema: DatabaseSchema) -> RecordStore:
    return RecordStore(
        schema,
        {
            "customers": [{"customer_id": "c1", "age": 30, "region": "east"}],
            "transactions": [
                {"transaction_id": "t_old", "customer_id": "c1", "product_id": "p2", "price": 30.0, "timestamp": "2022-11-01"},
                {"transaction_id": "t1", "customer_id": "c1", "product_id": "p1", "price": 10.0, "timestamp": "2023-01-01"},
                {"transaction_id": "t_missing", "customer_id": "c1", "product_id": "p1", "price": None, "timestamp": "2023-01-10"},
                {"transaction_id": "t2", "customer_id": "c1", "product_id": "p2", "price": 20.0, "timestamp": "2023-02-01"},
            ],
            "products": [
                {"product_id": "p1", "category": "electronics"},
                {"product_id": "p2", "category": "home"},
            ],
        },
    )


def test_route_basis_excludes_pk_fk_and_keeps_routes() -> None:
    schema = make_schema()
    routes = RouteEnumerator(schema, max_hops=2).enumerate("customers")
    route_names = {route.name for route in routes}

    assert "customers" in route_names
    assert any(route.table_path == ("customers", "transactions") for route in routes)
    assert any(route.table_path == ("customers", "transactions", "products") for route in routes)

    basis = build_basis(schema, routes, BasisConfig(windows=(None, timedelta(days=30))))
    assert all(basis_item.column_kind not in {"primary_key", "foreign_key"} for basis_item in basis)
    assert any(basis_item.route.table_path == ("customers", "transactions") for basis_item in basis)


def test_schema_rejects_misdeclared_primary_key() -> None:
    with pytest.raises(ValueError, match="primary key"):
        TableSchema(
            name="bad",
            primary_key="id",
            columns=(ColumnSchema("id", "numeric"),),
        )


def test_causal_coefficients_exclude_future_and_apply_window_after_full_route() -> None:
    schema = make_schema()
    store = make_store(schema)
    routes = RouteEnumerator(schema, max_hops=2).enumerate("customers")
    txn_route = next(route for route in routes if route.table_path == ("customers", "transactions"))
    product_route = next(route for route in routes if route.table_path == ("customers", "transactions", "products"))
    task = TaskRow(row_id=0, entity_id="c1", seed_time="2023-01-15")
    bases = (
        RelationalBasis(0, txn_route, "count", None),
        RelationalBasis(1, txn_route, "count", timedelta(days=30)),
        RelationalBasis(2, txn_route, "mean", timedelta(days=30), "price", "numeric"),
        RelationalBasis(3, product_route, "count", timedelta(days=30)),
    )

    matrix = CoefficientExtractor(store, bases).transform([task], target_table="customers")

    assert matrix.values[0, 0] == 3.0
    assert matrix.values[0, 1] == 2.0
    assert matrix.values[0, 2] == 10.0
    assert matrix.masks[0, 2]
    assert matrix.values[0, 3] == 2.0


def test_last_aggregation_ignores_missing_without_misaligned_times() -> None:
    schema = make_schema()
    store = make_store(schema)
    routes = RouteEnumerator(schema, max_hops=1).enumerate("customers")
    txn_route = next(route for route in routes if route.table_path == ("customers", "transactions"))
    task = TaskRow(row_id=0, entity_id="c1", seed_time="2023-01-15")
    bases = (RelationalBasis(0, txn_route, "last", None, "price", "numeric"),)

    matrix = CoefficientExtractor(store, bases).transform([task], target_table="customers")

    assert matrix.values[0, 0] == 10.0


def test_sequence_builder_and_model_forward() -> None:
    schema = make_schema()
    store = make_store(schema)
    routes = RouteEnumerator(schema, max_hops=1).enumerate("customers")
    basis = build_basis(schema, routes, BasisConfig(windows=(None, timedelta(days=30))))
    task = TaskRow(row_id=0, entity_id="c1", seed_time="2023-01-15", label=1)
    coeffs = CoefficientExtractor(store, basis).transform([task], target_table="customers")
    temporal_routes = tuple(route for route in routes if route.hop_count > 0)
    seq = TemporalSequenceBuilder(schema, store, temporal_routes, max_len=4).transform([task], target_table="customers")

    model = RiftMambaModel(
        bases=basis,
        d_model=16,
        output_dim=2,
        event_dim=seq.values.shape[-1],
        num_temporal_routes=len(temporal_routes),
        basis_layers=1,
        sequence_layers=1,
    )
    logits = model(
        torch.from_numpy(coeffs.values).float(),
        torch.from_numpy(coeffs.masks).bool(),
        torch.from_numpy(seq.values).float(),
        torch.from_numpy(seq.masks).bool(),
    )

    assert logits.shape == (1, 2)
    assert torch.isfinite(logits).all()


def test_atomic_routes_generate_fact_bridge_routes() -> None:
    schema = make_schema()
    routes = AtomicRouteEnumerator(schema, max_hops=2).enumerate("customers")

    assert any(route.table_path == ("customers", "transactions", "products") for route in routes)


def test_exclude_columns_leakage_audit_and_train_only_standardizer() -> None:
    schema = make_schema()
    task = TaskSpec(
        target_table="customers",
        entity_column="customer_id",
        seed_time_column="seed_time",
        label_column="region",
        task_type="autocomplete",
        target_column=("customers", "region"),
        leakage_columns=(("customers", "age"),),
    )
    excluded = set(build_exclude_columns(task, schema))

    assert ("customers", "region") in excluded
    assert ("customers", "age") in excluded

    findings = audit_proxy_leakage(
        [
            {"target": "a", "proxy": "a", "weak": "x"},
            {"target": "b", "proxy": "b", "weak": "x"},
            {"target": "c", "proxy": "c", "weak": "y"},
        ],
        target_column="target",
    )
    assert any(finding.column == "proxy" and finding.score == 1.0 for finding in findings)

    schema = make_schema()
    store = make_store(schema)
    routes = RouteEnumerator(schema, max_hops=1).enumerate("customers")
    basis = build_basis(schema, routes, BasisConfig(windows=(None,)))
    train = CoefficientExtractor(store, basis).transform([TaskRow(0, "c1", "2023-01-15")], "customers")
    scaler = CoefficientStandardizer().fit(train)
    transformed = scaler.transform(train)

    assert np.all(transformed.values[~transformed.masks] == 0.0)


def test_fourier_time_and_path_aware_event_tokens_include_intermediate_fact_values() -> None:
    schema = make_schema()
    store = make_store(schema)
    routes = RouteEnumerator(schema, max_hops=2).enumerate("customers")
    product_route = next(route for route in routes if route.table_path == ("customers", "transactions", "products"))
    task = TaskRow(row_id=0, entity_id="c1", seed_time="2023-01-15")
    builder = TemporalSequenceBuilder(schema, store, (product_route,), max_len=4, value_embedding_dim=4)
    seq = builder.transform([task], target_table="customers")

    assert seq.values.shape[-1] == builder.event_dim
    assert seq.masks[0, 0].sum() == 3
    txn_price_idx = next(i for i, col in enumerate(seq.event_columns) if col.name == "transactions.price")
    per_column = 2 + builder.value_embedding_dim
    price_values = seq.values[0, 0, :, txn_price_idx * per_column]
    assert 10.0 in price_values
    assert np.isfinite(fourier_time_features(14.0)).all()


def test_cnn_layout_and_semantic_basis_encoder_forward() -> None:
    schema = make_schema()
    store = make_store(schema)
    routes = RouteEnumerator(schema, max_hops=1).enumerate("customers")
    basis = build_basis(schema, routes, BasisConfig(windows=(None,)))
    coeffs = CoefficientExtractor(store, basis).transform([TaskRow(0, "c1", "2023-01-15")], "customers")
    layout = BasisLayout.from_bases(basis)
    model = RiftMambaModel(
        bases=basis,
        d_model=16,
        output_dim=2,
        basis_layers=1,
        basis_mode="cnn",
        semantic_encoder=SchemaSemanticEncoder(),
        use_mamba_ssm=False,
    )
    tokens = model.basis_synthesizer(
        torch.from_numpy(coeffs.values).float(),
        torch.from_numpy(coeffs.masks).bool(),
    )
    dense = layout.tokens_to_tensor(tokens)
    logits = model(torch.from_numpy(coeffs.values).float(), torch.from_numpy(coeffs.masks).bool())

    assert dense.shape[:3] == (1, layout.num_routes, layout.num_slots)
    assert logits.shape == (1, 2)


def test_task_vector_targets_and_dataset_alignment() -> None:
    schema = make_schema()
    store = make_store(schema)
    routes = RouteEnumerator(schema, max_hops=1).enumerate("customers")
    txn_route = next(route for route in routes if route.table_path == ("customers", "transactions"))
    bases = (RelationalBasis(0, txn_route, "count", None),)
    task = TaskRow(0, "c1", "2023-01-15", label=1)

    targets = TaskVectorTargetBuilder(store, bases, timedelta(days=45)).transform([task], "customers")
    assert targets.values[0, 0] == 1.0

    coeffs = CoefficientExtractor(store, bases).transform([task], "customers")
    seq = TemporalSequenceBuilder(schema, store, (txn_route,), max_len=4).transform([task], "customers")
    dataset = RiftDataset(coeffs, sequences=seq)
    assert len(dataset) == 1

    wrong_seq = TemporalSequenceBuilder(schema, store, (txn_route,), max_len=4).transform(
        [TaskRow(1, "c1", "2023-01-15")],
        "customers",
    )
    with pytest.raises(ValueError, match="align"):
        RiftDataset(coeffs, sequences=wrong_seq)


def test_path_aware_event_columns_preserve_repeated_table_occurrences() -> None:
    table = TableSchema(
        name="employees",
        primary_key="employee_id",
        columns=(
            ColumnSchema("employee_id", "primary_key", nullable=False),
            ColumnSchema("manager_id", "foreign_key"),
            ColumnSchema("level", "numeric"),
        ),
    )
    schema = DatabaseSchema.from_tables(
        [table],
        [ForeignKey("employees", "manager_id", "employees", "employee_id", role="manager")],
    )
    store = RecordStore(
        schema,
        {
            "employees": [
                {"employee_id": "e1", "manager_id": "e2", "level": 1},
                {"employee_id": "e2", "manager_id": None, "level": 2},
            ]
        },
    )
    routes = RouteEnumerator(schema, max_hops=1, allow_cycles=True).enumerate("employees")
    route = next(route for route in routes if route.hop_count == 1)
    seq = TemporalSequenceBuilder(schema, store, (route,), max_len=2).transform([TaskRow(0, "e1", "2023-01-01")], "employees")

    assert any(column.name == "employees.level" for column in seq.event_columns)
    assert any(column.name == "employees#1.level" for column in seq.event_columns)


def test_prepare_experiment_runs_leakage_audit_and_relbench_materialized_adapter() -> None:
    schema = make_schema()
    tables = {
        "customers": [
            {"customer_id": "c1", "age": 30, "region": "yes", "proxy": "yes"},
            {"customer_id": "c2", "age": 40, "region": "no", "proxy": "no"},
        ],
        "transactions": [
            {"transaction_id": "t1", "customer_id": "c1", "product_id": "p1", "price": 10, "timestamp": "2023-01-01"},
            {"transaction_id": "t2", "customer_id": "c2", "product_id": "p1", "price": 20, "timestamp": "2023-01-01"},
        ],
        "products": [{"product_id": "p1", "category": "x"}],
    }
    task = TaskSpec(
        target_table="customers",
        entity_column="customer_id",
        seed_time_column="seed_time",
        label_column="region",
        task_type="binary_classification",
    )
    rows = [TaskRow(0, "c1", "2023-01-15", 1), TaskRow(1, "c2", "2023-01-15", 0)]
    adapter = RelBenchAdapter.from_materialized(schema, tables, rows, task)
    bundle = adapter.load()
    dict_adapter = RelBenchAdapter.from_relbench(
        {"schema": schema, "tables": tables, "task_rows": rows, "task": task}
    )
    assert dict_adapter.load().task == task
    train, _ = prepare_experiment(
        bundle,
        train_rows=tuple(rows),
        config=ExperimentConfig(max_hops=1, sequence_max_len=4),
    )

    assert isinstance(bundle, RelationalDatasetBundle)
    assert any(finding.column == "proxy" for finding in train.leakage_findings)
    assert all("proxy" not in name for name in train.coefficients.basis_names)
