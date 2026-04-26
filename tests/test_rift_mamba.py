from __future__ import annotations

from datetime import timedelta

import numpy as np
import pytest
import torch

from rift_mamba import (
    BasisConfig,
    CoefficientExtractor,
    ColumnSchema,
    DatabaseSchema,
    ForeignKey,
    RecordStore,
    RelationalBasis,
    RiftMambaModel,
    RouteEnumerator,
    TableSchema,
    TaskRow,
    TemporalSequenceBuilder,
    build_basis,
)


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
