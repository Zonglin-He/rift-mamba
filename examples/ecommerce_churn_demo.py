"""Minimal RIFT-Mamba demo on an in-memory ecommerce schema."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F

from rift_mamba import (
    BasisConfig,
    CoefficientExtractor,
    CoefficientStandardizer,
    ColumnSchema,
    DatabaseSchema,
    ForeignKey,
    RecordStore,
    RiftMambaModel,
    RouteEnumerator,
    TableSchema,
    TaskRow,
    TemporalSequenceBuilder,
    build_basis,
)


def build_schema() -> DatabaseSchema:
    customers = TableSchema(
        name="customers",
        primary_key="customer_id",
        timestamp=None,
        columns=(
            ColumnSchema("customer_id", "primary_key", nullable=False),
            ColumnSchema("age", "numeric"),
            ColumnSchema("region", "categorical"),
            ColumnSchema("signup_date", "datetime"),
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
        timestamp=None,
        columns=(
            ColumnSchema("product_id", "primary_key", nullable=False),
            ColumnSchema("category", "categorical"),
            ColumnSchema("brand", "categorical"),
        ),
    )
    return DatabaseSchema.from_tables(
        [customers, transactions, products],
        [
            ForeignKey("transactions", "customer_id", "customers", "customer_id", role="buyer"),
            ForeignKey("transactions", "product_id", "products", "product_id", role="purchased_product"),
        ],
    )


def main() -> None:
    schema = build_schema()
    tables = {
        "customers": [
            {"customer_id": "c1", "age": 31, "region": "east", "signup_date": "2022-01-01"},
            {"customer_id": "c2", "age": 45, "region": "west", "signup_date": "2021-06-01"},
        ],
        "transactions": [
            {"transaction_id": "t1", "customer_id": "c1", "product_id": "p1", "price": 10.0, "timestamp": "2023-01-01"},
            {"transaction_id": "t2", "customer_id": "c1", "product_id": "p2", "price": 30.0, "timestamp": "2023-01-11"},
            {"transaction_id": "t3", "customer_id": "c1", "product_id": "p1", "price": 99.0, "timestamp": "2023-03-01"},
            {"transaction_id": "t4", "customer_id": "c2", "product_id": "p2", "price": 15.0, "timestamp": "2023-01-03"},
        ],
        "products": [
            {"product_id": "p1", "category": "electronics", "brand": "acme"},
            {"product_id": "p2", "category": "home", "brand": "nova"},
        ],
    }
    tasks = [
        TaskRow(row_id=0, entity_id="c1", seed_time="2023-01-15", label=0),
        TaskRow(row_id=1, entity_id="c2", seed_time="2023-01-15", label=1),
    ]

    store = RecordStore(schema, tables)
    routes = RouteEnumerator(schema, max_hops=2).enumerate("customers")
    basis = build_basis(
        schema,
        routes,
        BasisConfig(windows=(None, timedelta(days=30))),
    )

    coeffs = CoefficientExtractor(store, basis).transform(tasks, target_table="customers")
    coeffs = CoefficientStandardizer().fit_transform(coeffs)

    temporal_routes = tuple(route for route in routes if route.hop_count > 0)
    sequences = TemporalSequenceBuilder(schema, store, temporal_routes, max_len=8).transform(
        tasks,
        target_table="customers",
    )

    model = RiftMambaModel(
        bases=basis,
        d_model=32,
        output_dim=2,
        event_dim=sequences.values.shape[-1],
        num_temporal_routes=len(temporal_routes),
        dropout=0.1,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    alpha = torch.from_numpy(coeffs.values).float()
    alpha_mask = torch.from_numpy(coeffs.masks).bool()
    events = torch.from_numpy(sequences.values).float()
    event_mask = torch.from_numpy(sequences.masks).bool()
    labels = torch.tensor([task.label for task in tasks], dtype=torch.long)

    for _ in range(3):
        optimizer.zero_grad()
        logits = model(alpha, alpha_mask, events, event_mask)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()

    print({"routes": len(routes), "basis": len(basis), "loss": round(float(loss.item()), 4)})


if __name__ == "__main__":
    main()
