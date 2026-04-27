from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

from rift_mamba import (
    BasisConfig,
    AtomicRouteEnumerator,
    BasisLayout,
    CoefficientExtractor,
    CoefficientStandardizer,
    ColumnSchema,
    CompositeBasisSpec,
    DatabaseSchema,
    DuckDBBackend,
    EventFeatureStandardizer,
    ForeignKey,
    InMemoryBackend,
    LinkExperimentConfig,
    LinkRiftDataset,
    LinkTaskRow,
    PairRiftMambaModel,
    PolarsBackend,
    RecordStore,
    RelBenchAdapter,
    RelationalDatasetBundle,
    RelationalBasis,
    RiftDataset,
    RiftMambaModel,
    RouteEnumerator,
    SchemaSemanticEncoder,
    SentenceTransformerTextEncoder,
    TableSchema,
    TaskRow,
    TaskSpec,
    TaskVectorHead,
    TaskVectorTargetBuilder,
    TemporalSequenceBuilder,
    audit_proxy_leakage,
    build_basis,
    build_composite_basis,
    build_exclude_columns,
    audit_route_proxy_leakage,
    prepare_link_experiment,
    sample_negative_links,
    prepare_experiment,
    ExperimentConfig,
    task_vector_cosine_loss,
    train_epoch,
)
from rift_mamba.nn import MambaBlock
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
    assert not any(route.table_path == ("customers", "transactions", "customers") for route in routes)


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


def test_duckdb_backend_pushdown_matches_in_memory_coefficients() -> None:
    schema = make_schema()
    store = make_store(schema)
    routes = RouteEnumerator(schema, max_hops=2).enumerate("customers")
    basis = build_basis(schema, routes, BasisConfig(windows=(None, timedelta(days=30))))
    tasks = (
        TaskRow(row_id=0, entity_id="c1", seed_time="2023-01-15"),
        TaskRow(row_id=1, entity_id="c1", seed_time="2023-02-15"),
    )

    memory = CoefficientExtractor(store, basis).transform(tasks, target_table="customers")
    duckdb = DuckDBBackend(schema, store.tables).coefficient_extractor(basis).transform(tasks, target_table="customers")

    assert duckdb.basis_names == memory.basis_names
    assert np.allclose(duckdb.values, memory.values, atol=1e-5)
    assert np.array_equal(duckdb.masks, memory.masks)


def test_polars_backend_pushdown_matches_in_memory_coefficients() -> None:
    schema = make_schema()
    store = make_store(schema)
    routes = RouteEnumerator(schema, max_hops=2).enumerate("customers")
    basis = build_basis(schema, routes, BasisConfig(windows=(None, timedelta(days=30))))
    tasks = (
        TaskRow(row_id=0, entity_id="c1", seed_time="2023-01-15"),
        TaskRow(row_id=1, entity_id="c1", seed_time="2023-02-15"),
    )

    memory = CoefficientExtractor(store, basis).transform(tasks, target_table="customers")
    polars = PolarsBackend(schema, store.tables).coefficient_extractor(basis).transform(tasks, target_table="customers")

    assert polars.basis_names == memory.basis_names
    assert np.allclose(polars.values, memory.values, atol=1e-5)
    assert np.array_equal(polars.masks, memory.masks)


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


def test_event_feature_standardizer_uses_train_sequence_statistics() -> None:
    schema = make_schema()
    store = make_store(schema)
    routes = RouteEnumerator(schema, max_hops=1).enumerate("customers")
    txn_route = next(route for route in routes if route.table_path == ("customers", "transactions"))
    builder = TemporalSequenceBuilder(schema, store, (txn_route,), max_len=4, value_embedding_dim=0)
    train_task = TaskRow(row_id=0, entity_id="c1", seed_time="2023-01-15")
    eval_task = TaskRow(row_id=1, entity_id="c1", seed_time="2023-02-15")
    train_seq = builder.transform([train_task], target_table="customers")
    eval_seq = builder.transform([eval_task], target_table="customers")

    standardizer = EventFeatureStandardizer().fit(train_seq)
    train_scaled = standardizer.transform(train_seq)
    eval_scaled = standardizer.transform(eval_seq)

    price_index = next(
        index for index, column in enumerate(train_seq.event_columns) if column.name == "transactions.price"
    ) * 2
    price_offset = standardizer.feature_indices.index(price_index)
    train_present = train_seq.masks & (train_seq.values[..., price_index + 1] > 0.5)
    raw_eval_price = eval_seq.values[..., price_index][eval_seq.values[..., price_index] == 20.0][0]
    scaled_eval_price = eval_scaled.values[..., price_index][eval_seq.values[..., price_index] == 20.0][0]

    assert np.isclose(train_scaled.values[..., price_index][train_present].mean(), 0.0)
    assert scaled_eval_price == pytest.approx(
        (raw_eval_price - standardizer.mean_[price_offset]) / standardizer.scale_[price_offset]
    )


def test_atomic_routes_generate_fact_bridge_routes() -> None:
    schema = make_schema()
    routes = AtomicRouteEnumerator(schema, max_hops=2).enumerate("customers")
    atomic_only = AtomicRouteEnumerator(schema, max_hops=2, atomic_only=True).enumerate("customers")

    assert any(route.table_path == ("customers", "transactions", "products") for route in routes)
    assert any(route.table_path == ("customers", "transactions", "products") for route in atomic_only)
    assert not any(route.table_path == ("customers", "transactions", "customers") for route in atomic_only)


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
    reversed_layout = BasisLayout.from_bases(tuple(reversed(basis)))
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
    assert reversed_layout.route_names == layout.route_names
    assert reversed_layout.slots == layout.slots
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


def test_model_encode_and_tve_training_epoch() -> None:
    schema = make_schema()
    store = make_store(schema)
    routes = RouteEnumerator(schema, max_hops=1).enumerate("customers")
    txn_route = next(route for route in routes if route.table_path == ("customers", "transactions"))
    bases = (RelationalBasis(0, txn_route, "count", None),)
    tasks = (
        TaskRow(0, "c1", "2023-01-15", label=1),
        TaskRow(1, "c1", "2023-02-15", label=0),
    )
    coeffs = CoefficientExtractor(store, bases).transform(tasks, "customers")
    seq = TemporalSequenceBuilder(schema, store, (txn_route,), max_len=4).transform(tasks, "customers")
    targets = TaskVectorTargetBuilder(store, bases, timedelta(days=45)).transform(tasks, "customers")
    dataset = RiftDataset(coeffs, sequences=seq, task_vector_targets=targets)
    loader = DataLoader(dataset, batch_size=2)
    model = RiftMambaModel(
        bases=bases,
        d_model=8,
        output_dim=2,
        basis_layers=1,
        event_dim=seq.values.shape[-1],
        num_temporal_routes=1,
        sequence_layers=1,
        use_mamba_ssm=False,
    )
    head = TaskVectorHead(model.feature_dim, targets.training_matrix().shape[1])
    optimizer = torch.optim.Adam([*model.parameters(), *head.parameters()], lr=1.0e-3)

    features = model.encode(
        torch.from_numpy(coeffs.values).float(),
        torch.from_numpy(coeffs.masks).bool(),
        torch.from_numpy(seq.values).float(),
        torch.from_numpy(seq.masks).bool(),
    )
    metrics = train_epoch(
        model,
        loader,
        optimizer,
        torch.nn.CrossEntropyLoss(),
        task_vector_head=head,
        task_vector_weight=0.1,
    )

    assert features.shape == (2, model.feature_dim)
    assert metrics.num_batches == 1
    assert np.isfinite(metrics.loss)


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
    schema = DatabaseSchema.from_tables(
        [
            *schema.tables.values(),
            TableSchema(
                name="orders",
                primary_key="order_id",
                columns=(
                    ColumnSchema("order_id", "primary_key", nullable=False),
                    ColumnSchema("customer_id", "foreign_key"),
                    ColumnSchema("payterms", "categorical"),
                    ColumnSchema("proxy_payterms", "categorical"),
                ),
            ),
        ],
        [
            *schema.foreign_keys,
            ForeignKey("orders", "customer_id", "customers", "customer_id", role="order_customer"),
        ],
    )
    tables = {
        "customers": [
            {"customer_id": "c1", "age": 30, "region": "yes", "proxy": "yes"},
            {"customer_id": "c2", "age": 40, "region": "no", "proxy": "no"},
        ],
        "orders": [
            {"order_id": "o1", "customer_id": "c1", "payterms": "net30", "proxy_payterms": "net30"},
            {"order_id": "o2", "customer_id": "c2", "payterms": "cod", "proxy_payterms": "cod"},
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
    assert isinstance(bundle.coefficient_backend(), InMemoryBackend)
    assert any(finding.column == "proxy" for finding in train.leakage_findings)
    assert all("|proxy|" not in name for name in train.coefficients.basis_names)
    assert all(column.column not in {"region", "proxy"} for column in train.sequences.event_columns)


def test_route_proxy_leakage_audit_finds_cross_table_proxy() -> None:
    schema = make_schema()
    schema = DatabaseSchema.from_tables(
        [
            *schema.tables.values(),
            TableSchema(
                name="orders",
                primary_key="order_id",
                columns=(
                    ColumnSchema("order_id", "primary_key", nullable=False),
                    ColumnSchema("customer_id", "foreign_key"),
                    ColumnSchema("proxy_payterms", "categorical"),
                ),
            ),
        ],
        [
            *schema.foreign_keys,
            ForeignKey("orders", "customer_id", "customers", "customer_id", role="order_customer"),
        ],
    )
    store = RecordStore(
        schema,
        {
            "customers": [{"customer_id": "c1", "age": 30, "region": "east"}, {"customer_id": "c2", "age": 40, "region": "west"}],
            "orders": [
                {"order_id": "o1", "customer_id": "c1", "proxy_payterms": "net30"},
                {"order_id": "o2", "customer_id": "c2", "proxy_payterms": "cod"},
            ],
            "transactions": [],
            "products": [],
        },
    )
    routes = RouteEnumerator(schema, max_hops=1).enumerate("customers")
    findings = audit_route_proxy_leakage(
        store,
        schema,
        routes,
        [TaskRow(0, "c1", "2023-01-01", "net30"), TaskRow(1, "c2", "2023-01-01", "cod")],
        target_table="customers",
        exclude_columns={("customers", "region")},
    )

    assert any(finding.table == "orders" and finding.column == "proxy_payterms" for finding in findings)


def test_relbench_official_object_adapter_maps_entity_and_splits() -> None:
    class FakeTable:
        def __init__(self, df, pkey_col=None, time_col=None, fkeys=None):
            self.df = df
            self.pkey_col = pkey_col
            self.time_col = time_col
            self.fkey_col_to_pkey_table = fkeys or {}

    class FakeDB:
        table_dict = {
            "users": FakeTable(
                pd.DataFrame({"user_id": [0, 1], "created_at": pd.to_datetime(["2023-01-01", "2023-01-02"]), "age": [10, 20]}),
                pkey_col="user_id",
                time_col="created_at",
            )
        }

    class FakeDataset:
        val_timestamp = pd.Timestamp("2023-02-01")
        test_timestamp = pd.Timestamp("2023-03-01")

        def get_db(self, upto_test_timestamp=True):
            return FakeDB()

    class FakeTask:
        entity_table = "users"
        entity_col = "user_id"
        time_col = "seed_time"
        target_col = "label"
        task_type = "binary_classification"
        timedelta = pd.Timedelta(days=1)

        def get_table(self, split, mask_input_cols=None):
            df = pd.DataFrame(
                {
                    "seed_time": pd.to_datetime(["2023-01-10"]),
                    "user_id": [0],
                    "label": [1],
                }
            )
            if mask_input_cols:
                df = df[["seed_time", "user_id"]]
            return FakeTable(df, time_col="seed_time", fkeys={"user_id": "users"})

    bundle = RelBenchAdapter.from_relbench_objects(FakeDataset(), FakeTask(), include_test_labels=True).load()

    assert bundle.task.target_table == "users"
    assert bundle.split_rows["train"][0].entity_id == 0
    assert bundle.split_rows["test"][0].label == 1


def test_link_prediction_preparation_pair_model_and_negative_sampling() -> None:
    schema = make_schema()
    store = make_store(schema)
    task = TaskSpec(
        target_table="customers",
        entity_column="customer_id",
        seed_time_column="seed_time",
        label_column="product_id",
        task_type="link_prediction",
        src_entity_table="customers",
        src_entity_column="customer_id",
        dst_entity_table="products",
        dst_entity_column="product_id",
        eval_k=2,
    )
    positives = (LinkTaskRow(0, "c1", "p1", "2023-01-15", 1),)
    rows = sample_negative_links(positives, ["p1", "p2"], num_negatives=1, seed=1)
    bundle = RelationalDatasetBundle(schema, store.tables, rows, task)
    prepared = prepare_link_experiment(
        bundle,
        rows,
        LinkExperimentConfig(max_hops=1, sequence_max_len=4),
    )
    dataset = LinkRiftDataset(prepared)
    src_model = RiftMambaModel(
        prepared.src_coefficients.bases,
        d_model=8,
        output_dim=2,
        event_dim=prepared.src_sequences.values.shape[-1],
        num_temporal_routes=prepared.src_sequences.values.shape[1],
        basis_layers=1,
        sequence_layers=1,
        use_mamba_ssm=False,
    )
    dst_model = RiftMambaModel(
        prepared.dst_coefficients.bases,
        d_model=8,
        output_dim=2,
        event_dim=prepared.dst_sequences.values.shape[-1],
        num_temporal_routes=prepared.dst_sequences.values.shape[1],
        basis_layers=1,
        sequence_layers=1,
        use_mamba_ssm=False,
    )
    pair_model = PairRiftMambaModel(src_model, dst_model)
    batch = dataset[0]
    score = pair_model(
        batch["src_alpha"].unsqueeze(0),
        batch["src_alpha_mask"].unsqueeze(0),
        batch["dst_alpha"].unsqueeze(0),
        batch["dst_alpha_mask"].unsqueeze(0),
        batch["src_events"].unsqueeze(0),
        batch["src_event_mask"].unsqueeze(0),
        batch["dst_events"].unsqueeze(0),
        batch["dst_event_mask"].unsqueeze(0),
    )

    assert len(rows) == 2
    assert score.shape == (1,)
    assert torch.isfinite(score).all()


def test_sequence_builder_excludes_autocomplete_target_and_proxy_columns() -> None:
    schema = make_schema()
    store = make_store(schema)
    routes = RouteEnumerator(schema, max_hops=1).enumerate("customers")
    task = TaskSpec(
        target_table="customers",
        entity_column="customer_id",
        seed_time_column="seed_time",
        label_column="region",
        task_type="autocomplete",
        target_column=("customers", "region"),
        leakage_columns=(("customers", "age"),),
    )
    seq = TemporalSequenceBuilder(
        schema,
        store,
        routes,
        max_len=4,
        exclude_columns=build_exclude_columns(task, schema),
    ).transform([TaskRow(0, "c1", "2023-01-15")], "customers")

    assert all(column.column not in {"region", "age"} for column in seq.event_columns)


def test_composite_path_basis_conditions_on_intermediate_path_values() -> None:
    schema = make_schema()
    store = make_store(schema)
    routes = RouteEnumerator(schema, max_hops=2).enumerate("customers")
    product_route = next(route for route in routes if route.table_path == ("customers", "transactions", "products"))
    specs = (
        CompositeBasisSpec(
            route=product_route,
            value_table="transactions",
            value_column="price",
            condition_table="products",
            condition_column="category",
            condition_value="electronics",
            aggregator="group_mean",
            window=timedelta(days=30),
        ),
    )
    bases = build_composite_basis(schema, specs)
    matrix = CoefficientExtractor(store, bases).transform([TaskRow(0, "c1", "2023-01-15")], "customers")

    assert matrix.masks[0, 0]
    assert matrix.values[0, 0] == 10.0


def test_sentence_transformer_encoder_is_optional_and_mock_encoder_is_usable() -> None:
    class MockEncoder:
        dim = 3

        def __init__(self) -> None:
            self.calls = 0

        def encode(self, text: str) -> np.ndarray:
            self.calls += 1
            return np.ones(3, dtype=np.float32)

    mock = MockEncoder()
    encoder = SchemaSemanticEncoder(mock)
    schema = make_schema()
    routes = RouteEnumerator(schema, max_hops=0).enumerate("customers")
    basis = build_basis(schema, routes, BasisConfig(windows=(None,)))

    matrix = encoder.basis_matrix(basis)
    assert matrix.shape == (len(basis), 3)
    assert mock.calls == len(basis)
    assert SentenceTransformerTextEncoder is not None


def test_mamba_padding_diagnostics_and_exact_mask_are_padding_invariant() -> None:
    block = MambaBlock(d_model=4, use_mamba_ssm=False, allow_masked_mamba=False)
    x = torch.randn(2, 5, 4)
    mask = torch.tensor([[False, False, True, True, True], [False, True, True, True, True]])
    changed = x.clone()
    changed[~mask] = 1.0e6

    y1 = block(x, mask)
    y2 = block(changed, mask)
    assert block.implementation_name == "fallback_causal_gated_ssm"
    assert torch.allclose(y1[mask], y2[mask], atol=1e-4)
    assert torch.isfinite(block.masked_approximation_error(x, mask))


def test_tve_loss_handles_null_indicators_empty_masks_and_reweighting() -> None:
    schema = make_schema()
    store = make_store(schema)
    routes = RouteEnumerator(schema, max_hops=1).enumerate("customers")
    txn_route = next(route for route in routes if route.table_path == ("customers", "transactions"))
    mean_basis = (RelationalBasis(0, txn_route, "mean", None, "price", "numeric"),)
    empty_targets = TaskVectorTargetBuilder(store, mean_basis, timedelta(days=1)).transform(
        [TaskRow(0, "c1", "2023-01-15")],
        "customers",
    )

    assert empty_targets.masks.tolist() == [[False]]
    assert empty_targets.null_indicators.tolist() == [[True]]
    assert empty_targets.training_matrix().tolist() == [[0.0, 1.0]]
    assert empty_targets.training_mask().tolist() == [[True, True]]
    assert empty_targets.sample_weights().tolist() == [1.0]

    pred = torch.zeros(3, 4)
    target = torch.zeros(3, 4)
    mask = torch.zeros(3, 4, dtype=torch.bool)
    loss = task_vector_cosine_loss(pred, target, mask)

    assert torch.isfinite(loss)
    assert loss.item() == 0.0

    weighted = task_vector_cosine_loss(
        torch.ones(2, 2),
        torch.tensor([[1.0, 0.0], [1.0, 1.0]]),
        torch.tensor([[True, False], [True, True]]),
        sample_weight=torch.tensor([1.0, 0.5]),
    )
    assert torch.isfinite(weighted)
