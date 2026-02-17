from __future__ import annotations

"""Data generation logic for each supported column type.

The design keeps one function per type so behavior is easy to test/extend.
All functions return `(values, warning)` where warning is shown in the UI when
uniqueness cannot be guaranteed with the provided constraints.
"""

import uuid
from datetime import date, timedelta
from typing import Any

import numpy as np
import pandas as pd
from faker import Faker
from faker.exceptions import UniquenessException


def _sample_number_distribution(
    rows: int, config: dict[str, Any], rng: np.random.Generator
) -> np.ndarray:
    """Sample raw numeric values from the configured distribution."""
    distribution = config.get("number_distribution", "uniform")

    if distribution == "uniform":
        low = float(config.get("uniform_low", 0.0))
        high = float(config.get("uniform_high", 1.0))
        return rng.uniform(low, high, size=rows)
    if distribution == "normal":
        mean = float(config.get("mean", 0.0))
        std = float(config.get("std", 1.0))
        return rng.normal(loc=mean, scale=std, size=rows)
    if distribution == "lognormal":
        mu = float(config.get("mu", 0.0))
        sigma = float(config.get("sigma", 1.0))
        return rng.lognormal(mean=mu, sigma=sigma, size=rows)
    if distribution == "exponential":
        lambda_rate = float(config.get("lambda_rate", 1.0))
        scale = 1.0 / lambda_rate
        return rng.exponential(scale=scale, size=rows)

    return rng.uniform(0.0, 1.0, size=rows)


def _postprocess_number_values(values: np.ndarray, config: dict[str, Any]) -> np.ndarray:
    """Apply optional clamp and cast/round strategy for numeric columns."""
    has_min = bool(config.get("has_min", False))
    has_max = bool(config.get("has_max", False))
    clamp_to_range = bool(config.get("clamp_to_range", False))
    min_value = float(config.get("min", np.min(values) if values.size else 0.0))
    max_value = float(config.get("max", np.max(values) if values.size else 0.0))

    if clamp_to_range and (has_min or has_max):
        low = min_value if has_min else -np.inf
        high = max_value if has_max else np.inf
        values = np.clip(values, low, high)

    output_type = config.get("number_output_type", "float")
    if output_type == "int":
        values = np.rint(values).astype(int)
        return values

    decimals = int(config.get("decimals", 2))
    return np.round(values.astype(float), decimals=decimals)


def generate_number_column(
    rows: int, config: dict[str, Any], rng: np.random.Generator, unique: bool
) -> tuple[list[float | int], str | None]:
    """Generate a `number` column with optional uniqueness.

    For unique mode we sample in batches and accumulate unseen values up to a
    bounded number of attempts to keep runtime predictable.
    """
    warning: str | None = None

    if not unique:
        values = _sample_number_distribution(rows, config, rng)
        values = _postprocess_number_values(values, config)
        return values.tolist(), None

    collected: list[float | int] = []
    seen = set()
    attempts = 0
    max_attempts = max(rows * 12, 1200)
    batch_size = min(max(rows, 100), 5000)

    while len(collected) < rows and attempts < max_attempts:
        attempts += 1
        generated = _sample_number_distribution(batch_size, config, rng)
        processed = _postprocess_number_values(generated, config).tolist()
        for value in processed:
            key = (type(value), value)
            if key not in seen:
                seen.add(key)
                collected.append(value)
                if len(collected) == rows:
                    break

    if len(collected) < rows:
        warning = (
            "No fue posible garantizar unicidad completa en number. "
            "Se completó con valores repetidos."
        )
        remaining = rows - len(collected)
        extra = _postprocess_number_values(
            _sample_number_distribution(remaining, config, rng), config
        ).tolist()
        collected.extend(extra)

    return collected, warning


def generate_int_column(
    rows: int, min_value: int, max_value: int, rng: np.random.Generator, unique: bool
) -> tuple[list[int], str | None]:
    """Legacy integer generator kept for backward compatibility."""
    if not unique:
        values = rng.integers(min_value, max_value + 1, size=rows)
        return values.tolist(), None

    available = max_value - min_value + 1
    if rows > available:
        warning = (
            f"No fue posible garantizar unicidad en enteros "
            f"({rows} filas > {available} valores posibles)."
        )
        values = rng.integers(min_value, max_value + 1, size=rows)
        return values.tolist(), warning

    values = rng.choice(np.arange(min_value, max_value + 1), size=rows, replace=False)
    return values.tolist(), None


def generate_float_column(
    rows: int,
    min_value: float,
    max_value: float,
    decimals: int,
    rng: np.random.Generator,
    unique: bool,
) -> tuple[list[float], str | None]:
    """Legacy float generator kept for backward compatibility."""
    scale = 10**decimals
    min_scaled = int(round(min_value * scale))
    max_scaled = int(round(max_value * scale))

    if not unique:
        raw = rng.integers(min_scaled, max_scaled + 1, size=rows)
        values = raw / scale
        return values.tolist(), None

    available = max_scaled - min_scaled + 1
    if rows > available:
        warning = (
            f"No fue posible garantizar unicidad en floats "
            f"({rows} filas > {available} valores posibles con {decimals} decimales)."
        )
        raw = rng.integers(min_scaled, max_scaled + 1, size=rows)
        return (raw / scale).tolist(), warning

    raw = rng.choice(np.arange(min_scaled, max_scaled + 1), size=rows, replace=False)
    values = raw / scale
    return values.tolist(), None


def generate_bool_column(
    rows: int, p_true: float, rng: np.random.Generator, unique: bool
) -> tuple[list[bool], str | None]:
    """Generate booleans with Bernoulli probability `p_true`."""
    if not unique:
        values = rng.choice([True, False], size=rows, p=[p_true, 1 - p_true])
        return values.tolist(), None

    if rows > 2:
        warning = "No fue posible garantizar unicidad en bool (solo hay 2 valores posibles)."
        values = rng.choice([True, False], size=rows, p=[p_true, 1 - p_true])
        return values.tolist(), warning

    base = [True, False]
    rng.shuffle(base)
    return base[:rows], None


def generate_category_column(
    rows: int,
    values: list[str],
    weights: list[float] | None,
    rng: np.random.Generator,
    unique: bool,
) -> tuple[list[str], str | None]:
    """Generate categorical data with optional normalized weights."""
    probs = None
    if weights:
        arr = np.array(weights, dtype=float)
        probs = arr / arr.sum()

    if not unique:
        sampled = rng.choice(values, size=rows, p=probs)
        return sampled.tolist(), None

    if rows > len(values):
        warning = (
            f"No fue posible garantizar unicidad en category "
            f"({rows} filas > {len(values)} valores posibles)."
        )
        sampled = rng.choice(values, size=rows, p=probs)
        return sampled.tolist(), warning

    indices = rng.choice(np.arange(len(values)), size=rows, replace=False)
    return [values[i] for i in indices], None


def generate_date_column(
    rows: int, start_date: date, end_date: date, rng: np.random.Generator, unique: bool
) -> tuple[list[date], str | None]:
    """Generate uniformly distributed dates in the closed [start, end] range."""
    day_span = (end_date - start_date).days

    if not unique:
        offsets = rng.integers(0, day_span + 1, size=rows)
        return [(start_date + timedelta(days=int(i))) for i in offsets], None

    available = day_span + 1
    if rows > available:
        warning = (
            f"No fue posible garantizar unicidad en date "
            f"({rows} filas > {available} fechas posibles)."
        )
        offsets = rng.integers(0, day_span + 1, size=rows)
        return [(start_date + timedelta(days=int(i))) for i in offsets], warning

    offsets = rng.choice(np.arange(0, day_span + 1), size=rows, replace=False)
    return [(start_date + timedelta(days=int(i))) for i in offsets], None


def generate_uuid_column(
    rows: int, rng: np.random.Generator, unique: bool
) -> tuple[list[str], str | None]:
    """Generate UUID strings based on RNG bits for reproducibility."""
    if unique:
        values = set()
        result: list[str] = []
        attempts = 0
        max_attempts = max(rows * 10, 1000)
        while len(result) < rows and attempts < max_attempts:
            attempts += 1
            part_a = int(rng.integers(0, 1 << 64, dtype=np.uint64))
            part_b = int(rng.integers(0, 1 << 64, dtype=np.uint64))
            v = str(uuid.UUID(int=((part_a << 64) | part_b)))
            if v not in values:
                values.add(v)
                result.append(v)
        if len(result) < rows:
            warning = "No fue posible garantizar unicidad completa en uuid."
            while len(result) < rows:
                part_a = int(rng.integers(0, 1 << 64, dtype=np.uint64))
                part_b = int(rng.integers(0, 1 << 64, dtype=np.uint64))
                result.append(str(uuid.UUID(int=((part_a << 64) | part_b))))
            return result, warning
        return result, None

    result = []
    for _ in range(rows):
        part_a = int(rng.integers(0, 1 << 64, dtype=np.uint64))
        part_b = int(rng.integers(0, 1 << 64, dtype=np.uint64))
        result.append(str(uuid.UUID(int=((part_a << 64) | part_b))))
    return result, None


def generate_faker_column(
    rows: int,
    faker_obj: Faker,
    method: str,
    unique: bool,
) -> tuple[list[str], str | None]:
    """Generate values from a selected Faker provider method."""
    if not unique:
        return [str(getattr(faker_obj, method)()) for _ in range(rows)], None

    result: list[str] = []
    warning: str | None = None

    try:
        faker_obj.unique.clear()
        for _ in range(rows):
            result.append(str(getattr(faker_obj.unique, method)()))
    except UniquenessException:
        warning = (
            f"No fue posible garantizar unicidad completa para faker '{method}'. "
            "Se completó con valores no únicos."
        )
        while len(result) < rows:
            result.append(str(getattr(faker_obj, method)()))
    finally:
        faker_obj.unique.clear()

    return result, warning


def generate_text_column(
    rows: int, faker_obj: Faker, words: int, unique: bool
) -> tuple[list[str], str | None]:
    """Generate text as a small word sequence from Faker."""
    def _make_text() -> str:
        return " ".join(faker_obj.words(nb=words))

    if not unique:
        return [_make_text() for _ in range(rows)], None

    values = set()
    result: list[str] = []
    attempts = 0
    max_attempts = max(rows * 15, 1000)

    while len(result) < rows and attempts < max_attempts:
        attempts += 1
        t = _make_text()
        if t not in values:
            values.add(t)
            result.append(t)

    if len(result) < rows:
        warning = "No fue posible garantizar unicidad completa en text."
        while len(result) < rows:
            result.append(_make_text())
        return result, warning

    return result, None


def generate_dataset(
    rows: int,
    columns: list[dict[str, Any]],
    locale: str,
    seed: int | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Generate a full DataFrame from the column schema list."""
    faker_obj = Faker(locale)
    rng = np.random.default_rng(seed)

    if seed is not None:
        faker_obj.seed_instance(seed)

    data: dict[str, list[Any]] = {}
    warnings: list[str] = []

    for col in columns:
        name = col["column_name"]
        col_type = col["column_type"]
        unique = bool(col.get("unique", False))
        warning: str | None = None

        # Legacy branches are intentionally kept to support older saved schemas.
        if col_type == "int":
            values, warning = generate_int_column(
                rows,
                int(col["min"]),
                int(col["max"]),
                rng,
                unique,
            )
        elif col_type == "number":
            values, warning = generate_number_column(rows, col, rng, unique)
        elif col_type == "float":
            values, warning = generate_float_column(
                rows,
                float(col["min"]),
                float(col["max"]),
                int(col["decimals"]),
                rng,
                unique,
            )
        elif col_type == "bool":
            values, warning = generate_bool_column(
                rows, float(col["p_true"]), rng, unique
            )
        elif col_type == "category":
            values, warning = generate_category_column(
                rows,
                col["values"],
                col.get("weights"),
                rng,
                unique,
            )
        elif col_type == "date":
            values, warning = generate_date_column(
                rows,
                col["start_date"],
                col["end_date"],
                rng,
                unique,
            )
        elif col_type == "uuid":
            values, warning = generate_uuid_column(rows, rng, unique)
        elif col_type == "faker":
            values, warning = generate_faker_column(
                rows,
                faker_obj,
                col["method"],
                unique,
            )
        elif col_type == "text":
            values, warning = generate_text_column(
                rows,
                faker_obj,
                int(col["words"]),
                unique,
            )
        else:
            values = [None] * rows
            warning = f"Tipo de columna no soportado: {col_type}"

        data[name] = values
        if warning:
            warnings.append(f"Columna '{name}': {warning}")

    return pd.DataFrame(data), warnings
