from __future__ import annotations

import uuid
from datetime import date, timedelta
from typing import Any

import numpy as np
import pandas as pd
from faker import Faker
from faker.exceptions import UniquenessException


def generate_int_column(
    rows: int, min_value: int, max_value: int, rng: np.random.Generator, unique: bool
) -> tuple[list[int], str | None]:
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

        if col_type == "int":
            values, warning = generate_int_column(
                rows,
                int(col["min"]),
                int(col["max"]),
                rng,
                unique,
            )
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
