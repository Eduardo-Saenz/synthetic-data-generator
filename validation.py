from __future__ import annotations

"""Validation and schema-normalization helpers for the Streamlit app."""

from datetime import date
from typing import Any


def _parse_comma_separated(text: str) -> list[str]:
    """Split comma-separated user input while trimming empty entries."""
    return [part.strip() for part in text.split(",") if part.strip()]


def validate_config(rows: int, columns: list[dict[str, Any]]) -> list[str]:
    """Validate user configuration and return user-facing error messages."""
    errors: list[str] = []

    if rows <= 0:
        errors.append("El número de filas debe ser mayor a 0.")

    if not columns:
        errors.append("Debes agregar al menos una columna.")
        return errors

    names = [str(col.get("column_name", "")).strip() for col in columns]
    if any(not name for name in names):
        errors.append("Todos los nombres de columna son obligatorios.")

    if len(set(names)) != len(names):
        errors.append("Los nombres de columna deben ser únicos.")

    for idx, col in enumerate(columns, start=1):
        col_name = str(col.get("column_name", f"col_{idx}")).strip() or f"col_{idx}"
        col_type = col.get("column_type")

        if col_type == "int":
            if col.get("min", 0) > col.get("max", 0):
                errors.append(f"Columna '{col_name}': min debe ser <= max.")

        if col_type == "float":
            if col.get("min", 0.0) > col.get("max", 0.0):
                errors.append(f"Columna '{col_name}': min debe ser <= max.")

        if col_type == "number":
            # Numeric validation is distribution-aware.
            distribution = str(col.get("number_distribution", "uniform"))
            has_min = bool(col.get("has_min", False))
            has_max = bool(col.get("has_max", False))

            if distribution == "uniform":
                if float(col.get("uniform_low", 0.0)) >= float(col.get("uniform_high", 1.0)):
                    errors.append(
                        f"Columna '{col_name}': uniform_low debe ser < uniform_high."
                    )
            elif distribution == "normal":
                if float(col.get("std", 0.0)) <= 0:
                    errors.append(f"Columna '{col_name}': std debe ser > 0.")
            elif distribution == "lognormal":
                if float(col.get("sigma", 0.0)) <= 0:
                    errors.append(f"Columna '{col_name}': sigma debe ser > 0.")
            elif distribution == "exponential":
                if float(col.get("lambda_rate", 0.0)) <= 0:
                    errors.append(f"Columna '{col_name}': lambda debe ser > 0.")
            else:
                errors.append(
                    f"Columna '{col_name}': distribución numérica no soportada."
                )

            if has_min and has_max and float(col.get("min", 0.0)) > float(col.get("max", 0.0)):
                errors.append(f"Columna '{col_name}': min debe ser <= max.")

            if bool(col.get("clamp_to_range", False)) and not (has_min or has_max):
                errors.append(
                    f"Columna '{col_name}': clamp to range requiere min o max definido."
                )

        if col_type == "date":
            start = col.get("start_date")
            end = col.get("end_date")
            if isinstance(start, date) and isinstance(end, date) and start > end:
                errors.append(
                    f"Columna '{col_name}': start_date debe ser <= end_date."
                )

        if col_type == "category":
            values_raw = col.get("values_raw", "")
            values = _parse_comma_separated(values_raw) if isinstance(values_raw, str) else []
            if not values:
                errors.append(f"Columna '{col_name}': category requiere valores no vacíos.")
            weights_raw = col.get("weights_raw", "")
            if isinstance(weights_raw, str) and weights_raw.strip():
                weights = _parse_comma_separated(weights_raw)
                if len(weights) != len(values):
                    errors.append(
                        f"Columna '{col_name}': la cantidad de weights debe coincidir con values."
                    )
                else:
                    try:
                        numeric = [float(w) for w in weights]
                        if any(w < 0 for w in numeric):
                            errors.append(
                                f"Columna '{col_name}': los weights no pueden ser negativos."
                            )
                        if sum(numeric) == 0:
                            errors.append(
                                f"Columna '{col_name}': la suma de weights debe ser mayor a 0."
                            )
                    except ValueError:
                        errors.append(
                            f"Columna '{col_name}': los weights deben ser numéricos."
                        )

    return errors


def transform_columns(columns: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalize column schemas before generation.

    Includes backward compatibility mapping for legacy `int` and `float` types
    to the current `number` schema.
    """
    transformed: list[dict[str, Any]] = []

    for col in columns:
        out = dict(col)
        col_type = out["column_type"]

        if col_type == "int":
            # Legacy int -> number(uniform,int) mapping.
            out["column_type"] = "number"
            out["number_distribution"] = "uniform"
            out["number_output_type"] = "int"
            out["uniform_low"] = float(out.get("min", 0))
            out["uniform_high"] = float(out.get("max", 100))
            out["has_min"] = True
            out["has_max"] = True
            out["clamp_to_range"] = True

        if col_type == "float":
            # Legacy float -> number(uniform,float) mapping.
            out["column_type"] = "number"
            out["number_distribution"] = "uniform"
            out["number_output_type"] = "float"
            out["uniform_low"] = float(out.get("min", 0.0))
            out["uniform_high"] = float(out.get("max", 100.0))
            out["has_min"] = True
            out["has_max"] = True
            out["clamp_to_range"] = True

        if col_type == "category":
            # Parse category raw strings into typed generator inputs.
            values = [v.strip() for v in out.get("values_raw", "").split(",") if v.strip()]
            out["values"] = values
            weights_raw = str(out.get("weights_raw", "")).strip()
            if weights_raw:
                weights = [float(w.strip()) for w in weights_raw.split(",") if w.strip()]
                total = sum(weights)
                out["weights"] = [w / total for w in weights] if total > 0 else None
            else:
                out["weights"] = None

        transformed.append(out)

    return transformed
