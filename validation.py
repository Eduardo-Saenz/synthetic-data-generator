from __future__ import annotations

from datetime import date
from typing import Any


def _parse_comma_separated(text: str) -> list[str]:
    return [part.strip() for part in text.split(",") if part.strip()]


def validate_config(rows: int, columns: list[dict[str, Any]]) -> list[str]:
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
    transformed: list[dict[str, Any]] = []

    for col in columns:
        out = dict(col)
        col_type = out["column_type"]

        if col_type == "category":
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
