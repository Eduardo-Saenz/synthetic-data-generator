from __future__ import annotations

"""Export helpers that convert DataFrames to downloadable byte payloads."""

from io import BytesIO

import pandas as pd


def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    """Serialize a DataFrame as UTF-8 CSV bytes for Streamlit download."""
    return df.to_csv(index=False).encode("utf-8")


def dataframe_to_xlsx_bytes(df: pd.DataFrame) -> bytes:
    """Serialize a DataFrame as XLSX bytes using openpyxl engine."""
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="data")
    return output.getvalue()
