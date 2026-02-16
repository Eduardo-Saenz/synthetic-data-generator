from __future__ import annotations

from copy import deepcopy
from datetime import date

import streamlit as st

from export_utils import dataframe_to_csv_bytes, dataframe_to_xlsx_bytes
from generators import generate_dataset
from validation import transform_columns, validate_config


COLUMN_TYPES = ["faker", "int", "float", "bool", "category", "date", "uuid", "text"]
FAKER_METHODS = [
    "name",
    "email",
    "phone_number",
    "address",
    "city",
    "company",
    "job",
    "user_name",
    "ipv4",
    "country",
    "word",
]
LOCALES = ["es_MX", "es_ES", "en_US"]


def base_column() -> dict:
    return {
        "id": 0,
        "column_name": "col_1",
        "column_type": "int",
        "min": 0,
        "max": 100,
        "decimals": 2,
        "p_true": 0.5,
        "values_raw": "A,B,C",
        "weights_raw": "",
        "start_date": date(2024, 1, 1),
        "end_date": date.today(),
        "method": "name",
        "words": 5,
        "unique": False,
    }


def get_templates() -> dict[str, list[dict]]:
    today = date.today()
    return {
        "Web Users": [
            {
                "column_name": "user_id",
                "column_type": "uuid",
                "unique": True,
            },
            {
                "column_name": "full_name",
                "column_type": "faker",
                "method": "name",
                "unique": False,
            },
            {
                "column_name": "email",
                "column_type": "faker",
                "method": "email",
                "unique": True,
            },
            {
                "column_name": "age",
                "column_type": "int",
                "min": 18,
                "max": 70,
                "unique": False,
            },
            {
                "column_name": "country",
                "column_type": "faker",
                "method": "country",
                "unique": False,
            },
            {
                "column_name": "signup_date",
                "column_type": "date",
                "start_date": date(2022, 1, 1),
                "end_date": today,
                "unique": False,
            },
            {
                "column_name": "is_premium",
                "column_type": "bool",
                "p_true": 0.3,
                "unique": False,
            },
        ],
        "E-commerce Orders": [
            {
                "column_name": "order_id",
                "column_type": "uuid",
                "unique": True,
            },
            {
                "column_name": "user_email",
                "column_type": "faker",
                "method": "email",
                "unique": False,
            },
            {
                "column_name": "product_name",
                "column_type": "faker",
                "method": "word",
                "unique": False,
            },
            {
                "column_name": "category",
                "column_type": "category",
                "values_raw": "electronics,clothing,home,books,sports",
                "weights_raw": "",
                "unique": False,
            },
            {
                "column_name": "price",
                "column_type": "float",
                "min": 5.0,
                "max": 2000.0,
                "decimals": 2,
                "unique": False,
            },
            {
                "column_name": "quantity",
                "column_type": "int",
                "min": 1,
                "max": 5,
                "unique": False,
            },
            {
                "column_name": "order_date",
                "column_type": "date",
                "start_date": date(2023, 1, 1),
                "end_date": today,
                "unique": False,
            },
            {
                "column_name": "shipped",
                "column_type": "bool",
                "p_true": 0.85,
                "unique": False,
            },
        ],
        "Student Records": [
            {
                "column_name": "student_id",
                "column_type": "uuid",
                "unique": True,
            },
            {
                "column_name": "full_name",
                "column_type": "faker",
                "method": "name",
                "unique": False,
            },
            {
                "column_name": "age",
                "column_type": "int",
                "min": 17,
                "max": 30,
                "unique": False,
            },
            {
                "column_name": "major",
                "column_type": "category",
                "values_raw": "Engineering,Business,Arts,Medicine,Law",
                "weights_raw": "",
                "unique": False,
            },
            {
                "column_name": "gpa",
                "column_type": "float",
                "min": 0.0,
                "max": 4.0,
                "decimals": 2,
                "unique": False,
            },
            {
                "column_name": "enrollment_date",
                "column_type": "date",
                "start_date": date(2020, 1, 1),
                "end_date": today,
                "unique": False,
            },
            {
                "column_name": "scholarship",
                "column_type": "bool",
                "p_true": 0.25,
                "unique": False,
            },
        ],
    }


def normalize_column(raw: dict, col_id: int) -> dict:
    col = base_column()
    col.update(raw)
    col["id"] = col_id
    return col


def load_template(template_name: str) -> None:
    if template_name == "None":
        return
    templates = get_templates()
    selected = templates[template_name]
    columns = [normalize_column(col, idx) for idx, col in enumerate(deepcopy(selected))]
    st.session_state.columns = columns
    st.session_state.next_col_id = len(columns)


def on_template_change() -> None:
    selected = st.session_state.quick_template
    prev = st.session_state.get("last_loaded_template", "None")
    if selected != prev and selected != "None":
        load_template(selected)
    st.session_state.last_loaded_template = selected


def init_state() -> None:
    if "columns" not in st.session_state:
        st.session_state.columns = [normalize_column({}, 0)]
    if "next_col_id" not in st.session_state:
        st.session_state.next_col_id = 1
    if "generated_df" not in st.session_state:
        st.session_state.generated_df = None
    if "generation_warnings" not in st.session_state:
        st.session_state.generation_warnings = []
    if "last_loaded_template" not in st.session_state:
        st.session_state.last_loaded_template = "None"


def add_column() -> None:
    col = normalize_column({}, st.session_state.next_col_id)
    col["column_name"] = f"col_{len(st.session_state.columns) + 1}"
    st.session_state.columns.append(col)
    st.session_state.next_col_id += 1


def render_column_editor(idx: int, col: dict) -> bool:
    col_id = col["id"]
    with st.sidebar.expander(f"Column {idx + 1}", expanded=True):
        col["column_name"] = st.text_input(
            "column_name",
            value=col.get("column_name", ""),
            key=f"name_{col_id}",
        )
        current_type = col.get("column_type", "int")
        type_idx = COLUMN_TYPES.index(current_type) if current_type in COLUMN_TYPES else 0
        col["column_type"] = st.selectbox(
            "column_type",
            options=COLUMN_TYPES,
            index=type_idx,
            key=f"type_{col_id}",
        )

        col_type = col["column_type"]
        if col_type == "int":
            col["min"] = st.number_input(
                "min", value=int(col.get("min", 0)), step=1, key=f"int_min_{col_id}"
            )
            col["max"] = st.number_input(
                "max", value=int(col.get("max", 100)), step=1, key=f"int_max_{col_id}"
            )
        elif col_type == "float":
            col["min"] = st.number_input(
                "min",
                value=float(col.get("min", 0.0)),
                step=0.1,
                key=f"float_min_{col_id}",
            )
            col["max"] = st.number_input(
                "max",
                value=float(col.get("max", 100.0)),
                step=0.1,
                key=f"float_max_{col_id}",
            )
            col["decimals"] = st.number_input(
                "decimals",
                min_value=0,
                max_value=8,
                value=int(col.get("decimals", 2)),
                step=1,
                key=f"float_decimals_{col_id}",
            )
        elif col_type == "bool":
            col["p_true"] = st.slider(
                "p_true",
                min_value=0.0,
                max_value=1.0,
                value=float(col.get("p_true", 0.5)),
                step=0.01,
                key=f"bool_p_true_{col_id}",
            )
        elif col_type == "category":
            col["values_raw"] = st.text_area(
                "values (comma separated)",
                value=str(col.get("values_raw", "")),
                key=f"cat_values_{col_id}",
            )
            col["weights_raw"] = st.text_input(
                "weights (optional, comma separated)",
                value=str(col.get("weights_raw", "")),
                key=f"cat_weights_{col_id}",
            )
        elif col_type == "date":
            col["start_date"] = st.date_input(
                "start_date",
                value=col.get("start_date", date(2020, 1, 1)),
                key=f"date_start_{col_id}",
            )
            col["end_date"] = st.date_input(
                "end_date",
                value=col.get("end_date", date.today()),
                key=f"date_end_{col_id}",
            )
        elif col_type == "faker":
            current_method = col.get("method", "name")
            method_idx = FAKER_METHODS.index(current_method) if current_method in FAKER_METHODS else 0
            col["method"] = st.selectbox(
                "method",
                options=FAKER_METHODS,
                index=method_idx,
                key=f"faker_method_{col_id}",
            )
        elif col_type == "text":
            col["words"] = st.number_input(
                "words",
                min_value=1,
                max_value=50,
                value=int(col.get("words", 5)),
                step=1,
                key=f"text_words_{col_id}",
            )

        col["unique"] = st.checkbox(
            "unique (if applicable)",
            value=bool(col.get("unique", False)),
            key=f"unique_{col_id}",
        )

        return st.button("Remove column", key=f"remove_{col_id}")


def main() -> None:
    st.set_page_config(page_title="Synthetic Data Generator", layout="wide")
    init_state()

    st.title("Synthetic Data Generator")
    st.caption("Demo visual para generar datasets sintéticos configurables.")

    st.sidebar.header("Global Settings")
    rows = st.sidebar.slider("Rows", min_value=1, max_value=50000, value=1000, step=1)
    seed_is_random = st.sidebar.checkbox("Seed aleatoria (vacío)", value=False)
    seed_value = st.sidebar.number_input("Seed", min_value=0, value=42, step=1)
    locale = st.sidebar.selectbox("Locale", options=LOCALES, index=0)

    st.sidebar.divider()
    st.sidebar.subheader("Quick Templates")
    st.sidebar.selectbox(
        "Template",
        options=["None", "Web Users", "E-commerce Orders", "Student Records"],
        key="quick_template",
        on_change=on_template_change,
    )

    st.sidebar.divider()
    st.sidebar.subheader("Columns")
    st.sidebar.button("Add Column", on_click=add_column)

    remove_idx = None
    for i, column in enumerate(st.session_state.columns):
        to_remove = render_column_editor(i, column)
        if to_remove:
            remove_idx = i

    if remove_idx is not None:
        st.session_state.columns.pop(remove_idx)
        st.rerun()

    st.divider()
    generate_clicked = st.button("Generate Dataset", type="primary", use_container_width=True)

    if generate_clicked:
        errors = validate_config(rows, st.session_state.columns)
        if errors:
            st.session_state.generated_df = None
            for err in errors:
                st.error(err)
        else:
            seed = None if seed_is_random else int(seed_value)
            transformed = transform_columns(st.session_state.columns)
            df, warnings = generate_dataset(
                rows=rows,
                columns=transformed,
                locale=locale,
                seed=seed,
            )
            st.session_state.generated_df = df
            st.session_state.generation_warnings = warnings

    df = st.session_state.generated_df
    if df is not None:
        for warning in st.session_state.generation_warnings:
            st.warning(warning)

        st.success(f"Dataset generado: {df.shape[0]} filas x {df.shape[1]} columnas")
        st.subheader("Preview (first 50 rows)")
        st.dataframe(df.head(50), use_container_width=True)

        csv_bytes = dataframe_to_csv_bytes(df)
        st.download_button(
            label="Download CSV",
            data=csv_bytes,
            file_name="synthetic_dataset.csv",
            mime="text/csv",
        )

        export_xlsx = st.checkbox("Also generate XLSX")
        if export_xlsx:
            xlsx_bytes = dataframe_to_xlsx_bytes(df)
            st.download_button(
                label="Download XLSX",
                data=xlsx_bytes,
                file_name="synthetic_dataset.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )


if __name__ == "__main__":
    main()
