# Synthetic Data Generator (Streamlit Demo)

Aplicación visual para generar datos sintéticos con columnas configurables, plantillas rápidas, vista previa y exportación.

## Requisitos

- Python 3.10+

## Instalación

```bash
pip install -r requirements.txt
```

## Ejecución

```bash
streamlit run app.py
```

## Estructura

- `app.py`: interfaz Streamlit, controles dinámicos, plantillas y flujo principal.
- `generators.py`: funciones de generación por tipo de columna.
- `validation.py`: validaciones de configuración y transformación de parámetros.
- `export_utils.py`: exportación a CSV y XLSX en bytes para descarga.

## Funcionalidades principales

- Configuración global:
  - filas (1 a 50,000)
  - seed reproducible o aleatoria
  - locale (`es_MX`, `es_ES`, `en_US`)
- Definición dinámica de columnas:
  - tipos: `faker`, `int`, `float`, `bool`, `category`, `date`, `uuid`, `text`
  - parámetros dinámicos según el tipo
  - opción `unique` (si aplica)
- Quick Templates:
  - `Web Users`
  - `E-commerce Orders`
  - `Student Records`
- Validaciones claras para evitar configuraciones inválidas.
- Generación de dataset con preview de las primeras 50 filas.
- Descarga de resultados en CSV y opcionalmente XLSX.
