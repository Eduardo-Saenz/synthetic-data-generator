Project Overview

We are building a visual synthetic data generator app using Python + Streamlit.

The goal is NOT to create a production-grade system, but a visually interactive demo for a university class. The main objective is to demonstrate that synthetic data can be generated dynamically with user customization options (similar to generatedata.com), including configurable column types and parameters.

The emphasis is on:

Visual interaction (sliders, selectors, dynamic inputs)

Customizable dataset structure

Preview + download functionality

Clean, understandable architecture

This project should be simple, clean, and demo-friendly.

Tech Stack

Python 3.10+

streamlit

pandas

numpy

faker

openpyxl (for optional XLSX export)

Core Requirements

1. Main App (Streamlit)

File: app.py

The UI must include:

Sidebar Controls

Global settings:

Slider: number of rows (1 to 50,000; default 1000)

Number input: seed (default 42; allow empty for random)

Selectbox: locale ("es_MX", "es_ES", "en_US")

Columns section:

Button: “Add Column”

For each column:

Text input: column_name

Selectbox: column_type with options:

faker

int

float

bool

category

date

uuid

text

Dynamic parameters depending on type:

int → min, max

float → min, max, decimals

bool → p_true (0–1 slider)

category → textarea for values (comma separated), optional weights

date → start_date, end_date

faker → method selector (name, email, phone_number, address, city, company, job, user_name, ipv4)

text → words (int)

Checkbox: unique (if applicable)

Button: Remove column

Main Area

Big button: “Generate Dataset”

Validation messages if:

Empty column names

Duplicate names

Invalid ranges

Empty category values

Preview table: first 50 rows

Display dataset size (rows × columns)

Download CSV button

Optional checkbox: also generate XLSX + download button

Generator Logic

File: generators.py

Each column type must have its own function.

Requirements:

Use Faker with selected locale

Respect seed for reproducibility

Category weights:

If provided, normalize automatically

If not, use uniform distribution

Dates uniformly distributed between start and end

Unique option:

Try to enforce uniqueness

If impossible after reasonable attempts, show warning

Validation

File: validation.py

Must validate:

rows > 0

column names not empty

column names unique

min <= max

start_date <= end_date

category values not empty

weights length matches values length

Return clear error messages to display in Streamlit.

Export

File: export_utils.py

CSV export using pandas

XLSX export using openpyxl

Return bytes for Streamlit download button

Templates Feature (Important)

The app must include a section in the sidebar:

"Quick Templates"

Add a selectbox with predefined templates:

None

Web Users

E-commerce Orders

Student Records

When a template is selected:

Automatically populate the column configuration

Allow user to modify after loading

Template Definitions
Web Users

Columns:

user_id → uuid (unique)

full_name → faker name

email → faker email (unique)

age → int 18–70

country → faker country

signup_date → date (2022-01-01 to today)

is_premium → bool p_true 0.3

E-commerce Orders

Columns:

order_id → uuid (unique)

user_email → faker email

product_name → faker word

category → category ["electronics", "clothing", "home", "books", "sports"]

price → float 5–2000 (2 decimals)

quantity → int 1–5

order_date → date (2023-01-01 to today)

shipped → bool p_true 0.85

Student Records

Columns:

student_id → uuid (unique)

full_name → faker name

age → int 17–30

major → category ["Engineering", "Business", "Arts", "Medicine", "Law"]

gpa → float 0–4 (2 decimals)

enrollment_date → date (2020-01-01 to today)

scholarship → bool p_true 0.25

UX Expectations

Clean layout

Dynamic parameter rendering

No crashes on invalid input

Clear warnings

Fast generation for up to 50k rows

Code must be modular and readable

Deliverables

Generate:

app.py

generators.py

validation.py

export_utils.py

requirements.txt

README.md with instructions:

pip install -r requirements.txt

streamlit run app.py

Keep code clean and well-commented.
Avoid unnecessary complexity.
This is a demo-oriented project.
