import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# =========================
# USER SETTINGS
# =========================
PROJECT_ROOT = Path("/Users/sebastianwiegandmoller/PycharmProjects/Speciale_final")
DATA_PATH = PROJECT_ROOT / "Regression" / "Data" / "MSCI Europe Energy Index.xlsx"
OUT_DIR = PROJECT_ROOT / "Regression" / "Output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PNG_OUT = OUT_DIR / "msci_europe_energy_index.png"
CSV_OUT = OUT_DIR / "msci_europe_energy_index_clean.csv"

# =========================
# LOAD + CLEAN
# =========================
if not DATA_PATH.exists():
    raise FileNotFoundError(f"Could not find input file: {DATA_PATH}")

if DATA_PATH.suffix.lower() in {".xlsx", ".xls"}:
    df = pd.read_excel(DATA_PATH)
else:
    df = pd.read_csv(DATA_PATH)
df.columns = [c.strip() for c in df.columns]

# Parse dates (accept both MM/DD/YYYY and ISO-like YYYY-MM-DD)
if "Date" not in df.columns:
    raise KeyError(f"Expected a 'Date' column, found: {list(df.columns)}")

df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df = df.dropna(subset=["Date"]).sort_values("Date")
# Restrict sample to start from 2005
df = df[df["Date"] >= "2005-01-01"]

# Detect the index/value column and clean numeric columns
value_col = None
for candidate in ["Price", "MSCI Europe Energy Index", "Index level", "Index"]:
    if candidate in df.columns:
        value_col = candidate
        break

if value_col is None:
    raise KeyError(f"Could not find a value column. Available columns: {list(df.columns)}")

for col in [value_col, "Open", "High", "Low"]:
    if col in df.columns:
        df[col] = (
            df[col].astype(str)
            .str.replace(",", ".", regex=False)
            .str.replace("%", "", regex=False)
            .str.strip()
        )
        df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=[value_col])

# Save a cleaned version (useful for later)
df[["Date", value_col]].rename(columns={value_col: "Price"}).to_csv(CSV_OUT, index=False)

# =========================
# PLOT (longest available)
# =========================
fig, ax = plt.subplots(figsize=(12, 5.5))

ax.plot(df["Date"], df[value_col], linewidth=2.0, color="#1f4e79")

# Add full borders around plot
for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(1.2)

ax.grid(axis="y", linestyle="--", alpha=0.4)

# Labels
ax.set_xlabel("Year", fontsize=11)
ax.set_ylabel("Index level", fontsize=11)
ax.tick_params(axis="both", labelsize=10)

import matplotlib.dates as mdates

# Improve x-axis date frequency (every 2 years)
ax.xaxis.set_major_locator(mdates.YearLocator(2))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

# Layout and save
fig.tight_layout()
fig.savefig(PNG_OUT, dpi=300, bbox_inches="tight")
plt.close(fig)

print("Saved cleaned series to:", CSV_OUT)
print("Saved plot to:", PNG_OUT)
print("Date range:", df["Date"].min().date(), "to", df["Date"].max().date())