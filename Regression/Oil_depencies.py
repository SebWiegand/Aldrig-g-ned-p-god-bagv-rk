
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt


# ============================================================
# Paths
# ============================================================
HERE = Path(__file__).resolve().parent
BASE_DIR = HERE
DATA_DIR = BASE_DIR / "Data"
OUTPUT_DIR = BASE_DIR / "Output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

STOCK_PATH = DATA_DIR / "Stock_data_final.xlsx"
OIL_PATH = DATA_DIR / "Oil price.xlsx"
GAS_PATH = DATA_DIR / "gas_price.xlsx"

for path_obj, label in [
    (STOCK_PATH, "stock"),
    (OIL_PATH, "oil"),
    (GAS_PATH, "gas"),
]:
    if not path_obj.exists():
        raise FileNotFoundError(f"Could not find the {label} file at: {path_obj}")


# ============================================================
# Settings
# ============================================================
MIN_OBS_PER_COMPANY = 100
USE_LOG_RETURNS = True
WINSORIZE_RETURNS = True
WINSOR_LOWER = 0.01
WINSOR_UPPER = 0.99


# ============================================================
# Helper functions
# ============================================================
def winsorize_series(s: pd.Series, lower: float = 0.01, upper: float = 0.99) -> pd.Series:
    """Winsorize a numeric series using quantiles."""
    if s.dropna().empty:
        return s
    lo = s.quantile(lower)
    hi = s.quantile(upper)
    return s.clip(lower=lo, upper=hi)


def load_price_file(file_path: Path, label: str, return_col: str) -> pd.DataFrame:
    """Load a commodity price file with a title row above the actual header."""
    print(f"Loading {label} data from: {file_path}")

    raw = pd.read_excel(file_path, header=None)

    header_row = None
    for idx, row in raw.iterrows():
        values = [str(v).strip() for v in row.tolist()]
        if "Date" in values and "Price" in values:
            header_row = idx
            break

    if header_row is None:
        raise ValueError(
            f"Could not find a header row in the {label} file containing both 'Date' and 'Price'."
        )

    df = pd.read_excel(file_path, header=header_row)
    df.columns = [str(c).strip() for c in df.columns]

    if not {"Date", "Price"}.issubset(df.columns):
        raise ValueError(
            f"{label.capitalize()} file must contain columns 'Date' and 'Price' after header detection. Found: {df.columns.tolist()}"
        )

    df = df[["Date", "Price"]].copy()
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    df["Price"] = (
        df["Price"]
        .astype(str)
        .str.replace(",", "", regex=False)
        .str.replace(" ", "", regex=False)
    )
    df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
    df = df.dropna(subset=["Date", "Price"]).sort_values("Date").reset_index(drop=True)

    if USE_LOG_RETURNS:
        df[return_col] = np.log(df["Price"] / df["Price"].shift(1))
    else:
        df[return_col] = df["Price"].pct_change()

    df = df.dropna(subset=[return_col]).copy()

    print(f"{label.capitalize()} rows:", len(df))
    print(f"Detected {label} columns:", df.columns.tolist())
    print(f"{label.capitalize()} date range:", df["Date"].min().date(), "to", df["Date"].max().date())

    return df


# ============================================================
# Load oil data
# ============================================================
oil = load_price_file(OIL_PATH, label="oil", return_col="oil_ret")


# ============================================================
# Load gas data
# ============================================================
gas = load_price_file(GAS_PATH, label="gas", return_col="gas_ret")


# ============================================================
# Load stock data
# ============================================================
print(f"Loading stock data from: {STOCK_PATH}")
stocks = pd.read_excel(STOCK_PATH)
stocks.columns = [str(c).strip() for c in stocks.columns]

required_cols = {"Ticker", "Company", "Date", "Price Close"}
if not required_cols.issubset(stocks.columns):
    raise ValueError(
        f"Stock file must contain columns {sorted(required_cols)}. Found: {stocks.columns.tolist()}"
    )

stocks = stocks[["Ticker", "Company", "Date", "Price Close"]].copy()
stocks["Date"] = pd.to_datetime(stocks["Date"], dayfirst=True, errors="coerce")
stocks["Price Close"] = (
    stocks["Price Close"]
    .astype(str)
    .str.replace(".", "", regex=False)
    .str.replace(",", ".", regex=False)
)
stocks["Price Close"] = pd.to_numeric(stocks["Price Close"], errors="coerce")
stocks["Company"] = stocks["Company"].astype(str).str.strip()
stocks["Ticker"] = stocks["Ticker"].astype(str).str.strip()

stocks = stocks.dropna(subset=["Company", "Date", "Price Close"]).copy()
stocks = stocks.sort_values(["Company", "Date"]).reset_index(drop=True)

print("Stock rows:", len(stocks))
print("Number of companies in stock file:", stocks["Company"].nunique())
print("Stock date range:", stocks["Date"].min().date(), "to", stocks["Date"].max().date())


# ============================================================
# Compute firm returns
# ============================================================
def compute_returns(group: pd.DataFrame) -> pd.DataFrame:
    group = group.sort_values("Date").copy()
    if USE_LOG_RETURNS:
        group["firm_ret"] = np.log(group["Price Close"] / group["Price Close"].shift(1))
    else:
        group["firm_ret"] = group["Price Close"].pct_change()
    return group


stocks = stocks.sort_values(["Company", "Date"]).copy()
if USE_LOG_RETURNS:
    stocks["firm_ret"] = stocks.groupby("Company")["Price Close"].transform(
        lambda s: np.log(s / s.shift(1))
    )
else:
    stocks["firm_ret"] = stocks.groupby("Company")["Price Close"].pct_change()
stocks = stocks.dropna(subset=["firm_ret"]).copy()

if WINSORIZE_RETURNS:
    stocks["firm_ret"] = stocks.groupby("Company")["firm_ret"].transform(
        lambda s: winsorize_series(s, WINSOR_LOWER, WINSOR_UPPER)
    )
    oil["oil_ret"] = winsorize_series(oil["oil_ret"], WINSOR_LOWER, WINSOR_UPPER)
    gas["gas_ret"] = winsorize_series(gas["gas_ret"], WINSOR_LOWER, WINSOR_UPPER)


# ============================================================
# Merge firm returns with oil and gas returns
# ============================================================
merged = stocks.merge(oil[["Date", "oil_ret"]], on="Date", how="inner")
merged = merged.merge(gas[["Date", "gas_ret"]], on="Date", how="inner")
print("Merged columns:", merged.columns.tolist())
merged = merged.dropna(subset=["firm_ret", "oil_ret", "gas_ret"]).copy()

print("Merged rows:", len(merged))
print("Merged unique dates:", merged["Date"].nunique())
print("Merged companies:", merged["Company"].astype(str).nunique())
print("Merged date range:", merged["Date"].min().date(), "to", merged["Date"].max().date())


# ============================================================
# 1. Pooled regression: do firms move with oil?
# ============================================================
X_pool = sm.add_constant(merged["oil_ret"])
y_pool = merged["firm_ret"]
pooled_model = sm.OLS(y_pool, X_pool).fit(cov_type="HC1")

print("\n=== POOLED REGRESSION: FIRM RETURNS ON OIL RETURNS ===")
print(pooled_model.summary())

pooled_results = pd.DataFrame(
    {
        "coef": pooled_model.params,
        "std_err": pooled_model.bse,
        "t_stat": pooled_model.tvalues,
        "p_value": pooled_model.pvalues,
    }
)
pooled_results.to_csv(OUTPUT_DIR / "oil_pooled_regression_results.csv")


# ============================================================
# 1b. Pooled regression: do firms move with gas?
# ============================================================
X_pool_gas = sm.add_constant(merged["gas_ret"])
y_pool_gas = merged["firm_ret"]
pooled_gas_model = sm.OLS(y_pool_gas, X_pool_gas).fit(cov_type="HC1")

print("\n=== POOLED REGRESSION: FIRM RETURNS ON GAS RETURNS ===")
print(pooled_gas_model.summary())

pooled_gas_results = pd.DataFrame(
    {
        "coef": pooled_gas_model.params,
        "std_err": pooled_gas_model.bse,
        "t_stat": pooled_gas_model.tvalues,
        "p_value": pooled_gas_model.pvalues,
    }
)
pooled_gas_results.to_csv(OUTPUT_DIR / "gas_pooled_regression_results.csv")


# ============================================================
# 2. Firm-by-firm oil and gas beta estimation
# ============================================================
firm_beta_rows = []

for company, g in merged.groupby("Company"):
    g = g.dropna(subset=["firm_ret", "oil_ret", "gas_ret"]).copy()
    if len(g) < MIN_OBS_PER_COMPANY:
        continue

    X_oil = sm.add_constant(g["oil_ret"])
    y = g["firm_ret"]
    res_oil = sm.OLS(y, X_oil).fit(cov_type="HC1")

    X_gas = sm.add_constant(g["gas_ret"])
    res_gas = sm.OLS(y, X_gas).fit(cov_type="HC1")

    corr_oil = g[["firm_ret", "oil_ret"]].corr().iloc[0, 1]
    corr_gas = g[["firm_ret", "gas_ret"]].corr().iloc[0, 1]

    firm_beta_rows.append(
        {
            "Company": company,
            "Ticker": g["Ticker"].iloc[0],
            "n_obs": len(g),
            "oil_beta": res_oil.params.get("oil_ret", np.nan),
            "oil_beta_se": res_oil.bse.get("oil_ret", np.nan),
            "oil_beta_t": res_oil.tvalues.get("oil_ret", np.nan),
            "oil_beta_p": res_oil.pvalues.get("oil_ret", np.nan),
            "oil_r_squared": res_oil.rsquared,
            "corr_with_oil": corr_oil,
            "gas_beta": res_gas.params.get("gas_ret", np.nan),
            "gas_beta_se": res_gas.bse.get("gas_ret", np.nan),
            "gas_beta_t": res_gas.tvalues.get("gas_ret", np.nan),
            "gas_beta_p": res_gas.pvalues.get("gas_ret", np.nan),
            "gas_r_squared": res_gas.rsquared,
            "corr_with_gas": corr_gas,
        }
    )

firm_betas = pd.DataFrame(firm_beta_rows)

if firm_betas.empty:
    raise ValueError(
        "No firm-level oil or gas betas were estimated. Check whether the stock, oil, and gas date ranges overlap and whether MIN_OBS_PER_COMPANY is too high."
    )

firm_betas = firm_betas.sort_values("oil_beta", ascending=False)
firm_betas.to_csv(OUTPUT_DIR / "oil_gas_firm_betas.csv", index=False)

print("\n=== FIRM-BY-FIRM OIL BETAS ===")
print(
    firm_betas[["Company", "Ticker", "n_obs", "oil_beta", "oil_beta_p", "corr_with_oil"]]
    .head(20)
    .to_string(index=False)
)
print("\nSummary of oil betas:")
print(firm_betas["oil_beta"].describe())

print("\n=== FIRM-BY-FIRM GAS BETAS ===")
print(
    firm_betas.sort_values("gas_beta", ascending=False)[["Company", "Ticker", "n_obs", "gas_beta", "gas_beta_p", "corr_with_gas"]]
    .head(20)
    .to_string(index=False)
)
print("\nSummary of gas betas:")
print(firm_betas["gas_beta"].describe())


# ============================================================
# 3. Daily average sample return vs oil and gas returns
# ============================================================
daily_avg = (
    merged.groupby("Date", as_index=False)
    .agg(
        avg_firm_ret=("firm_ret", "mean"),
        oil_ret=("oil_ret", "first"),
        gas_ret=("gas_ret", "first"),
    )
    .dropna()
)

daily_corr_oil = daily_avg[["avg_firm_ret", "oil_ret"]].corr().iloc[0, 1]
daily_corr_gas = daily_avg[["avg_firm_ret", "gas_ret"]].corr().iloc[0, 1]
print(f"\nCorrelation between average firm return and oil return: {daily_corr_oil:.4f}")
print(f"Correlation between average firm return and gas return: {daily_corr_gas:.4f}")

daily_avg.to_csv(OUTPUT_DIR / "oil_gas_daily_average_returns.csv", index=False)


# ============================================================
# 4. Plot average firm return vs oil return
# ============================================================
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(daily_avg["Date"], daily_avg["avg_firm_ret"], label="Average firm return")
ax.plot(daily_avg["Date"], daily_avg["oil_ret"], label="Oil return", alpha=0.8)
ax.set_title("Average sample return and oil return")
ax.set_xlabel("Date")
ax.set_ylabel("Daily return")
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
oil_plot_path = OUTPUT_DIR / "oil_vs_avg_firm_returns.png"
fig.savefig(oil_plot_path, dpi=300)
plt.close(fig)


# ============================================================
# 4b. Plot average firm return vs gas return
# ============================================================
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(daily_avg["Date"], daily_avg["avg_firm_ret"], label="Average firm return")
ax.plot(daily_avg["Date"], daily_avg["gas_ret"], label="Gas return", alpha=0.8)
ax.set_title("Average sample return and gas return")
ax.set_xlabel("Date")
ax.set_ylabel("Daily return")
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
gas_plot_path = OUTPUT_DIR / "gas_vs_avg_firm_returns.png"
fig.savefig(gas_plot_path, dpi=300)
plt.close(fig)


# ============================================================
# 5. Simple summary text file
# ============================================================
summary_lines = [
    "Oil and gas dependency test summary",
    "===================================",
    f"Oil data rows: {len(oil)}",
    f"Gas data rows: {len(gas)}",
    f"Merged rows: {len(merged)}",
    f"Merged companies: {merged['Company'].nunique()}",
    "",
    "Pooled oil regression:",
    f"Intercept: {pooled_model.params.get('const', np.nan):.6f}",
    f"Oil beta: {pooled_model.params.get('oil_ret', np.nan):.6f}",
    f"Oil beta t-stat: {pooled_model.tvalues.get('oil_ret', np.nan):.3f}",
    f"Oil beta p-value: {pooled_model.pvalues.get('oil_ret', np.nan):.6f}",
    f"Oil R-squared: {pooled_model.rsquared:.6f}",
    "",
    "Pooled gas regression:",
    f"Intercept: {pooled_gas_model.params.get('const', np.nan):.6f}",
    f"Gas beta: {pooled_gas_model.params.get('gas_ret', np.nan):.6f}",
    f"Gas beta t-stat: {pooled_gas_model.tvalues.get('gas_ret', np.nan):.3f}",
    f"Gas beta p-value: {pooled_gas_model.pvalues.get('gas_ret', np.nan):.6f}",
    f"Gas R-squared: {pooled_gas_model.rsquared:.6f}",
    "",
    f"Average firm-level oil beta: {firm_betas['oil_beta'].mean():.6f}",
    f"Median firm-level oil beta: {firm_betas['oil_beta'].median():.6f}",
    f"Average firm-level gas beta: {firm_betas['gas_beta'].mean():.6f}",
    f"Median firm-level gas beta: {firm_betas['gas_beta'].median():.6f}",
    f"Average daily correlation between sample return and oil: {daily_corr_oil:.6f}",
    f"Average daily correlation between sample return and gas: {daily_corr_gas:.6f}",
    "",
    f"Saved oil pooled regression results to: {OUTPUT_DIR / 'oil_pooled_regression_results.csv'}",
    f"Saved gas pooled regression results to: {OUTPUT_DIR / 'gas_pooled_regression_results.csv'}",
    f"Saved firm betas to: {OUTPUT_DIR / 'oil_gas_firm_betas.csv'}",
    f"Saved daily averages to: {OUTPUT_DIR / 'oil_gas_daily_average_returns.csv'}",
    f"Saved oil plot to: {oil_plot_path}",
    f"Saved gas plot to: {gas_plot_path}",
]

summary_path = OUTPUT_DIR / "oil_gas_dependency_summary.txt"
summary_path.write_text("\n".join(summary_lines), encoding="utf-8")

print(f"\nSaved oil pooled regression results to: {OUTPUT_DIR / 'oil_pooled_regression_results.csv'}")
print(f"Saved gas pooled regression results to: {OUTPUT_DIR / 'gas_pooled_regression_results.csv'}")
print(f"Saved firm betas to: {OUTPUT_DIR / 'oil_gas_firm_betas.csv'}")
print(f"Saved daily averages to: {OUTPUT_DIR / 'oil_gas_daily_average_returns.csv'}")
print(f"Saved oil plot to: {oil_plot_path}")
print(f"Saved gas plot to: {gas_plot_path}")
print(f"Saved summary to: {summary_path}")