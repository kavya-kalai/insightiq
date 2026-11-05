"""InsightIQ Sample Data Generator.

Generates a realistic 2,000-row enterprise sales dataset with deliberately
planted data quality issues to test and benchmark the Data Intelligence Engine.

Planted Issues:
1. Missing Values (NaNs in customer_id, quantity, revenue, marketing_spend)
2. Exact Duplicate Rows (~30 records)
3. Mixed & Inconsistent Date Formats (ISO, US, UK, text, and invalid strings)
4. Negative or Invalid Prices (negative values, zero values, string 'INVALID')
5. Statistical Outliers (extreme quantity and revenue values detectable via IQR)
"""

from __future__ import annotations
from pathlib import Path
import random
import numpy as np
import pandas as pd


def generate_dirty_sales_data(num_rows: int = 2000, seed: int = 42) -> pd.DataFrame:
    """Generate a realistic sales DataFrame with deliberately planted anomalies."""
    np.random.seed(seed)
    random.seed(seed)

    products = {
        "Laptop Pro": 1200.0,
        "Wireless Mouse": 29.99,
        "4K Monitor": 399.50,
        "Ergonomic Chair": 249.00,
        "USB-C Multi-Port Dock": 69.99,
        "Noise-Cancelling Headphones": 199.00,
        "Mechanical Keyboard": 89.00,
    }
    product_names = list(products.keys())
    regions = ["North", "South", "East", "West"]
    
    # Base clean generation
    product_choices = np.random.choice(product_names, size=num_rows)
    region_choices = np.random.choice(regions, size=num_rows, p=[0.35, 0.25, 0.20, 0.20])
    customer_ids = [f"CUST-{random.randint(1000, 1400)}" for _ in range(num_rows)]
    quantities = np.random.randint(1, 12, size=num_rows).astype(float)
    base_prices = np.array([products[p] for p in product_choices])
    
    # Introduce small realistic pricing variations (+- 5%)
    prices = np.round(base_prices * np.random.uniform(0.95, 1.05, size=num_rows), 2)
    revenues = np.round(quantities * prices, 2)
    marketing_spends = np.round(revenues * np.random.uniform(0.08, 0.22, size=num_rows) + 50.0, 2)

    # Dates spanning 12 months
    start_date = pd.Timestamp("2024-01-01")
    date_offsets = [pd.Timedelta(days=int(d)) for d in np.random.randint(0, 365, size=num_rows)]
    base_dates = [start_date + offset for offset in date_offsets]

    # Format dates with MIXED representations (deliberate inconsistency)
    formatted_dates: list[str] = []
    for dt in base_dates:
        fmt_rand = random.random()
        if fmt_rand < 0.65:
            # Standard ISO YYYY-MM-DD
            formatted_dates.append(dt.strftime("%Y-%m-%d"))
        elif fmt_rand < 0.80:
            # UK format DD/MM/YYYY
            formatted_dates.append(dt.strftime("%d/%m/%Y"))
        elif fmt_rand < 0.92:
            # US format MM-DD-YYYY
            formatted_dates.append(dt.strftime("%m-%d-%Y"))
        elif fmt_rand < 0.98:
            # Verbose Mon DD, YYYY
            formatted_dates.append(dt.strftime("%b %d, %Y"))
        else:
            # Corrupted / unparseable date
            formatted_dates.append(random.choice(["invalid_date", "2024-02-31", "pending", "N/A"]))

    df = pd.DataFrame({
        "date": formatted_dates,
        "product": product_choices,
        "region": region_choices,
        "customer_id": customer_ids,
        "quantity": quantities,
        "price": prices.astype(object),
        "revenue": revenues,
        "marketing_spend": marketing_spends,
    })

    # Plant Missing Values (NaNs)
    missing_idx_cust = np.random.choice(num_rows, size=65, replace=False)
    df.loc[missing_idx_cust, "customer_id"] = np.nan

    missing_idx_qty = np.random.choice(num_rows, size=40, replace=False)
    df.loc[missing_idx_qty, "quantity"] = np.nan

    missing_idx_rev = np.random.choice(num_rows, size=50, replace=False)
    df.loc[missing_idx_rev, "revenue"] = np.nan

    missing_idx_mkt = np.random.choice(num_rows, size=75, replace=False)
    df.loc[missing_idx_mkt, "marketing_spend"] = np.nan

    # Plant Negative & Invalid Prices
    neg_price_idx = np.random.choice(num_rows, size=15, replace=False)
    for idx in neg_price_idx:
        df.at[idx, "price"] = -round(abs(float(df.at[idx, "price"])), 2)

    zero_price_idx = np.random.choice(num_rows, size=10, replace=False)
    for idx in zero_price_idx:
        df.at[idx, "price"] = 0.0

    invalid_str_price_idx = np.random.choice(num_rows, size=8, replace=False)
    for idx in invalid_str_price_idx:
        df.at[idx, "price"] = random.choice(["INVALID", "N/A", "error"])

    # Plant Statistical Outliers (Quantities & Revenue)
    outlier_qty_idx = np.random.choice(num_rows, size=12, replace=False)
    for idx in outlier_qty_idx:
        df.at[idx, "quantity"] = float(random.choice([250, 400, 650]))
        # Correlate revenue with large quantity to create realistic huge transaction
        try:
            curr_price = float(df.at[idx, "price"])
            if curr_price > 0:
                df.at[idx, "revenue"] = round(df.at[idx, "quantity"] * curr_price, 2)
        except (ValueError, TypeError):
            pass

    outlier_rev_idx = np.random.choice(num_rows, size=8, replace=False)
    for idx in outlier_rev_idx:
        df.at[idx, "revenue"] = float(random.choice([125000.0, 189000.0, 240000.0]))

    # Plant Exact Duplicate Rows (~30 duplicates)
    dup_sources = np.random.choice(num_rows, size=30, replace=False)
    duplicates = df.iloc[dup_sources].copy()
    df = pd.concat([df, duplicates], ignore_index=True)

    # Shuffle dataset so anomalies and duplicates are distributed evenly
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    return df


def save_sample_dataset(output_path: Path | str | None = None) -> Path:
    """Generate and save the sample dataset to disk."""
    if output_path is None:
        project_root = Path(__file__).resolve().parent.parent
        output_path = project_root / "data" / "sample" / "sales_data_dirty.csv"
    else:
        output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = generate_dirty_sales_data()
    df.to_csv(output_path, index=False)
    return output_path


if __name__ == "__main__":
    saved_path = save_sample_dataset()
    print(f"Generated sample dataset at: {saved_path}")
    
    # Quick diagnostics
    df = pd.read_csv(saved_path)
    print(f"Total Rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    print(f"Duplicate Rows Count: {df.duplicated().sum()}")
    print("Missing Values Per Column:")
    print(df.isna().sum())

