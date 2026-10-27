"""Build a deliberately messy synthetic DataFrame and run ``elessar.see`` on it.

Run:  python examples/demo.py
Writes the plots to ./elessar_report/ plus report.md, report.json and summary.csv.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import elessar


def make_messy_df(n: int = 2_000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    age = rng.normal(41, 12, n).round().clip(-3, 95)
    age[rng.choice(n, 25, replace=False)] = rng.choice([25, 30, 35, 40, 45, 50], 25)
    income = rng.lognormal(10.6, 0.55, n).round(-1)
    tenure = rng.gamma(2.0, 2.5, n).round(1)
    city = rng.choice(["Paris", "Lyon", "Marseille", "Lille", "Nantes", "Toulouse", "Nice", "Brest"], n,
                      p=[0.34, 0.2, 0.14, 0.1, 0.09, 0.07, 0.05, 0.01]).astype(object)
    city[rng.choice(n, 30, replace=False)] = "paris "
    signup = pd.Timestamp("2021-01-01") + pd.to_timedelta(rng.exponential(300, n), unit="D")
    satisfaction = rng.choice(["low", "medium", "high"], n, p=[0.2, 0.5, 0.3])
    visits = rng.negative_binomial(3, 0.35, n)
    spend = 0.012 * income + 25 * visits + rng.normal(0, 60, n)
    logit = 0.6 + 1.6 * (satisfaction == "low") - 1.4 * (satisfaction == "high") - 0.22 * visits
    churn = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)

    df = pd.DataFrame({
        "customer_id": np.arange(100_000, 100_000 + n),
        "uuid": [f"{rng.integers(16**8):08x}-{rng.integers(16**4):04x}-4abc-8def-{rng.integers(16**12):012x}" for _ in range(n)],
        "age": age,
        "income": income,
        "income_eur": income * 0.92,
        "tenure_years": tenure,
        "visits_per_month": visits,
        "spend": spend.round(2),
        "city": city,
        "satisfaction": satisfaction,
        "is_premium": rng.choice(["yes", "no"], n, p=[0.25, 0.75]),
        "signup_date": signup,
        "last_login": (signup + pd.to_timedelta(rng.integers(0, 400, n), unit="D")).strftime("%Y-%m-%d"),
        "email": [f"user{i}@{rng.choice(['gmail.com', 'yahoo.fr', 'proton.me', 'corp.io'])}" for i in range(n)],
        "phone": [f"+33 6 {rng.integers(10, 99)} {rng.integers(10, 99)} {rng.integers(10, 99)} {rng.integers(10, 99)}"
                  for _ in range(n)],
        "price_text": [f"{v:,.2f} €".replace(",", " ").replace(".", ",") for v in rng.gamma(3, 40, n)],
        "comment": rng.choice([
            "Great service, would recommend to my friends",
            "Delivery was late and the support never answered my emails",
            "Average experience overall, nothing special to report",
            "The app keeps crashing when I try to pay with my card",
            "Excellent quality for the price, I will order again soon",
        ], n) + " " + pd.Series(rng.integers(0, 10_000, n)).astype(str),
        "country": "FR",
        "legacy_score": np.nan,
        "tags": [["a", "b"] if i % 3 else {"k": i} for i in range(n)],
        "mixed_bag": [i if i % 4 else f"item-{i}" for i in range(n)],
        "account_status": np.where(churn == 1, "closed", "active"),
        "churn": churn,
    })
    df.loc[rng.choice(n, 400, replace=False), "income"] = np.nan
    df.loc[df["income"].isna(), "income_eur"] = np.nan
    df.loc[rng.choice(n, 1_200, replace=False), "tenure_years"] = np.nan
    df.loc[rng.choice(n, 30, replace=False), "spend"] = -999
    df.loc[rng.choice(n, 40, replace=False), "satisfaction"] = "N/A"
    df.loc[3, "spend"] = np.inf
    return pd.concat([df, df.iloc[:15]], ignore_index=True)


if __name__ == "__main__":
    df = make_messy_df()
    report = elessar.see(df, target="churn", out_dir="elessar_report")
    report.to_markdown("elessar_report/report.md")
    report.to_json("elessar_report/report.json")
    report.to_csv("elessar_report/summary.csv")
    print(f"\n{report!r}")
