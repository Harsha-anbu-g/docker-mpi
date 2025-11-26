# t3q4.py — Q4/T3 worker logic:
# Top 10 highest-price books with exact average RScore < 4 (integer-safe).
import os
from collections import defaultdict, Counter
import pandas as pd
from dotenv import load_dotenv

# Load shared env (master will also pass path/size explicitly)
load_dotenv("/workspace/.env")

class MPISolution:
    """
    Per-slice worker for Q4/T3.
    _work(lo, hi) -> (payload_dict, seen_books_count)
      payload_dict maps BId -> {
        "sum": int total RScore,
        "cnt": int count of RScore entries,
        "price": float|None (first non-null seen),
        "title_counts": {title: occurrences}
      }
    """

    def __init__(self, dataset_path=None, dataset_size=None):
        self.dataset_path = dataset_path
        self.dataset_size = int(dataset_size) if dataset_size else 0

    def _work(self, lo: int, hi: int):
        # Read only [lo, hi) rows
        df = pd.read_csv(
            self.dataset_path,
            skiprows=range(1, lo + 1),
            nrows=max(0, hi - lo),
            header=0,
            encoding="utf-8",
            low_memory=False,
            dtype={"BId": "string", "BTitle": "string"},
        )

        # Validate required columns
        for col in ("BId", "BTitle", "BPrice", "RScore"):
            if col not in df.columns:
                raise Exception(f"Column '{col}' not found in dataset.")

        # Normalize types
        df["RScore"] = pd.to_numeric(df["RScore"], errors="coerce")
        df["BPrice"] = pd.to_numeric(df["BPrice"], errors="coerce")

        # Aggregate per book
        local = defaultdict(lambda: {"sum": 0, "cnt": 0, "price": None, "title_counts": Counter()})
        for _, row in df.iterrows():
            r = row["RScore"]
            if pd.isna(r):
                continue
            bid = row["BId"]
            if pd.isna(bid):
                continue
            bid = str(bid)

            entry = local[bid]
            entry["sum"] += int(r)      # keep integer sums to use exact sum < 4*cnt check
            entry["cnt"] += 1

            price = row["BPrice"]
            if entry["price"] is None and pd.notna(price):
                entry["price"] = float(price)

            title = row["BTitle"]
            if not pd.isna(title):
                s = str(title).strip()
                if s:
                    entry["title_counts"][s] += 1

        # Make payload pickle-friendly
        payload = {
            bid: {
                "sum": info["sum"],
                "cnt": info["cnt"],
                "price": info["price"],
                "title_counts": dict(info["title_counts"]),
            }
            for bid, info in local.items()
        }
        return payload, len(local)