# t3q3.py — Q3/T3 worker logic:
# Find user name(s) who reviewed the largest set of books with avg RScore == 4.
import os
from collections import defaultdict, Counter
import pandas as pd
from dotenv import load_dotenv

# Load shared env path (master will also pass path/size explicitly)
load_dotenv("/workspace/.env")

class MPISolution:
    """
    Per-slice worker:
      _work(lo, hi) -> (payload_dict, users_seen_count)
    payload_dict maps UId -> {
        "sum": int total of RScore,
        "cnt": int count of RScore entries,
        "books": list of distinct BId reviewed,
        "name_counts": {display_name: occurrences}
    }
    """

    def __init__(self, dataset_path=None, dataset_size=None):
        self.dataset_path = dataset_path
        self.dataset_size = int(dataset_size) if dataset_size else 0

    def _work(self, lo: int, hi: int):
        # Read only this rank's slice
        df = pd.read_csv(
            self.dataset_path,
            skiprows=range(1, lo + 1),
            nrows=max(0, hi - lo),
            header=0,
            encoding="utf-8",
            low_memory=False,
            dtype={"UId": "string", "UName": "string", "BId": "string"},
        )

        # Validate required columns
        for col in ("UId", "UName", "BId", "RScore"):
            if col not in df.columns:
                raise Exception(f"Column '{col}' not found in dataset.")

        # Normalize types
        df["RScore"] = pd.to_numeric(df["RScore"], errors="coerce")

        # Aggregate per user
        local = defaultdict(lambda: {"sum": 0, "cnt": 0, "books": set(), "name_counts": Counter()})
        for _, row in df.iterrows():
            r = row["RScore"]
            if pd.isna(r):
                continue
            uid = row["UId"]
            bid = row["BId"]

            # Skip rows with missing identifiers
            if pd.isna(uid) or pd.isna(bid):
                continue

            uid = str(uid)
            bid = str(bid)

            local[uid]["sum"] += int(r)          # integer sums keep exact avg check
            local[uid]["cnt"] += 1
            local[uid]["books"].add(bid)

            name = row["UName"]
            if not pd.isna(name):
                s = str(name).strip()
                if s:
                    local[uid]["name_counts"][s] += 1

        users_seen = len(local)

        # Make payload pickle-friendly (no sets/Counters)
        payload = {
            uid: {
                "sum": info["sum"],
                "cnt": info["cnt"],
                "books": list(info["books"]),
                "name_counts": dict(info["name_counts"]),
            }
            for uid, info in local.items()
        }
        return payload, users_seen