import os
import time
from mpi4py import MPI
import pandas as pd
from dotenv import load_dotenv

load_dotenv()


class MPISolution:
    """Q2/T3: Count books with avg RScore == 5 and price == 2 using MPI."""

    def __init__(self, dataset_path=None, dataset_size=None):
        self.dataset_path = dataset_path
        self.dataset_size = dataset_size

    def _process_slice(self, lo: int, hi: int):
        """
        Each rank reads only its slice and returns:
          - part_map: {BId(str): (sum_scores, cnt_scores, price_or_None)}
          - local_hits: quick per-slice count (diagnostic only)
        """
        df = pd.read_csv(
            self.dataset_path,
            skiprows=range(1, lo + 1),
            nrows=hi - lo,
            header=0,
            encoding="utf-8",
            low_memory=False,
            dtype={"BId": "string"},  # keep IDs as strings (handles ISBN-like values)
        )
        for col in ("BId", "RScore", "BPrice"):
            if col not in df.columns:
                raise Exception(f"Column '{col}' not found in dataset.")

        df["RScore"] = pd.to_numeric(df["RScore"], errors="coerce")
        df["BPrice"] = pd.to_numeric(df["BPrice"], errors="coerce")
        df["BId"] = df["BId"].astype("string")

        # Per-slice aggregates
        sum_per_book = df.groupby("BId", as_index=False)["RScore"].sum(numeric_only=True)
        sum_per_book.rename(columns={"RScore": "rsum"}, inplace=True)

        cnt_per_book = (
            df.assign(rcnt=df["RScore"].notna().astype(int))
              .groupby("BId", as_index=False)["rcnt"].sum()
        )

        price_any = (
            df[["BId", "BPrice"]]
            .dropna(subset=["BPrice"])
            .drop_duplicates(subset=["BId"], keep="last")
        )

        merged = (
            sum_per_book
            .merge(cnt_per_book, on="BId", how="outer")
            .merge(price_any, on="BId", how="left")
        )

        # Local diagnostic count (final answer will be computed after global merge)
        has_cnt = merged["rcnt"] > 0
        merged.loc[has_cnt, "avg"] = merged.loc[has_cnt, "rsum"] / merged.loc[has_cnt, "rcnt"]
        local_hits = int(((merged.get("avg") == 5) & (merged["BPrice"] == 2)).fillna(False).sum())

        # Compact map back to master
        part_map = {
            row["BId"]: (
                float(row.get("rsum", 0.0) or 0.0),
                int(row.get("rcnt", 0) or 0),
                None if pd.isna(row.get("BPrice")) else float(row.get("BPrice")),
            )
            for _, row in merged.iterrows()
            if pd.notna(row.get("BId"))
        }
        return part_map, local_hits

    def run(self):
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        t0 = time.time()
        try:
            if not self.dataset_path:
                raise Exception("Dataset path not provided.")

            total = int(self.dataset_size)
            base = total // size
            lo = rank * base
            hi = (rank + 1) * base if rank < size - 1 else total

            part_map, local_hits = self._process_slice(lo, hi)

            all_maps = comm.gather(part_map, root=0)
            per_rank_hits = comm.gather(int(local_hits), root=0)

            if rank == 0:
                # Merge partials: BId -> [sum, cnt, price]
                global_map = {}
                for m in all_maps:
                    for bid, (s, c, p) in m.items():
                        if bid not in global_map:
                            global_map[bid] = [0.0, 0, None]
                        global_map[bid][0] += s
                        global_map[bid][1] += c
                        if global_map[bid][2] is None and p is not None:
                            global_map[bid][2] = p

                # Final count with exact equality
                final = 0
                for bid, (s, c, price) in global_map.items():
                    if c <= 0:
                        continue
                    avg = s / c
                    if avg == 5 and price == 2:
                        final += 1

                chunks = [base] * (size - 1) + [total - base * (size - 1)]
                elapsed = time.time() - t0
                result = {
                    "final_answer": final,
                    "chunkSizePerThread": chunks,
                    "answerPerThread": per_rank_hits,  # diagnostic only
                    "totalTimeTaken": elapsed,
                }
                # Master prints the dict for test.py (ast.literal_eval)
                print(result)
                return final, chunks, per_rank_hits, elapsed
            else:
                return 0, [], [], 0.0

        except Exception as err:
            if rank == 0:
                print({
                    "final_answer": str(err),
                    "chunkSizePerThread": [],
                    "answerPerThread": [],
                    "totalTimeTaken": 0.0
                })
            return str(err), [], [], 0.0


if __name__ == "__main__":
    DATA_PATH = os.getenv("PATH_DATASET")
    sol = MPISolution(dataset_path=DATA_PATH, dataset_size=300_000)
    sol.run()