# t3q2.py — Q2/T3 worker logic: average RScore == 5 and BPrice == 2
import os, time
from mpi4py import MPI
import pandas as pd
from dotenv import load_dotenv

# Load shared env for convenience (master-only run will pass values anyway)
load_dotenv("/workspace/.env")

class MPISolution:
    """
    Count unique BId where average RScore == 5 and BPrice == 2.
    Designed so the master-only coordinator can call _process_slice(lo, hi)
    on workers. Also includes a 'run' that lets master participate if desired.
    """

    def __init__(self, dataset_path=None, dataset_size=None):
        self.dataset_path = dataset_path
        self.dataset_size = int(dataset_size) if dataset_size else 0

    def _process_slice(self, lo: int, hi: int):
        """
        Read only [lo, hi) rows and return:
          part_map: {BId: (sum_scores: float, cnt_scores: int, price_or_None: float|None)}
          local_hits: quick per-slice diagnostic (not the final answer)
        """
        df = pd.read_csv(
            self.dataset_path,
            skiprows=range(1, lo + 1),
            nrows=hi - lo,
            header=0,
            encoding="utf-8",
            low_memory=False,
            dtype={"BId": "string"},
        )
        for col in ("BId", "RScore", "BPrice"):
            if col not in df.columns:
                raise Exception(f"Column '{col}' not found in dataset.")

        df["RScore"] = pd.to_numeric(df["RScore"], errors="coerce")
        df["BPrice"] = pd.to_numeric(df["BPrice"], errors="coerce")
        df["BId"] = df["BId"].astype("string")

        # Aggregates per BId (avoid numeric_only on SeriesGroupBy for pandas compat)
        sum_per_book = (
            df.groupby("BId", as_index=False)["RScore"].sum()
              .rename(columns={"RScore": "rsum"})
        )
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

        # Local diagnostic only
        has_cnt = merged["rcnt"] > 0
        merged.loc[has_cnt, "avg"] = merged.loc[has_cnt, "rsum"] / merged.loc[has_cnt, "rcnt"]
        local_hits = int(((merged.get("avg") == 5) & (merged["BPrice"] == 2)).fillna(False).sum())

        # Build compact map: BId -> (sum, cnt, price_or_None)
        part_map = {}
        for _, row in merged.iterrows():
            bid = row.get("BId")
            if pd.isna(bid):
                continue
            s = float(row.get("rsum", 0.0) or 0.0)
            c = int(row.get("rcnt", 0) or 0)
            p = None if pd.isna(row.get("BPrice")) else float(row.get("BPrice"))
            part_map[str(bid)] = (s, c, p)

        return part_map, local_hits

    # Optional: participating run (not used by master-only, but handy to test)
    def run(self):
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
        t0 = time.time()

        if rank == 0:
            DATA_PATH = self.dataset_path or os.getenv("PATH_DATASET")
            DATASET_SIZE = self.dataset_size or int(os.getenv("DATASET_SIZE", "3000000"))
        else:
            DATA_PATH = None
            DATASET_SIZE = None

        DATA_PATH = comm.bcast(DATA_PATH, root=0)
        DATASET_SIZE = comm.bcast(DATASET_SIZE, root=0)

        if not DATA_PATH or not os.path.exists(DATA_PATH) or DATASET_SIZE <= 0:
            if rank == 0:
                print({"final_answer": "Dataset path/size missing or invalid",
                       "chunkSizePerThread": [], "answerPerThread": [], "totalTimeTaken": 0.0})
            return 0, [], [], 0.0

        self.dataset_path = DATA_PATH
        self.dataset_size = int(DATASET_SIZE)

        base = self.dataset_size // size
        lo = rank * base
        hi = (rank + 1) * base if rank < size - 1 else self.dataset_size

        try:
            part_map, local_hits = self._process_slice(lo, hi)
            err = ""
        except Exception as e:
            part_map, local_hits, err = {}, 0, str(e)

        all_maps = comm.gather(part_map, root=0)
        per_rank_hits = comm.gather(int(local_hits), root=0)
        errors = comm.gather(err, root=0)

        if rank == 0:
            if any(errors):
                print({"final_answer": f"Worker errors: {[e for e in errors if e]}",
                       "chunkSizePerThread": [], "answerPerThread": [], "totalTimeTaken": 0.0})
                return 0, [], [], 0.0

            global_map = {}
            for m in all_maps:
                for bid, (s, c, p) in m.items():
                    g = global_map.setdefault(bid, [0.0, 0, None])
                    g[0] += s
                    g[1] += c
                    if g[2] is None and p is not None:
                        g[2] = p

            final = 0
            for bid, (s, c, price) in global_map.items():
                if c > 0 and (s == 5 * c) and (price == 2):
                    final += 1

            chunks = [base] * (size - 1) + [self.dataset_size - base * (size - 1)]
            elapsed = time.time() - t0
            print({
                "final_answer": final,
                "chunkSizePerThread": chunks,
                "answerPerThread": per_rank_hits,
                "totalTimeTaken": elapsed
            })
            return final, chunks, per_rank_hits, elapsed
        return 0, [], [], 0.0


if __name__ == "__main__":
    # Optional standalone run (master participates)
    DATA_PATH = os.getenv("PATH_DATASET")
    DATASET_SIZE = int(os.getenv("DATASET_SIZE", "3000000"))
    MPISolution(dataset_path=DATA_PATH, dataset_size=DATASET_SIZE).run()