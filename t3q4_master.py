# t3q4_master.py — Master-only coordinator for Q4/T3
# Rank 0 coordinates; ranks 1..N-1 run MPISolution._work(lo, hi) and return partials.
import os, time
from collections import Counter, defaultdict
from mpi4py import MPI
from dotenv import load_dotenv
from t3q4 import MPISolution  # worker logic

# Load shared config from mounted path (robust to CWD)
load_dotenv("/workspace/.env")

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

TAG_RANGE = 401
TAG_RESULT = 402
TAG_ERROR = 403

def pick_title(tc: dict) -> str:
    """Most frequent title; on tie, lexicographically smallest. Empty if none."""
    if not tc:
        return ""
    c = Counter(tc)
    top = max(c.values())
    cands = sorted([t for t, v in c.items() if v == top])
    return cands[0]

def main():
    t0 = time.time()

    DATA_PATH = os.getenv("PATH_DATASET")
    try:
        DATASET_SIZE = int(os.getenv("DATASET_SIZE", "3000000") or "3000000")
    except Exception:
        DATASET_SIZE = 0

    workers = size - 1
    if workers < 1:
        if rank == 0:
            print({"final_answer": "Need at least 1 worker (run with -n >= 2)",
                   "chunkSizePerThread": [], "answerPerThread": [], "totalTimeTaken": 0.0})
        return

    if rank == 0:
        # Validate once on master
        if (not DATA_PATH) or (not os.path.exists(DATA_PATH)) or (DATASET_SIZE <= 0):
            print({"final_answer": "Dataset path/size missing or invalid",
                   "chunkSizePerThread": [], "answerPerThread": [], "totalTimeTaken": 0.0})
            return

        # Split rows across workers only (master does zero compute)
        base = DATASET_SIZE // workers
        ranges = []
        for w in range(workers):
            lo = w * base
            hi = (w + 1) * base if w < workers - 1 else DATASET_SIZE
            ranges.append((lo, hi))

        # Send (DATA_PATH, DATASET_SIZE, lo, hi) to each worker rank
        for r in range(1, size):
            lo, hi = ranges[r - 1]
            comm.send((DATA_PATH, DATASET_SIZE, lo, hi), dest=r, tag=TAG_RANGE)

        # Gather from workers
        worker_partials = []
        per_rank_seen = []
        worker_errors = []

        for r in range(1, size):
            status = MPI.Status()
            comm.probe(source=r, tag=MPI.ANY_TAG, status=status)
            if status.Get_tag() == TAG_ERROR:
                err = comm.recv(source=r, tag=TAG_ERROR)
                worker_errors.append(f"[rank {r}] {err}")
            else:
                part_map, seen_books = comm.recv(source=r, tag=TAG_RESULT)
                worker_partials.append(part_map)
                per_rank_seen.append(int(seen_books))

        if worker_errors:
            print({"final_answer": f"Worker errors: {worker_errors}",
                   "chunkSizePerThread": [], "answerPerThread": [], "totalTimeTaken": 0.0})
            return

        # Merge per-book aggregates
        books = defaultdict(lambda: {"sum": 0, "cnt": 0, "price": None, "title_counts": Counter()})
        for pm in worker_partials:
            for bid, info in pm.items():
                b = books[bid]
                b["sum"] += int(info["sum"])
                b["cnt"] += int(info["cnt"])
                if b["price"] is None and info["price"] is not None:
                    b["price"] = float(info["price"])
                b["title_counts"].update(info.get("title_counts", {}))

        # Candidate filter: exact integer-safe check sum < 4 * cnt, and price known
        candidates = []
        for bid, info in books.items():
            c = info["cnt"]
            if c <= 0:
                continue
            if info["sum"] < 4 * c and info["price"] is not None:
                title = pick_title(info["title_counts"])
                price = float(info["price"])
                candidates.append((title, price))

        # Sort by price desc; tie-break by title asc; take top 10
        candidates.sort(key=lambda x: (-x[1], x[0]))
        top10 = candidates[:10]

        # Build final dict {Title: Price}
        final = {title: price for title, price in top10}

        chunk_sizes = [0] + [hi - lo for (lo, hi) in ranges]
        elapsed = time.time() - t0
        print({
            "final_answer": final,
            "chunkSizePerThread": chunk_sizes,
            "answerPerThread": [0] + per_rank_seen,   # diagnostic only
            "totalTimeTaken": elapsed
        })

    else:
        # Worker ranks: receive config + range, compute, send result
        try:
            DATA_PATH, DATASET_SIZE, lo, hi = comm.recv(source=0, tag=TAG_RANGE)
            sol = MPISolution(dataset_path=DATA_PATH, dataset_size=DATASET_SIZE)
            part_map, seen_books = sol._work(lo, hi)
            comm.send((part_map, int(seen_books)), dest=0, tag=TAG_RESULT)
        except Exception as e:
            comm.send(str(e), dest=0, tag=TAG_ERROR)

if __name__ == "__main__":
    main()