# t3q3_master.py — Master-only coordinator for Q3/T3
# Rank 0 coordinates; ranks 1..N-1 run MPISolution._work(lo, hi) and return partials.
import os, time
from collections import Counter, defaultdict
from mpi4py import MPI
from dotenv import load_dotenv
from t3q3 import MPISolution  # worker logic

# Load shared config from mounted path (robust to CWD)
load_dotenv("/workspace/.env")

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

TAG_RANGE = 301
TAG_RESULT = 302
TAG_ERROR = 303

def pick_display_name(uid: str, counts: dict) -> str:
    """Most frequent name; on ties choose lexicographically smallest. Fallback to uid."""
    if not counts:
        return str(uid)
    c = Counter(counts)
    top = max(c.values())
    candidates = sorted([n for n, v in c.items() if v == top])
    return candidates[0]

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

        # Dispatch work to workers
        for r in range(1, size):
            lo, hi = ranges[r - 1]
            comm.send((DATA_PATH, DATASET_SIZE, lo, hi), dest=r, tag=TAG_RANGE)

        # Gather results (or errors) from workers
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
                part_map, users_seen = comm.recv(source=r, tag=TAG_RESULT)
                worker_partials.append(part_map)
                per_rank_seen.append(int(users_seen))

        if worker_errors:
            print({"final_answer": f"Worker errors: {worker_errors}",
                   "chunkSizePerThread": [], "answerPerThread": [], "totalTimeTaken": 0.0})
            return

        # Merge partial user maps
        users = defaultdict(lambda: {"sum": 0, "cnt": 0, "books": set(), "name_counts": Counter()})
        for m in worker_partials:
            for uid, info in m.items():
                u = users[uid]
                u["sum"] += int(info["sum"])
                u["cnt"] += int(info["cnt"])
                u["books"].update(info.get("books", []))
                u["name_counts"].update(info.get("name_counts", {}))

        # Determine stable display names
        name_of = {uid: pick_display_name(uid, dict(u["name_counts"])) for uid, u in users.items()}

        # Eligible users: exact average == 4  -> (sum == 4 * cnt). Score = number of distinct books.
        eligible_counts = {}
        for uid, u in users.items():
            if u["cnt"] and (u["sum"] == 4 * u["cnt"]):
                eligible_counts[uid] = len(u["books"])

        if not eligible_counts:
            final = ""
        else:
            best = max(eligible_counts.values())
            winners = [uid for uid, nbooks in eligible_counts.items() if nbooks == best]
            final = ", ".join(sorted({name_of[uid] for uid in winners}))

        chunk_sizes = [0] + [hi - lo for (lo, hi) in ranges]
        elapsed = time.time() - t0
        print({
            "final_answer": final,
            "chunkSizePerThread": chunk_sizes,
            "answerPerThread": [0] + per_rank_seen,   # diagnostic: users seen per worker
            "totalTimeTaken": elapsed
        })

    else:
        # Worker ranks: receive config + range, compute, send result
        try:
            DATA_PATH, DATASET_SIZE, lo, hi = comm.recv(source=0, tag=TAG_RANGE)
            sol = MPISolution(dataset_path=DATA_PATH, dataset_size=DATASET_SIZE)
            part_map, users_seen = sol._work(lo, hi)
            comm.send((part_map, int(users_seen)), dest=0, tag=TAG_RESULT)
        except Exception as e:
            comm.send(str(e), dest=0, tag=TAG_ERROR)

if __name__ == "__main__":
    main()