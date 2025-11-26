# t3_mater.py — Master-only coordinator for Q2/T3 (rank 0 coordinates; workers compute)
import os, time
from mpi4py import MPI
from dotenv import load_dotenv
from t3q2 import MPISolution  # uses the worker logic from t3q2.py

# Load shared config from the mounted path on every rank
load_dotenv("/workspace/.env")

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

TAG_RANGE = 201
TAG_RESULT = 202
TAG_ERROR = 203

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

        # Receive from workers
        worker_maps = []
        per_rank_hits = []
        worker_errors = []

        for r in range(1, size):
            status = MPI.Status()
            comm.probe(source=r, tag=MPI.ANY_TAG, status=status)
            if status.Get_tag() == TAG_ERROR:
                err = comm.recv(source=r, tag=TAG_ERROR)
                worker_errors.append(f"[rank {r}] {err}")
            else:
                part_map, local_hits = comm.recv(source=r, tag=TAG_RESULT)
                worker_maps.append(part_map)
                per_rank_hits.append(int(local_hits))

        if worker_errors:
            print({"final_answer": f"Worker errors: {worker_errors}",
                   "chunkSizePerThread": [], "answerPerThread": [], "totalTimeTaken": 0.0})
            return

        # Merge partials: BId -> [sum, cnt, price]
        global_map = {}
        for m in worker_maps:
            for bid, (s, c, p) in m.items():
                g = global_map.setdefault(bid, [0.0, 0, None])
                g[0] += float(s)
                g[1] += int(c)
                if g[2] is None and p is not None:
                    g[2] = float(p)

        # Final exact check: sum == 5 * count AND price == 2
        final = 0
        for bid, (s, c, price) in global_map.items():
            if c > 0 and (s == 5 * c) and (price == 2):
                final += 1

        # Master did no compute → first chunk is 0
        chunk_sizes = [0] + [hi - lo for (lo, hi) in ranges]
        elapsed = time.time() - t0
        print({
            "final_answer": final,
            "chunkSizePerThread": chunk_sizes,
            "answerPerThread": [0] + per_rank_hits,  # diagnostic only
            "totalTimeTaken": elapsed
        })

    else:
        # Worker ranks: receive config + range, compute locally, send result
        try:
            DATA_PATH, DATASET_SIZE, lo, hi = comm.recv(source=0, tag=TAG_RANGE)
            sol = MPISolution(dataset_path=DATA_PATH, dataset_size=DATASET_SIZE)
            part_map, local_hits = sol._process_slice(lo, hi)
            comm.send((part_map, int(local_hits)), dest=0, tag=TAG_RESULT)
        except Exception as e:
            # Signal failure but keep MPI clean
            comm.send(str(e), dest=0, tag=TAG_ERROR)

if __name__ == "__main__":
    main()