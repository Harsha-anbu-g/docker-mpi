# t3_master_only.py — master-only coordinator for Q2/T3
import os, time
from mpi4py import MPI
from dotenv import load_dotenv
from t3 import MPISolution  # uses your class with _process_slice()

load_dotenv("/workspace/.env")
DATA_PATH = os.getenv("PATH_DATASET")
DATASET_SIZE = int(os.getenv("DATASET_SIZE", "3000000"))

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

TAG_RANGE  = 11
TAG_RESULT = 22

def main():
    t0 = time.time()

    if not DATA_PATH or DATASET_SIZE <= 0:
        if rank == 0:
            print({"final_answer":"Dataset path/size missing",
                   "chunkSizePerThread":[], "answerPerThread":[], "totalTimeTaken":0.0})
        return

    workers = size - 1
    if workers < 1:
        if rank == 0:
            print({"final_answer":"Need at least 1 worker (-n >= 2)",
                   "chunkSizePerThread":[], "answerPerThread":[], "totalTimeTaken":0.0})
        return

    sol = MPISolution(dataset_path=DATA_PATH, dataset_size=DATASET_SIZE)

    try:
        if rank == 0:
            # split rows across WORKERS (master does no compute)
            base = DATASET_SIZE // workers
            ranges = []
            for w in range(workers):
                lo = w * base
                hi = (w + 1) * base if w < workers - 1 else DATASET_SIZE
                ranges.append((lo, hi))

            # send ranges
            for r in range(1, size):
                comm.send(ranges[r - 1], dest=r, tag=TAG_RANGE)

            # gather partial maps (and optional local diagnostic counts)
            partial_maps = []
            local_counts = []
            for r in range(1, size):
                part_map, local_hits = comm.recv(source=r, tag=TAG_RESULT)
                partial_maps.append(part_map)
                local_counts.append(int(local_hits))

            # merge maps: BId -> [sum, cnt, price]
            global_map = {}
            for m in partial_maps:
                for bid, (s, c, p) in m.items():
                    if bid not in global_map:
                        global_map[bid] = [0.0, 0, None]
                    global_map[bid][0] += s
                    global_map[bid][1] += c
                    if global_map[bid][2] is None and p is not None:
                        global_map[bid][2] = p

            # compute final answer on merged data
            final = 0
            for bid, (s, c, price) in global_map.items():
                if c > 0:
                    avg = s / c
                    if avg == 5 and price == 2:
                        final += 1

            elapsed = time.time() - t0
            chunk_sizes = [0] + [hi - lo for (lo, hi) in ranges]   # master did 0 work
            answers = [0] + local_counts                           # diagnostics only

            print({
                "final_answer": final,
                "chunkSizePerThread": chunk_sizes,
                "answerPerThread": answers,
                "totalTimeTaken": elapsed
            })

        else:
            # worker: receive range, compute partials using your _process_slice()
            lo, hi = comm.recv(source=0, tag=TAG_RANGE)
            part_map, local_hits = sol._process_slice(lo, hi)
            comm.send((part_map, int(local_hits)), dest=0, tag=TAG_RESULT)

    except Exception as err:
        if rank == 0:
            print({"final_answer": str(err),
                   "chunkSizePerThread":[], "answerPerThread":[], "totalTimeTaken":0.0})

if __name__ == "__main__":
    main()