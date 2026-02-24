# Distributed Book Review Analytics with MPI & Docker

A **parallel computing** project that uses **MPI (Message Passing Interface)** with **mpi4py** to distribute large-scale book review dataset analysis across multiple Docker containers. The system implements a **master–worker architecture** where a coordinator (rank 0) splits the workload and worker containers process data chunks in parallel.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Queries Implemented](#queries-implemented)
- [Prerequisites](#prerequisites)
- [Setup & Installation](#setup--installation)
- [Usage](#usage)
  - [Task 1 — Single Run (1 Master + 3 Workers)](#task-1--single-run-1-master--3-workers)
  - [Task 2 — Scalability Analysis (4→10 Processes)](#task-2--scalability-analysis-410-processes)
- [Performance Results](#performance-results)
- [Environment Variables](#environment-variables)
- [How It Works](#how-it-works)
- [Tech Stack](#tech-stack)
- [License](#license)

---

## Overview

This project analyzes a large **book reviews CSV dataset** (~3 million rows) by distributing the computation across up to **10 Docker containers** communicating via MPI over SSH. Four analytical queries are implemented, each using a **map-reduce** style approach:

1. The dataset is **split into row ranges** and assigned to worker processes.
2. Each worker **reads only its assigned slice** from the CSV.
3. Workers compute **partial aggregates** and send results back to the master.
4. The master **merges** all partial results to produce the final answer.

---

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                    Docker Compose Cluster                 │
│                                                          │
│  ┌──────────────┐    SSH + MPI     ┌──────────────┐     │
│  │  mpi_master   │◄──────────────►│   mpi_w1      │     │
│  │  (rank 0)     │                 │  (rank 1)     │     │
│  │  Coordinator  │    SSH + MPI    ├──────────────┤     │
│  │               │◄──────────────►│   mpi_w2      │     │
│  │  - Splits     │                 │  (rank 2)     │     │
│  │    workload   │    SSH + MPI    ├──────────────┤     │
│  │  - Merges     │◄──────────────►│   mpi_w3      │     │
│  │    results    │                 │  (rank 3)     │     │
│  │  - Prints     │       ...       ├──────────────┤     │
│  │    final      │◄──────────────►│   ...         │     │
│  │    answer     │                 ├──────────────┤     │
│  │               │◄──────────────►│   mpi_w9      │     │
│  └──────────────┘                 │  (rank 9)     │     │
│                                    └──────────────┘     │
│                                                          │
│  Shared Volume: ./  ←→  /workspace (all containers)     │
└──────────────────────────────────────────────────────────┘
```

- **1 Master** (`mpi_master`) — coordinates work, does no computation
- **Up to 9 Workers** (`mpi_w1` – `mpi_w9`) — each processes a data chunk
- All containers share the project directory via a Docker volume at `/workspace`
- Containers communicate via **SSH** (pre-configured in the Docker image)

---

## Project Structure

```
docker-mpi/
│
├── docker-compose.yml        # Defines the MPI cluster (1 master + 9 workers)
├── hostfile                  # MPI hostfile listing all container hostnames
├── master_key.pub            # SSH public key for passwordless inter-container auth
│
├── t3.py                     # Q1 worker logic — all ranks participate (including master)
├── t3_master_only.py         # Q1 master-only coordinator — master delegates, workers compute
│
├── t3q2.py                   # Q2 worker logic — count books with avg RScore == 5 & price == 2
├── t3q2_master.py            # Q2 master-only coordinator
│
├── t3q3.py                   # Q3 worker logic — user(s) who reviewed most books with avg RScore == 4
├── t3q3_master.py            # Q3 master-only coordinator
│
├── t3q4.py                   # Q4 worker logic — top 10 highest-price books with avg RScore < 4
├── t3q4_master.py            # Q4 master-only coordinator
│
├── q1_times.csv              # Benchmark results: Q1 execution time vs. container count
├── q2_times.csv              # Benchmark results: Q2
├── q3_times.csv              # Benchmark results: Q3
├── q4_times.csv              # Benchmark results: Q4
│
├── help.txt                  # Step-by-step execution guide
└── .env                      # Environment variables (PATH_DATASET, DATASET_SIZE) — not committed
```

> **Note:** The book reviews CSV dataset (`book.csv`) is **not included** in this repository due to its large size. You must place it in the project root before running.

---

## Queries Implemented

| Query  | File(s)                       | Description                                                                                                                       |
| ------ | ----------------------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| **Q1** | `t3.py` / `t3_master_only.py` | Count books where **average RScore == 5** and **BPrice == 2** (all ranks participate / master-only mode)                          |
| **Q2** | `t3q2.py` / `t3q2_master.py`  | Count books where **average RScore == 5** and **BPrice == 2** (improved version with integer-safe comparison: `sum == 5 * count`) |
| **Q3** | `t3q3.py` / `t3q3_master.py`  | Find the **user name(s)** who reviewed the **largest number of distinct books** among users with an **exact average RScore == 4** |
| **Q4** | `t3q4.py` / `t3q4_master.py`  | Find the **top 10 highest-priced books** with an **average RScore < 4**, sorted by price descending (title ascending on tie)      |

Each query follows a **two-file pattern**:

- `t3qX.py` — contains the `MPISolution` class with `_process_slice()` or `_work()` for chunk processing
- `t3qX_master.py` — the master-only coordinator that distributes ranges, collects partial results, and merges them

---

## Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running
- The book reviews dataset CSV file (place it in the project root as configured in `.env`)
- ~4 GB of available RAM (for running up to 10 containers)

---

## Setup & Installation

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/docker-mpi.git
cd docker-mpi
```

### 2. Add the Dataset

Place your `book.csv` file (or whichever name your dataset uses) in the project root directory.

### 3. Create the `.env` File

```bash
cat > .env << 'EOF'
PATH_DATASET=/workspace/book.csv
DATASET_SIZE=3000000
EOF
```

Adjust `DATASET_SIZE` to match the actual number of data rows in your CSV (excluding the header).

### 4. Start the Docker Cluster

```bash
docker compose up -d
```

This launches **10 containers**: 1 master + 9 workers, all running SSH and sharing the project via `/workspace`.

### 5. Enter the Master Container

```bash
docker exec -it mpi_master bash
cd /workspace
```

---

## Usage

### Task 1 — Single Run (1 Master + 3 Workers)

Run the master-only program with 4 MPI processes (1 master + 3 workers):

```bash
mpirun -n 4 -f /workspace/hostfile python /workspace/t3_master_only.py
```

**Expected output:**

```python
{
  'final_answer': 1192657,
  'chunkSizePerThread': [0, 1000000, 1000000, 1000000],
  'answerPerThread': [0, 395814, 397612, 399231],
  'totalTimeTaken': 17.78
}
```

### Task 2 — Scalability Analysis (4→10 Processes)

Run each query with increasing numbers of MPI processes (4 through 10) and record the execution time:

**Q1 — `t3_master_only.py`**

```bash
echo "containers,seconds" > q1_times.csv
for n in 4 5 6 7 8 9 10; do
  start=$(date +%s)
  mpirun -n $n -f /workspace/hostfile python /workspace/t3_master_only.py > tmp_out.txt 2>&1
  end=$(date +%s)
  elapsed=$((end - start))
  echo "$n,$elapsed" | tee -a q1_times.csv
done
```

**Q2 — `t3q2_master.py`**

```bash
echo "containers,seconds" > q2_times.csv
for n in 4 5 6 7 8 9 10; do
  start=$(date +%s)
  mpirun -n $n -f /workspace/hostfile python /workspace/t3q2_master.py > tmp_out.txt 2>&1
  end=$(date +%s)
  elapsed=$((end - start))
  echo "$n,$elapsed" | tee -a q2_times.csv
done
```

**Q3 — `t3q3_master.py`**

```bash
echo "containers,seconds" > q3_times.csv
for n in 4 5 6 7 8 9 10; do
  start=$(date +%s)
  mpirun -n $n -f /workspace/hostfile python /workspace/t3q3_master.py > tmp_out.txt 2>&1
  end=$(date +%s)
  elapsed=$((end - start))
  echo "$n,$elapsed" | tee -a q3_times.csv
done
```

**Q4 — `t3q4_master.py`**

```bash
echo "containers,seconds" > q4_times.csv
for n in 4 5 6 7 8 9 10; do
  start=$(date +%s)
  mpirun -n $n -f /workspace/hostfile python /workspace/t3q4_master.py > tmp_out.txt 2>&1
  end=$(date +%s)
  elapsed=$((end - start))
  echo "$n,$elapsed" | tee -a q4_times.csv
done
```

---

## Performance Results

Execution times (in seconds) recorded with **3 million rows** on a local Docker cluster:

### Q1 — Count books (avg RScore == 5, price == 2) — All-ranks mode

| Processes | Time (s) |
| --------- | -------- |
| 4         | 21       |
| 5         | 24       |
| 6         | 23       |
| 7         | 22       |
| 8         | 23       |
| 9         | 23       |
| 10        | 23       |

### Q2 — Count books (avg RScore == 5, price == 2) — Master-only mode

| Processes | Time (s) |
| --------- | -------- |
| 4         | 26       |
| 5         | 25       |
| 6         | 24       |
| 7         | 25       |
| 8         | 22       |
| 9         | 23       |
| 10        | 22       |

### Q3 — User(s) with most books reviewed (avg RScore == 4)

| Processes | Time (s) |
| --------- | -------- |
| 4         | 182      |
| 5         | 146      |
| 6         | 124      |
| 7         | 127      |
| 8         | 119      |
| 9         | 117      |
| 10        | 119      |

### Q4 — Top 10 highest-price books (avg RScore < 4)

| Processes | Time (s) |
| --------- | -------- |
| 4         | 148      |
| 5         | 113      |
| 6         | 94       |
| 7         | 98       |
| 8         | 88       |
| 9         | 100      |
| 10        | 79       |

**Key Observations:**

- **Q1/Q2** (vectorized Pandas aggregations) are I/O-bound — adding workers beyond 4 provides minimal speedup since CSV reading is the bottleneck.
- **Q3/Q4** (row-level iteration) show clear **near-linear speedup** from 4→10 processes, with Q4 improving from 148s to 79s (~1.87× speedup).
- Diminishing returns appear around 8–9 processes due to inter-container communication overhead on a single host.

---

## Environment Variables

| Variable       | Description                                        | Example               |
| -------------- | -------------------------------------------------- | --------------------- |
| `PATH_DATASET` | Absolute path to the CSV file inside the container | `/workspace/book.csv` |
| `DATASET_SIZE` | Number of data rows in the CSV (excluding header)  | `3000000`             |

These are defined in a `.env` file at the project root, loaded by `python-dotenv` in each script.

---

## How It Works

### Master–Worker Communication Pattern

```
Master (rank 0)                    Workers (rank 1..N-1)
──────────────                     ─────────────────────
1. Read DATASET_SIZE
2. Compute row ranges
3. Send (lo, hi) to workers  ───►  4. Receive (lo, hi)
                                    5. Read CSV slice [lo, hi)
                                    6. Compute partial aggregates
7. Receive partial maps     ◄───   8. Send (partial_map, diag_count)
9. Merge all partials
10. Compute final answer
11. Print result dict
```

### Data Flow Per Worker

1. **Read slice** — `pd.read_csv(skiprows=..., nrows=...)` reads only the assigned chunk
2. **Aggregate** — Group by `BId` or `UId` to compute sums, counts, prices, and distinct sets
3. **Serialize** — Convert results to a pickle-friendly dictionary
4. **Send** — MPI point-to-point `comm.send()` back to master

### Output Format

Each script prints a Python dictionary:

```python
{
    "final_answer": <result>,            # The computed answer
    "chunkSizePerThread": [0, ...],      # Rows processed per rank (0 = master)
    "answerPerThread": [0, ...],         # Per-worker diagnostic counts
    "totalTimeTaken": <float>            # Wall-clock time in seconds
}
```

---

## Tech Stack

| Component                     | Technology                                                     |
| ----------------------------- | -------------------------------------------------------------- |
| Parallel Framework            | [mpi4py](https://mpi4py.readthedocs.io/) (MPI for Python)      |
| Container Orchestration       | [Docker Compose](https://docs.docker.com/compose/)             |
| Base Image                    | `ozxx33/mpi4py-cluster-base` (Ubuntu + OpenMPI + mpi4py + SSH) |
| Data Processing               | [Pandas](https://pandas.pydata.org/)                           |
| Inter-container Communication | SSH (passwordless via shared keys)                             |
| Configuration                 | [python-dotenv](https://pypi.org/project/python-dotenv/)       |

---

## Stopping the Cluster

```bash
# Exit the master container
exit

# Stop and remove all containers
docker compose down
```

---

## License

This project is for educational/academic purposes.
