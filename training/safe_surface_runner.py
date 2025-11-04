import sys, os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
import gc
import time
import pickle
import psutil
import numpy as np
import multiprocessing
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from generation.surface_generation import generate_surfaces, SimulationConfig

# ==========================================
# CONFIGURATION
# ==========================================
randomize_grid = False        # whether to randomize time grid per param set
NUM_BATCHES = 100                # total number of batches to run
BATCH_SIZE = 60     
if randomize_grid:             # number of parameter sets per batch
    SAVE_ROOT = "data/longrun" # root directory for all runs
else:
    SAVE_ROOT = "data/fixed_longrun"  # root directory for all runs
SLEEP_ON_ERROR = 30              # seconds to wait after failure before retry
SEED_BASE = 123                  # deterministic seed offset per batch
MAX_WORKERS = 6                  # number of CPU cores to use
CHUNK_SIZE = 10                  # number of parameter sets per worker

os.makedirs(SAVE_ROOT, exist_ok=True)

cfg = SimulationConfig(M=50000, n=int(2.1*252), T_max=2.1, S0=1.0, G=1, dtype=np.float32)

# ==========================================
# HELPER FUNCTIONS
# ==========================================

def memory_usage_gb():
    """Return current memory usage in GB."""
    return psutil.virtual_memory().used / 1e9


def save_checkpoint(batch_idx, data, randomized=randomize_grid):
    """Safely save batch results to disk."""
    if randomized:
        out_path = os.path.join(SAVE_ROOT, f"batch_{batch_idx:04d}.pkl")
    else:
        out_path = os.path.join(SAVE_ROOT, f"fixed_batch_{batch_idx:04d}.pkl")
    tmp_path = out_path + ".tmp"
    with open(tmp_path, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp_path, out_path)
    print(f" Saved batch {batch_idx} to {out_path}")


# ==========================================
# WORKER FUNCTION (must be top-level!)
# ==========================================

def worker_generate_surfaces(chunk_seeds,
                             forward_curves_per_set,
                             cfg,
                             randomize_grid,
                             grid_jitter,
                             save_every):
    """
    Top-level worker for multiprocessing.
    Each worker handles one chunk of seeds, processed in a single call to generate_surfaces().
    """
    import os
    from generation.surface_generation import generate_surfaces

    chunk_seed = chunk_seeds[0]
    print(f"[PID {os.getpid()}] Generating {len(chunk_seeds)} param sets (seed={chunk_seed})")

    try:
        res = generate_surfaces(
            num_sets=len(chunk_seeds),
            forward_curves_per_set=forward_curves_per_set,
            cfg=cfg,
            seed=chunk_seed,
            randomize_grid=randomize_grid,
            grid_jitter=grid_jitter,
            save_every=save_every,
        )
        return res

    except Exception as e:
        print(f"[PID {os.getpid()}]  Worker failed with {type(e).__name__}: {e}")
        return []


# ==========================================
# PARALLEL WRAPPER
# ==========================================

def generate_surfaces_parallel(num_sets=1,
                               forward_curves_per_set=1,
                               cfg=None,
                               seed=42,
                               randomize_grid=randomize_grid,
                               grid_jitter=0.25,
                               save_every=200,
                               max_workers=None,
                               chunk_size=None):
    """
    Parallel wrapper around generate_surfaces().

    Splits num_sets across multiple workers. Each worker processes `chunk_size`
    parameter sets sequentially inside one generate_surfaces() call.
    """
    if cfg is None:
        raise ValueError("cfg (SimulationConfig) must be provided")

    cpu_count = multiprocessing.cpu_count()
    max_workers = max_workers or min(cpu_count, num_sets)
    chunk_size = chunk_size or int(np.ceil(num_sets / max_workers))

    print(f"Using {max_workers} CPU workers | chunk_size={chunk_size}")

    # Create deterministic, non-overlapping seeds for each param set
    seeds = [seed + 10000 * i for i in range(num_sets)]
    chunks = [seeds[i:i + chunk_size] for i in range(0, len(seeds), chunk_size)]

    all_surfaces = []
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = [
            ex.submit(
                worker_generate_surfaces,
                chunk,
                forward_curves_per_set,
                cfg,
                randomize_grid,
                grid_jitter,
                save_every,
            )
            for chunk in chunks
        ]

        for fut in as_completed(futures):
            try:
                all_surfaces.extend(fut.result())
            except Exception as e:
                print(f" Worker failed: {type(e).__name__} — {e}")

    print(f" Completed parallel generation: {len(all_surfaces)} surfaces total")
    return all_surfaces


# ==========================================
# MAIN LOOP
# ==========================================

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    START_BATCH = 0
    print(" Starting long-run data generation...\n")

    for batch_idx in range(START_BATCH, NUM_BATCHES):
        print(f"\n--- Batch {batch_idx+1}/{NUM_BATCHES} | Mem={memory_usage_gb():.2f} GB ---")

        try:
            t0 = time.time()
            batch_seed = SEED_BASE + batch_idx * 10000

            surfaces = generate_surfaces_parallel(
                num_sets=BATCH_SIZE,
                forward_curves_per_set=10,
                cfg=cfg,
                seed=batch_seed,
                randomize_grid=False,
                max_workers=MAX_WORKERS,
                chunk_size=CHUNK_SIZE,
            )

            save_checkpoint(batch_idx, surfaces)
            del surfaces
            gc.collect()

            t1 = time.time()
            print(f" Batch {batch_idx+1} complete | Mem={memory_usage_gb():.2f} GB | "
                  f"Time={(t1 - t0)/60:.1f} min")

        except Exception as e:
            print(f" Error in batch {batch_idx}: {type(e).__name__} — {e}")
            print(f" Waiting {SLEEP_ON_ERROR}s before retry...")
            time.sleep(SLEEP_ON_ERROR)
            continue

    print("\n All batches completed successfully!")
