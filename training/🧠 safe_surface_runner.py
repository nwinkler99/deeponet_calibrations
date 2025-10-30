import os
import gc
import time
import pickle
import psutil
import numpy as np
from datetime import datetime
from generation.surface_generation import generate_surfaces, SimulationConfig

# ==========================================
# CONFIGURATION
# ==========================================

NUM_BATCHES = 100          # total number of batches to run
BATCH_SIZE = 50            # number of parameter sets per batch
SAVE_ROOT = "data/longrun" # root directory for all runs
SLEEP_ON_ERROR = 30        # seconds to wait after failure before retry
SEED_BASE = 4235           # deterministic seed offset per batch

os.makedirs(SAVE_ROOT, exist_ok=True)

cfg = SimulationConfig(M=50000, n=500, T_max=2.0, S0=1.0, G=2)

# ==========================================
# HELPER FUNCTIONS
# ==========================================

def memory_usage_gb():
    return psutil.virtual_memory().used / 1e9

def save_checkpoint(batch_idx, data):
    """Append or overwrite batch data safely."""
    out_path = os.path.join(SAVE_ROOT, f"batch_{batch_idx:04d}.pkl")
    tmp_path = out_path + ".tmp"
    with open(tmp_path, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp_path, out_path)
    print(f"💾 Saved batch {batch_idx} → {out_path}")

# ==========================================
# MAIN LOOP
# ==========================================

for batch_idx in range(NUM_BATCHES):
    print(f"\n🚀 Starting batch {batch_idx+1}/{NUM_BATCHES} | mem={memory_usage_gb():.2f} GB")

    try:
        # run generation for this batch
        batch_seed = SEED_BASE + batch_idx * 10000
        surfaces = generate_surfaces(
            num_sets=BATCH_SIZE,
            forward_curves_per_set=10,
            cfg=cfg,
            seed=batch_seed,
            randomize_grid=True,
        )

        # save and clean up
        save_checkpoint(batch_idx, surfaces)
        del surfaces
        gc.collect()
        print(f"✅ Batch {batch_idx+1} complete | mem={memory_usage_gb():.2f} GB")

    except Exception as e:
        print(f"⚠️ Error in batch {batch_idx}: {type(e).__name__} — {e}")
        print(f"🕒 Waiting {SLEEP_ON_ERROR}s before retry...")
        time.sleep(SLEEP_ON_ERROR)
        continue  # proceed to next batch

print("\n🎉 All batches completed successfully!")
