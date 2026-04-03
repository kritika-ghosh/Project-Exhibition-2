# executer.py

import pandas as pd
import csv
import json
import os
import time
from integrated1_original import process_sample

# =========================
# CONFIG
# =========================

DATA_PATH = "D:/Desktop/Smart digital archivist/dump/Meta-data/Data.xlsx"
RESULTS_FILE = "results_knn.csv"
CHECKPOINT_FILE = "checkpoint_knn.json"

TEXT_COL = "Abstract"
LABEL_COL = "Domain"

EXPERIMENTS = [
    {"name": "flat", "hier": False, "ssp": False, "dlts": False},
    {"name": "hier", "hier": True, "ssp": False, "dlts": False},
    {"name": "hier+dlts", "hier": True, "ssp": False, "dlts": True},
    {"name": "hier+dlts+ssp", "hier": True, "ssp": True, "dlts": True}
]

# =========================
# LOAD DATA
# =========================

df = pd.read_excel(DATA_PATH)
TOTAL_ROWS = len(df)

# =========================
# CHECKPOINT SYSTEM (SAFE)
# =========================

def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        try:
            with open(CHECKPOINT_FILE, "r") as f:
                data = f.read().strip()
                if not data:
                    raise ValueError
                return json.loads(data)
        except:
            print("⚠️ Corrupted checkpoint. Resetting...")
    return {"exp_idx": 0, "row_idx": 0}

def save_checkpoint(exp_idx, row_idx):
    temp_file = CHECKPOINT_FILE + ".tmp"

    with open(temp_file, "w") as f:
        json.dump({
            "exp_idx": exp_idx,
            "row_idx": row_idx
        }, f)

    os.replace(temp_file, CHECKPOINT_FILE)

# =========================
# CSV LOGGING (SAFE)
# =========================

def log_result(result):
    write_header = not os.path.exists(RESULTS_FILE)

    with open(RESULTS_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=result.keys())


        if write_header:
            writer.writeheader()

        writer.writerow(result)

# =========================
# STOP CONTROL
# =========================

def should_stop():
    return os.path.exists("STOP")

# =========================
# MAIN LOOP
# =========================

checkpoint = load_checkpoint()
start_exp = checkpoint["exp_idx"]
start_row = checkpoint["row_idx"]

print(f"Resuming from Experiment {start_exp}, Row {start_row}")

global_start = time.time()

try:
    for exp_idx in range(start_exp, len(EXPERIMENTS)):

        exp = EXPERIMENTS[exp_idx]
        print(f"\n🚀 Running Experiment: {exp['name']}")

        row_start = start_row if exp_idx == start_exp else 0

        # ---- Progress Tracking ----
        start_time = time.time()
        processed = 0
        total = TOTAL_ROWS - row_start

        for i, row in df.iloc[row_start:].iterrows():

            text = str(row[TEXT_COL])
            label = str(row[LABEL_COL]).strip().title()

            result = process_sample(
                text=text,
                true_label=label,
                use_hierarchy=exp["hier"],
                use_ssp=exp["ssp"],
                use_dlts=exp["dlts"]
            )

            result["experiment"] = exp["name"]
            result["row_index"] = i

            log_result(result)

            save_checkpoint(exp_idx, i)

            processed += 1

            # ---- PROGRESS + ETA ----
            if processed % 50 == 0:

                elapsed = time.time() - start_time
                speed = processed / elapsed if elapsed > 0 else 0

                remaining = total - processed
                eta_sec = remaining / speed if speed > 0 else 0

                # format ETA
                if eta_sec > 3600:
                    eta_str = f"{eta_sec/3600:.2f} hr"
                else:
                    eta_str = f"{eta_sec/60:.1f} min"

                print(
                    f"[{exp['name']}] "
                    f"{processed}/{total} "
                    f"({(processed/total)*100:.2f}%) | "
                    f"{speed:.2f} it/s | ETA: {eta_str}"
                )

            # ---- STOP ----
            if should_stop():
                print("\n🛑 STOP file detected. Exiting safely.")
                raise KeyboardInterrupt

        start_row = 0

except KeyboardInterrupt:
    print("\n⏹ Execution stopped safely. Resume anytime.")

# =========================
# FINAL TIME
# =========================

total_time = (time.time() - global_start) / 60
print(f"\n✅ Done. Total runtime: {total_time:.2f} minutes")
