"""
inspect_hf_datasets.py
======================
Run locally BEFORE the Modal job.
Prints column names, domain values, and sample counts for each HF dataset.
Tells you exactly what to put in HF_CONFIGS in the Modal script.

Usage:
    python inspect_hf_datasets.py

Requirements:
    pip install datasets huggingface_hub
"""

from datasets import load_dataset

# ── datasets to inspect ───────────────────────────────────────────
TARGETS = [
    {
        "name":    "PACS",
        "hf_id":   "flwrlabs/pacs",
        "split":   "train",
    },
    {
        "name":    "DomainNet",
        "hf_id":   "wltjr1007/DomainNet",
        "split":   "train",
    },
    {
        "name":    "Camelyon17",
        "hf_id":   "wltjr1007/Camelyon17-WILDS",
        "split":   "train",
    },
]


def inspect(cfg: dict):
    print(f"\n{'='*60}")
    print(f"  {cfg['name']}  ({cfg['hf_id']})")
    print(f"{'='*60}")

    # Load first 5 rows only — fast, no full download
    ds = load_dataset(cfg["hf_id"], split=cfg["split"], streaming=True)
    first = list(ds.take(5))

    # ── column names ──────────────────────────────────────────
    cols = list(first[0].keys())
    print(f"\nColumns:  {cols}")

    # ── sample row (without image bytes) ─────────────────────
    sample = {k: v for k, v in first[0].items() if k != "image"}
    print(f"Sample:   {sample}")

    # ── now load full split to count domains ──────────────────
    print(f"\nLoading full split to count domains...")
    ds_full = load_dataset(cfg["hf_id"], split=cfg["split"])
    print(f"Total rows: {len(ds_full)}")

    # Find which columns look like domain identifiers
    # (string columns with low cardinality)
    for col in cols:
        if col == "image":
            continue
        vals = ds_full[col]
        if isinstance(vals[0], str):
            unique = sorted(set(vals))
            if len(unique) <= 50:  # plausible domain column
                print(f"\nColumn '{col}'  ({len(unique)} unique values):")
                for v in unique:
                    count = sum(1 for x in vals if x == v)
                    print(f"  {v:<30}  {count:>6} rows")
        elif isinstance(vals[0], int):
            unique = sorted(set(vals))
            if len(unique) <= 50:
                print(f"\nColumn '{col}'  ({len(unique)} unique int values):")
                for v in unique:
                    count = sum(1 for x in vals if x == v)
                    print(f"  {v:<10}  {count:>6} rows")


if __name__ == "__main__":
    for cfg in TARGETS:
        try:
            inspect(cfg)
        except Exception as e:
            print(f"\n  ERROR loading {cfg['name']}: {e}")

    print("\n\nDone. Copy the column names into HF_CONFIGS in credal_dg_modal.py")