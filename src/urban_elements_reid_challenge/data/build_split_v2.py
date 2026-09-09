"""
Build UAM-as-validation split v2 (corrected).

UAM is already pre-split by the dataset authors with DISJOINT identity sets:
  - train_classes.csv : IDs 1–479,   cameras c001/c002/c003/c004  → training
  - query_classes.csv : IDs 480–691, camera  c004 only            → val queries
  - test_classes.csv  : IDs 480–691, cameras c001/c002/c003       → val gallery

No artificial holdout needed. Rivas training data (c001/c002/c003) goes entirely
into the training pool.

Outputs to /kaggle/working/splits_v2/:
  train_list.csv    — UAM train_classes + all Rivas
  val_query.csv     — UAM query_classes (c004)
  val_gallery.csv   — UAM test_classes  (c001/c002/c003, includes 59 distractor IDs)
  split_metadata.json
"""

import json
import hashlib
import pandas as pd
from pathlib import Path

# ── paths ─────────────────────────────────────────────────────────────────────
UAM_DIR   = "/kaggle/input/datasets/mahibulhaque/uam-dataset/UAM_Unified"
RIVAS_CSV = "/kaggle/input/datasets/mahibulhaque/urban2026/Urban2026/train_classes.csv"
OUT_DIR   = Path("/kaggle/working/splits_v2")

UAM_TRAIN_CSV = f"{UAM_DIR}/train_classes.csv"
UAM_QUERY_CSV = f"{UAM_DIR}/query_classes.csv"
UAM_TEST_CSV  = f"{UAM_DIR}/test_classes.csv"


# ── helpers ───────────────────────────────────────────────────────────────────
def sha256_of_filenames(df):
    keys = sorted(f"{s}:{n}" for s, n in zip(df["source_dataset"], df["filename"]))
    return hashlib.sha256(",".join(keys).encode()).hexdigest()


def load_uam_csv(path, img_subdir):
    df = pd.read_csv(path)
    df = df.rename(columns={"imageName": "filename", "objectID": "identity_id", "Class": "class"})
    df["camera_id"]      = df["cameraID"].str.extract(r"c0*(\d+)").astype(int)
    df["source_dataset"] = "uam"
    df["img_subdir"]     = img_subdir   # image_train / image_query / image_test
    return df[["filename", "identity_id", "class", "camera_id", "source_dataset", "img_subdir"]]


def load_rivas(path):
    df = pd.read_csv(path)
    df = df.rename(columns={"imageName": "filename", "Corresponding Indexes": "identity_id", "Class": "class"})
    df["camera_id"]      = df["cameraID"].str.extract(r"c0*(\d+)").astype(int)
    df["source_dataset"] = "rivas"
    df["img_subdir"]     = "image_train"
    return df[["filename", "identity_id", "class", "camera_id", "source_dataset", "img_subdir"]]


# ── load ──────────────────────────────────────────────────────────────────────
print("Loading data …")
uam_train   = load_uam_csv(UAM_TRAIN_CSV, "image_train")
uam_query   = load_uam_csv(UAM_QUERY_CSV, "image_query")
uam_gallery = load_uam_csv(UAM_TEST_CSV,  "image_test")
rivas       = load_rivas(RIVAS_CSV)

print(f"  UAM train  : {len(uam_train):,} rows  {uam_train['identity_id'].nunique()} ids  cams={sorted(uam_train['camera_id'].unique())}")
print(f"  UAM query  : {len(uam_query):,} rows  {uam_query['identity_id'].nunique()} ids  cams={sorted(uam_query['camera_id'].unique())}")
print(f"  UAM gallery: {len(uam_gallery):,} rows  {uam_gallery['identity_id'].nunique()} ids  cams={sorted(uam_gallery['camera_id'].unique())}")
print(f"  Rivas      : {len(rivas):,} rows  {rivas['identity_id'].nunique()} ids")


# ── identity disjointness ─────────────────────────────────────────────────────
train_ids   = set(uam_train["identity_id"])
val_ids     = set(uam_query["identity_id"]) | set(uam_gallery["identity_id"])
rivas_ids   = set(rivas["identity_id"])

assert train_ids.isdisjoint(val_ids),  "FATAL: UAM train/val identity overlap"
print(f"\n✓ UAM train IDs ({len(train_ids)}) and val IDs ({len(val_ids)}) are disjoint")

# Rivas IDs numerically overlap with UAM (both use small integers) — handled in dataset class
rivas_uam_overlap = rivas_ids & train_ids
print(f"  Rivas ∩ UAM-train numeric IDs: {len(rivas_uam_overlap)} (expected — different datasets, disambiguated by source_dataset column)")


# ── build final splits ────────────────────────────────────────────────────────
# train = ALL UAM train (IDs 1-479, all cameras) + ALL Rivas
train_list  = pd.concat([uam_train, rivas], ignore_index=True)
val_query   = uam_query.copy()
val_gallery = uam_gallery.copy()

# identities in query that have no gallery match (shouldn't happen, but check)
q_ids = set(val_query["identity_id"])
g_ids = set(val_gallery["identity_id"])
unmatched = q_ids - g_ids
if unmatched:
    print(f"WARNING: {len(unmatched)} query IDs have no gallery match — dropping them")
    val_query = val_query[val_query["identity_id"].isin(g_ids)]

distractor_ids = g_ids - q_ids
print(f"\n✓ Val query IDs: {len(q_ids)}  gallery IDs: {len(g_ids)}  distractor IDs (gallery-only): {len(distractor_ids)}")

print(f"\nSplit sizes:")
print(f"  train_list  : {len(train_list):,} rows")
print(f"  val_query   : {len(val_query):,} rows  ({val_query['identity_id'].nunique()} identities)")
print(f"  val_gallery : {len(val_gallery):,} rows  ({val_gallery['identity_id'].nunique()} identities)")

print(f"\nVal query per class:")
for cls, grp in val_query.groupby("class"):
    print(f"  {cls:<14}: {grp['identity_id'].nunique()} ids, {len(grp)} images")
print(f"Val gallery per class:")
for cls, grp in val_gallery.groupby("class"):
    print(f"  {cls:<14}: {grp['identity_id'].nunique()} ids, {len(grp)} images")


# ── sanity assertions ─────────────────────────────────────────────────────────
print("\nRunning sanity assertions …")

def keyset(df):
    return set(zip(df["source_dataset"], df["img_subdir"], df["filename"]))

train_keys = keyset(train_list)
vq_keys    = keyset(val_query)
vg_keys    = keyset(val_gallery)

assert train_keys.isdisjoint(vq_keys), "FAIL: train ∩ val_query"
assert train_keys.isdisjoint(vg_keys), "FAIL: train ∩ val_gallery"
assert vq_keys.isdisjoint(vg_keys),    "FAIL: val_query ∩ val_gallery"
print("  ✓ no overlap between splits")

uam_val_ids = set(val_query["identity_id"]) | set(val_gallery["identity_id"])
uam_train_ids_in_list = set(train_list[train_list["source_dataset"] == "uam"]["identity_id"])
assert uam_val_ids.isdisjoint(uam_train_ids_in_list), "FAIL: val identity in train"
print("  ✓ val identities disjoint from UAM train identities")

for qid in val_query["identity_id"].unique():
    assert qid in g_ids, f"FAIL: query id {qid} has no gallery"
print("  ✓ every val_query identity has gallery images")

assert (val_query["camera_id"] == 4).all(), "FAIL: val_query has non-c4 images"
print("  ✓ all val_query images are camera 4")

assert val_gallery["camera_id"].isin([1, 2, 3]).all(), "FAIL: val_gallery has c4 images"
print("  ✓ all val_gallery images are cameras 1/2/3")

# coverage
uam_in_train   = len(train_list[train_list["source_dataset"] == "uam"])
rivas_in_train = len(train_list[train_list["source_dataset"] == "rivas"])
assert uam_in_train   == len(uam_train), f"FAIL: UAM train coverage {uam_in_train} != {len(uam_train)}"
assert rivas_in_train == len(rivas),     f"FAIL: Rivas coverage {rivas_in_train} != {len(rivas)}"
print(f"  ✓ UAM train coverage: {uam_in_train}")
print(f"  ✓ Rivas coverage: {rivas_in_train}")

print("All assertions passed.\n")


# ── write artifacts ───────────────────────────────────────────────────────────
OUT_DIR.mkdir(parents=True, exist_ok=True)

# drop img_subdir from output (SplitV2ReID will derive paths from source+subdir)
train_list.to_csv(OUT_DIR / "train_list.csv",   index=False)
val_query.to_csv(OUT_DIR / "val_query.csv",     index=False)
val_gallery.to_csv(OUT_DIR / "val_gallery.csv", index=False)

per_class_counts = {}
for cls in sorted(set(val_query["class"])):
    vq_cls = val_query[val_query["class"] == cls]
    vg_cls = val_gallery[val_gallery["class"] == cls]
    tr_cls = train_list[(train_list["class"] == cls) & (train_list["source_dataset"] == "uam")]
    per_class_counts[cls] = {
        "val_query_ids":    int(vq_cls["identity_id"].nunique()),
        "val_gallery_ids":  int(vg_cls["identity_id"].nunique()),
        "val_queries":      int(len(vq_cls)),
        "val_gallery_imgs": int(len(vg_cls)),
        "train_images_uam": int(len(tr_cls)),
    }

metadata = {
    "seed": "N/A — using UAM pre-made split (no random holdout)",
    "method": "UAM official query/test split as val; UAM train + Rivas fully in training",
    "counts": {
        "train":                    int(len(train_list)),
        "val_query":                int(len(val_query)),
        "val_gallery":              int(len(val_gallery)),
        "val_query_identities":     int(val_query["identity_id"].nunique()),
        "val_gallery_identities":   int(val_gallery["identity_id"].nunique()),
        "val_distractor_identities": int(len(distractor_ids)),
    },
    "per_class_counts": per_class_counts,
    "per_source_counts": {
        "uam_train":       int(len(uam_train)),
        "rivas_train":     int(len(rivas)),
        "uam_val_query":   int(len(val_query)),
        "uam_val_gallery": int(len(val_gallery)),
    },
    "camera_id_rule": "c001=1, c002=2, c003=3, c004=4",
    "image_dirs": {
        "uam_train":   "image_train",
        "uam_query":   "image_query",
        "uam_gallery": "image_test",
        "rivas_train": "image_train",
    },
    "train_filename_hash":       sha256_of_filenames(train_list),
    "val_query_filename_hash":   sha256_of_filenames(val_query),
    "val_gallery_filename_hash": sha256_of_filenames(val_gallery),
    "known_limitations": (
        "Validation uses UAM's official camera-4 query / camera-1/2/3 gallery split. "
        "Leaderboard measures Rivas camera-4 generalization. "
        "Gap = UAM->Rivas domain gap. Both should move in the same direction."
    ),
}

with open(OUT_DIR / "split_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

total_kb = sum((OUT_DIR / f).stat().st_size for f in
    ["train_list.csv", "val_query.csv", "val_gallery.csv", "split_metadata.json"]) / 1024
print(f"Artifacts written. Total: {total_kb:.1f} KB")
assert total_kb < 1024, "FAIL: artifacts exceed 1 MB"

print("\nHashes:")
for k in ["train_filename_hash", "val_query_filename_hash", "val_gallery_filename_hash"]:
    print(f"  {k}: {metadata[k]}")
