#!/usr/bin/env python3
"""Auto-select best model per class and create hybrid submission.

Scans all eval_*.json files, identifies the best model for each class,
and uses their submission CSVs to create a hybrid submission where each
test query is matched using its class-specific best model.
"""
import argparse
import csv
import json
import os
from collections import defaultdict

SUBMISSIONS_DIR = 'submissions'
DATA_DIR = 'Dataset/UrbanUAM_Merged'


def norm(c):
    """Normalize class names."""
    c = c.strip().lower()
    return 'trafficsign' if c == 'trafficsignal' else c


def read_classes(path):
    """Read imageName -> class mapping from CSV."""
    d = {}
    with open(path, newline='') as f:
        rd = csv.reader(f)
        hdr = next(rd)
        ni = hdr.index('imageName')
        ci = hdr.index('Class')
        for r in rd:
            d[r[ni]] = norm(r[ci])
    return d


def read_submission(path):
    """Read submission CSV: imageName -> ranked indices."""
    rows = {}
    with open(path, newline='') as f:
        rd = csv.reader(f)
        next(rd)  # skip header
        for r in rd:
            rows[r[0]] = r[1]
    return rows


def find_best_models_per_class():
    """Scan eval_*.json files and find best model for each class.
    
    Returns:
        dict: {class_name: (best_model_name, best_map, submission_path)}
    """
    classes = ['container', 'crosswalk', 'rubbishbins', 'trafficsign']
    best_per_class = {cls: (-1.0, None, None) for cls in classes}
    
    # Scan all eval JSON files
    for fname in os.listdir(SUBMISSIONS_DIR):
        if not fname.startswith('eval_') or fname.startswith('eval_cls_n'):
            continue
        if not fname.endswith('.json'):
            continue
        
        fpath = os.path.join(SUBMISSIONS_DIR, fname)
        try:
            with open(fpath, 'r') as f:
                data = json.load(f)
        except:
            continue
        
        if 'perclass_params' not in data:
            continue
        
        model_name = data.get('tag', fname.replace('eval_', '').replace('.json', ''))
        perclass_data = data['perclass_params']
        
        # Check for corresponding submission file
        sub_path = os.path.join(SUBMISSIONS_DIR, f'submission_{model_name}_perclass.csv')
        if not os.path.exists(sub_path):
            continue
        
        # Update best for each class
        for cls in classes:
            if cls in perclass_data:
                map_score = perclass_data[cls]['mAP']
                best_score, best_model, best_sub = best_per_class[cls]
                if map_score > best_score:
                    best_per_class[cls] = (map_score, model_name, sub_path)
    
    return best_per_class


def main():
    ap = argparse.ArgumentParser(description='Create hybrid submission using best model per class')
    ap.add_argument('--out', default='submissions/submission_best_per_class_hybrid.csv',
                    help='Output submission CSV path')
    ap.add_argument('--query_classes', default='Dataset/UrbanUAM_Merged/query_classes.csv',
                    help='Path to query_classes.csv')
    ap.add_argument('--verbose', action='store_true', help='Print detailed info')
    args = ap.parse_args()
    
    print("=" * 80)
    print("AUTO-HYBRID: Best Model Per Class")
    print("=" * 80)
    
    # Find best models
    best_per_class = find_best_models_per_class()
    
    print("\nBest model per class:")
    print("-" * 80)
    routing = {}
    for cls in ['container', 'crosswalk', 'rubbishbins', 'trafficsign']:
        best_map, best_model, best_sub = best_per_class[cls]
        if best_model is None:
            print(f"  {cls:15s} ERROR: No submission found")
            continue
        print(f"  {cls:15s} mAP={best_map:.4f}  model={best_model:20s}  {best_sub}")
        routing[cls] = best_sub
    
    if len(routing) < 4:
        print("\nERROR: Could not find all 4 classes")
        return 1
    
    # Load query classes
    print(f"\nLoading query classes from {args.query_classes}...")
    q_cls_test = read_classes(args.query_classes)
    print(f"  Loaded {len(q_cls_test)} test queries")
    
    # Load all submissions
    print("\nLoading submission files...")
    cached = {}
    for cls, sub_path in routing.items():
        cached[cls] = read_submission(sub_path)
        print(f"  {cls:15s}: loaded {len(cached[cls])} entries from {os.path.basename(sub_path)}")
    
    # Build hybrid submission
    print(f"\nBuilding hybrid submission to {args.out}...")
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    
    src_count = {cls: 0 for cls in routing.keys()}
    missing_class = 0
    
    # Get canonical query order (sorted by number)
    any_sub = next(iter(cached.values()))
    names = sorted(any_sub.keys(), key=lambda s: int(s.split('.')[0]))
    
    with open(args.out, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['imageName', 'Corresponding Indexes'])
        
        for name in names:
            cls = q_cls_test.get(name)
            if cls is None or cls not in routing:
                # If no class info, use first available (shouldn't happen)
                cls = next(iter(routing.keys()))
                missing_class += 1
            
            # Get row from best model's submission for this class
            row = cached[cls].get(name)
            if row:
                w.writerow([name, row])
                src_count[cls] += 1
            else:
                print(f"  WARNING: {name} not found in {cls} submission")
    
    print(f"\nHybrid submission created!")
    print(f"  Total queries: {len(names)}")
    if missing_class > 0:
        print(f"  Missing class assignments: {missing_class}")
    print(f"\n  Source distribution:")
    for cls in ['container', 'crosswalk', 'rubbishbins', 'trafficsign']:
        count = src_count.get(cls, 0)
        pct = 100.0 * count / len(names) if names else 0
        print(f"    {cls:15s}: {count:4d} ({pct:5.1f}%)")
    
    print(f"\n[SUCCESS] Saved to {args.out}")
    return 0


if __name__ == '__main__':
    exit(main())
