"""Original UrbanElementsReID training data + MNN-pseudo-labeled c004 test images.

The MNN pairs are produced by scripts/mnn_pseudo_label.py and stored at
`outputs/MNN/pseudo_labels.csv` (each pseudo-id corresponds to two images:
one c004 query and one c001-c003 gallery item that mutually nearest-neighbor
each other in Hplus feature space).

We treat each MNN pair as a *new identity* on top of the original training
identities, then relabel the union to a contiguous 0..N-1 PID range.
"""
import csv
import os
import os.path as osp

from .bases import ImageDataset
from ..datasets import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class UrbanElementsReID_with_MNN(ImageDataset):

    PSEUDO_LABELS_PATH = '/home/mahibulhaque/ReID/Urban-Elements-ReID-Challenge-2026/outputs/MNN/pseudo_labels.csv'

    def __init__(self, root='', verbose=True, **kwargs):
        self.dataset_dir = root
        self.train_dir = osp.join(self.dataset_dir, 'image_train/')
        self.train_csv = osp.join(self.dataset_dir, 'train_classes.csv')
        for p in [self.dataset_dir, self.train_dir, self.train_csv,
                  self.PSEUDO_LABELS_PATH]:
            if not osp.exists(p):
                raise RuntimeError(f"missing {p}")

        # 1. Original training samples
        orig = []
        with open(self.train_csv, newline='') as f:
            rd = csv.reader(f); next(rd)
            for row in rd:
                cam_str, name, pid_str = row[0], row[1], row[2]
                pid = int(pid_str)
                if pid == -1: continue
                cam = int(cam_str.lstrip('c'))
                orig.append((osp.join(self.train_dir, name), pid, cam))

        # 2. MNN pseudo-labeled samples (already pid-offset to avoid collision)
        mnn = []
        with open(self.PSEUDO_LABELS_PATH, newline='') as f:
            rd = csv.DictReader(f)
            for row in rd:
                pid = int(row['pseudo_id'])
                cam = int(row['camid'])
                mnn.append((row['image_path'], pid, cam))

        # 3. Relabel union into contiguous 0..N-1 PID range
        all_samples = orig + mnn
        pid_set = sorted({pid for _, pid, _ in all_samples})
        pid2label = {p: i for i, p in enumerate(pid_set)}
        train = [(path, pid2label[pid], cam) for path, pid, cam in all_samples]

        if verbose:
            n_orig_ids = len({p for _, p, _ in orig})
            n_mnn_ids  = len({p for _, p, _ in mnn})
            print(f'[UrbanElementsReID_with_MNN] '
                  f'orig: {len(orig)} imgs / {n_orig_ids} ids   '
                  f'+ MNN: {len(mnn)} imgs / {n_mnn_ids} ids   '
                  f'-> total: {len(train)} imgs / {len(pid_set)} ids')

        self.train = train
        # Use train as query/gallery placeholders (val mAP comes from
        # UrbanElementsReID_val via cfg.DATASETS.TEST).
        self.query = train
        self.gallery = train
        super().__init__(self.train, self.query, self.gallery, **kwargs)
