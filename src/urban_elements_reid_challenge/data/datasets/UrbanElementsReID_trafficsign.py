"""Trafficsign-only subset of UrbanElementsReID (train) and val.

For specialist training: filter the training data to trafficsign IDs only,
relabel into a contiguous 0..N-1 PID range. Same logic for the val split so
EVAL_PERIOD sees only trafficsign queries (318 queries on val).
"""
import csv
import os.path as osp

from .bases import ImageDataset
from ..datasets import DATASET_REGISTRY


def _norm_cls(c: str) -> str:
    c = c.strip().lower()
    return 'trafficsign' if c == 'trafficsignal' else c


def _read_csv_with_class(csv_path):
    """Returns rows of (camid, imageName, objectID, class)."""
    rows = []
    with open(csv_path, newline='') as f:
        rd = csv.reader(f, delimiter=',')
        hdr = next(rd)
        ni = hdr.index('imageName')
        ci = hdr.index('Class')
        oi = hdr.index('objectID') if 'objectID' in hdr else None
        cmi = hdr.index('cameraID')
        for row in rd:
            cls = _norm_cls(row[ci])
            pid = int(row[oi]) if oi is not None else -1
            rows.append((row[cmi], row[ni], pid, cls))
    return rows


@DATASET_REGISTRY.register()
class UrbanElementsReID_trafficsign(ImageDataset):
    """Train-only subset filtered to trafficsign class."""

    TARGET_CLASS = 'trafficsign'

    def __init__(self, root='', verbose=True, **kwargs):
        self.dataset_dir = root
        self.train_dir = osp.join(self.dataset_dir, 'image_train/')
        self.train_csv = osp.join(self.dataset_dir, 'train_classes.csv')
        for p in [self.dataset_dir, self.train_dir, self.train_csv]:
            if not osp.exists(p):
                raise RuntimeError(f"missing {p}")

        rows = _read_csv_with_class(self.train_csv)
        rows = [r for r in rows if r[3] == self.TARGET_CLASS and r[2] != -1]

        pid_container = sorted({r[2] for r in rows})
        pid2label = {pid: lab for lab, pid in enumerate(pid_container)}

        train = []
        for camid, imageName, pid, _cls in rows:
            train.append((osp.join(self.train_dir, imageName),
                          pid2label[pid], int(camid[1:])))
        # Use train as both query/gallery for sampler-side compatibility (the
        # actual val mAP is computed via UrbanElementsReID_val_trafficsign).
        self.train = train
        self.query = train
        self.gallery = train

        super().__init__(self.train, self.query, self.gallery, **kwargs)


@DATASET_REGISTRY.register()
class UrbanElementsReID_val_trafficsign(ImageDataset):
    """Validation split filtered to trafficsign queries / gallery."""

    TARGET_CLASS = 'trafficsign'

    def __init__(self, root='', verbose=True, **kwargs):
        self.dataset_dir = root
        self.train_dir = osp.join(self.dataset_dir, 'image_train/')
        self.query_dir = osp.join(self.dataset_dir, 'val_image_query/')
        self.gallery_dir = osp.join(self.dataset_dir, 'val_image_test/')
        self.train_csv = osp.join(self.dataset_dir, 'train_classes.csv')
        self.query_csv = osp.join(self.dataset_dir, 'val_query_classes.csv')
        self.gallery_csv = osp.join(self.dataset_dir, 'val_test_classes.csv')
        for p in [self.dataset_dir, self.train_dir, self.query_dir, self.gallery_dir,
                  self.train_csv, self.query_csv, self.gallery_csv]:
            if not osp.exists(p):
                raise RuntimeError(f"missing {p}")

        # train side (relabeled, filtered to trafficsign)
        train_rows = _read_csv_with_class(self.train_csv)
        train_rows = [r for r in train_rows if r[3] == self.TARGET_CLASS and r[2] != -1]
        pid_container = sorted({r[2] for r in train_rows})
        pid2label = {pid: lab for lab, pid in enumerate(pid_container)}
        train = [(osp.join(self.train_dir, n), pid2label[pid], int(c[1:]))
                 for c, n, pid, _ in train_rows]

        # query / gallery (real PIDs preserved, filtered to trafficsign)
        def _filtered_split(dir_path, csv_path):
            rows = _read_csv_with_class(csv_path)
            rows = [r for r in rows if r[3] == self.TARGET_CLASS]
            return [(osp.join(dir_path, n), pid, int(c[1:])) for c, n, pid, _ in rows]

        self.train = train
        self.query = _filtered_split(self.query_dir, self.query_csv)
        self.gallery = _filtered_split(self.gallery_dir, self.gallery_csv)

        super().__init__(self.train, self.query, self.gallery, **kwargs)
