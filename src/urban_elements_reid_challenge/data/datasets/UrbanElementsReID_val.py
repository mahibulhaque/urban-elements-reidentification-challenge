# encoding: utf-8
"""
Validation split for UrbanElementsReID.

Uses val_image_query/ vs val_image_test/ for monitoring mAP during training.
"""
import csv
import os.path as osp

from .bases import ImageDataset
from ..datasets import DATASET_REGISTRY


def _read_csv_eval(csv_path):
    """Reads val_*_classes.csv, returns (camid, imageName, pid)."""
    out = []
    with open(csv_path, newline='') as f:
        rd = csv.reader(f, delimiter=',')
        hdr = next(rd)
        ni = hdr.index('imageName')
        ci = hdr.index('Class')
        oi = hdr.index('objectID') if 'objectID' in hdr else None
        cmi = hdr.index('cameraID')
        for row in rd:
            pid = int(row[oi]) if oi is not None else -1
            out.append((row[cmi], row[ni], pid))
    return out


@DATASET_REGISTRY.register()
class UrbanElementsReID_val(ImageDataset):
    """Validation split: val_image_query/ (query) vs val_image_test/ (gallery).

    Also loads train for the triplet sampler.
    """

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

        # train split (contiguous PIDs for classifier head)
        train_rows = _read_csv_eval(self.train_csv)
        train_rows = [r for r in train_rows if r[2] != -1]
        pid2label = {pid: lab for lab, pid in enumerate(sorted({r[2] for r in train_rows}))}
        train = [(osp.join(self.train_dir, n), pid2label[pid], int(c[1:]))
                 for c, n, pid in train_rows]

        # query / gallery splits (original PIDs preserved for eval)
        def _split(dir_path, csv_path):
            rows = _read_csv_eval(csv_path)
            return [(osp.join(dir_path, n), pid, int(c[1:])) for c, n, pid in rows]

        self.train = train
        self.query = _split(self.query_dir, self.query_csv)
        self.gallery = _split(self.gallery_dir, self.gallery_csv)

        super().__init__(self.train, self.query, self.gallery, **kwargs)
