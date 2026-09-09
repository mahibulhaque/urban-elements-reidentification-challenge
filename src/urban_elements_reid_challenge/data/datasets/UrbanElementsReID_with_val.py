"""Train pool = original train + validation query + validation gallery.

Used to ablate whether folding the val split into training (with De90's
hyperparameters, which were tuned on val) improves test-side performance.

  train images: 17562 (image_train/) + 517 (val_image_query/) + 1791 (val_image_test/) = 19870
  ids:          1567 (original) + 212 (val, disjoint) = 1779

Query/gallery still point at the val split for monitoring — those numbers
are heavily inflated since the model has seen those images, so save the
*final* checkpoint, not best-by-val.
"""
import csv
import os.path as osp

from .bases import ImageDataset
from ..datasets import DATASET_REGISTRY


def _read_csv(path):
    out = []
    with open(path, newline='') as f:
        rd = csv.reader(f, delimiter=',')
        hdr = next(rd)
        ni = hdr.index('imageName')
        cmi = hdr.index('cameraID')
        oi = hdr.index('objectID') if 'objectID' in hdr else None
        for row in rd:
            pid = int(row[oi]) if oi is not None else -1
            out.append((row[cmi], row[ni], pid))
    return out


@DATASET_REGISTRY.register()
class UrbanElementsReID_with_val(ImageDataset):
    def __init__(self, root='', verbose=True, **kwargs):
        d = self.dataset_dir = root
        train_csv = osp.join(d, 'train_classes.csv')
        valq_csv  = osp.join(d, 'val_query_classes.csv')
        valg_csv  = osp.join(d, 'val_test_classes.csv')
        train_dir = osp.join(d, 'image_train/')
        valq_dir  = osp.join(d, 'val_image_query/')
        valg_dir  = osp.join(d, 'val_image_test/')
        for p in [d, train_csv, valq_csv, valg_csv, train_dir, valq_dir, valg_dir]:
            if not osp.exists(p):
                raise RuntimeError(f'missing {p}')

        train_rows = [r for r in _read_csv(train_csv) if r[2] != -1]
        valq_rows  = _read_csv(valq_csv)
        valg_rows  = _read_csv(valg_csv)

        # Single relabel over the union of pids. Train pids are 0..1566 already
        # contiguous; val pids are disjoint integers, mapped to 1567..1778.
        all_pids = sorted({r[2] for r in train_rows} |
                          {r[2] for r in valq_rows} |
                          {r[2] for r in valg_rows})
        pid2label = {p: i for i, p in enumerate(all_pids)}

        train = []
        for c, n, p in train_rows:
            train.append((osp.join(train_dir, n), pid2label[p], int(c[1:])))
        for c, n, p in valq_rows:
            train.append((osp.join(valq_dir, n), pid2label[p], int(c[1:])))
        for c, n, p in valg_rows:
            train.append((osp.join(valg_dir, n), pid2label[p], int(c[1:])))

        # Keep query/gallery pointing at val (eval will be inflated; informational only)
        self.train = train
        self.query   = [(osp.join(valq_dir, n), p, int(c[1:])) for c, n, p in valq_rows]
        self.gallery = [(osp.join(valg_dir, n), p, int(c[1:])) for c, n, p in valg_rows]

        super().__init__(self.train, self.query, self.gallery, **kwargs)
