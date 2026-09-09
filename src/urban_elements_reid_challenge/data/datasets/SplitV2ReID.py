import os
import pandas as pd
from .bases import ImageDataset
from ..datasets import DATASET_REGISTRY

_UAM_ROOT   = '/kaggle/input/datasets/mahibulhaque/uam-dataset/UAM_Unified'
_RIVAS_ROOT = '/kaggle/input/datasets/mahibulhaque/urban2026/Urban2026'


def _img_path(row):
    """Resolve full image path from a split CSV row."""
    if row.source_dataset == 'uam':
        return os.path.join(_UAM_ROOT, row.img_subdir, row.filename)
    else:
        return os.path.join(_RIVAS_ROOT, 'image_train', row.filename)


@DATASET_REGISTRY.register()
class SplitV2ReID(ImageDataset):
    """
    Split-v2 (corrected): uses UAM's official pre-made query/test split.
      train_list.csv  → UAM train (IDs 1-479, all cameras) + all Rivas
      val_query.csv   → UAM query_classes (IDs 480-691, c004)       [image_query/]
      val_gallery.csv → UAM test_classes  (IDs 480-691, c001/2/3)   [image_test/]

    UAM and Rivas share overlapping numeric identity_ids — training PIDs are
    disambiguated via a (source_dataset, identity_id) → global int mapping.
    Val query/gallery both come from UAM's disjoint ID range (480-691) so raw
    identity_id is used directly for matching.
    """

    def __init__(self, root='/kaggle/working/splits_v2', verbose=True, **kwargs):
        self.split_dir = root
        train   = self._load_train('train_list.csv')
        query   = self._load_eval('val_query.csv')
        gallery = self._load_eval('val_gallery.csv')
        super(SplitV2ReID, self).__init__(train, query, gallery, **kwargs)

    def _load_train(self, csv_name):
        df = pd.read_csv(os.path.join(self.split_dir, csv_name))

        # Globally unique PID: (source_dataset, identity_id) → int
        pid_map, counter = {}, 0
        for row in df.itertuples(index=False):
            key = (row.source_dataset, int(row.identity_id))
            if key not in pid_map:
                pid_map[key] = counter
                counter += 1

        items = []
        for row in df.itertuples(index=False):
            items.append((
                _img_path(row),
                pid_map[(row.source_dataset, int(row.identity_id))],
                int(row.camera_id),
            ))
        return items

    def _load_eval(self, csv_name):
        df = pd.read_csv(os.path.join(self.split_dir, csv_name))
        return [
            (_img_path(row), int(row.identity_id), int(row.camera_id))
            for row in df.itertuples(index=False)
        ]
