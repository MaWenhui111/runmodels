"""Data-source definitions (one class per file)."""
from .iris import Iris
from .irisTest_scale import IrisTest_SCALE
from .iris import my_collate_fn


__all__ = ('dirty_label_check', 'Iris', 'IrisTest_SCALE', 'my_collate_fn')
