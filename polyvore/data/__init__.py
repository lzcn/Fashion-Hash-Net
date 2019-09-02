"""The :mode:`data` is created for dataset and data-loader."""
from .fashionset import get_dataloader, get_dataset

# TODO: Check 'mode' data
__all__ = ["get_dataset", "get_dataloader"]
