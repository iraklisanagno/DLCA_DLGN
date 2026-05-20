from custom_datasets.wmt_dataloader import load_wmt_dataset
from custom_datasets.timeseries_dataloader import load_timeseries_dataset
from custom_datasets.video_dataloader import load_video_dataset
from custom_datasets.rle_dataloader import load_rle_dataset
from custom_datasets.code_translation_dataloader import load_code_translation_dataset
from custom_datasets.permutation_dataloader import load_permutation_dataset

DATASET_REGISTRY = {
    "wmt": load_wmt_dataset,
    "timeseries": load_timeseries_dataset,
    "video": load_video_dataset,
    "rle": load_rle_dataset,
    "code_translation": load_code_translation_dataset,
    "permutation": load_permutation_dataset
}