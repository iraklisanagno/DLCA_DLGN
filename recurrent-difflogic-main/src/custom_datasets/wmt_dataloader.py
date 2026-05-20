from datasets import load_dataset, DatasetDict, load_from_disk
from pathlib import Path
import os
from typing import Optional, Union

def load_wmt_dataset(
    dataset_name: str = "wmt14",
    language_pair: str = "de-en",
    dataset_path: Optional[Union[str, Path]] = None,
    streaming: bool = False,
    force_reload: bool = False,
    subset_size: Optional[int] = None,
    **kwargs
) -> DatasetDict:
    """
    Load WMT dataset with robust path handling and optional subsampling.
    
    Args:
        dataset_name: One of wmt14, wmt16, wmt19
        language_pair: Language pair (e.g. 'de-en', 'cs-en')
        dataset_path: Absolute path to store/load the dataset
        streaming: If True, streams the dataset without saving to disk
        force_reload: Force fresh download even if local copy exists
        subset_size: Optional int. If set, truncates the train split to this size.
        **kwargs: Additional arguments for load_dataset
        
    Returns:
        DatasetDict with train/validation/test splits
    """
    if streaming:
        return load_dataset(
            f"wmt/{dataset_name}",
            language_pair,
            streaming=True,
            **kwargs
        )

    save_dir = None
    if dataset_path:
        dataset_path = Path(dataset_path).resolve()
        save_dir = dataset_path / f"{dataset_name}_{language_pair.replace('-', '_')}"
        
        if not force_reload and save_dir.exists():
            try:
                print(f"Loading existing dataset from: {save_dir}")
                ds = DatasetDict.load_from_disk(str(save_dir))
                if subset_size:
                    ds["train"] = ds["train"].select(range(subset_size))
                return ds
            except Exception as e:
                print(f"Invalid existing dataset: {str(e)}")
                print("Attempting fresh download...")

        save_dir.mkdir(parents=True, exist_ok=True)

    try:
        ds = load_dataset(
            f"wmt/{dataset_name}",
            language_pair,
            **kwargs
        )

        if subset_size:
            ds["train"] = ds["train"].select(range(subset_size))
        
        if save_dir and not streaming:
            print(f"Saving dataset to: {save_dir}")
            ds.save_to_disk(str(save_dir), max_shard_size="1GB")

        return ds

    except Exception as e:
        raise RuntimeError(
            f"Failed to load {dataset_name} {language_pair}.\n"
            f"1. Accept terms at https://huggingface.co/datasets/wmt/{dataset_name}\n"
            f"2. Original error: {str(e)}"
        )
