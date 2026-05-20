from datasets import load_dataset, DatasetDict, load_from_disk
from pathlib import Path
import os
from typing import Optional, Union

def load_video_dataset(
    dataset_name: str = "kinetics400",
    config_name: str = "256x256",  # Common video resolution config
    dataset_path: Optional[Union[str, Path]] = None,
    streaming: bool = False,
    force_reload: bool = False,
    subset_size: Optional[int] = None,
    **kwargs
) -> DatasetDict:
    """
    Load video dataset with robust path handling and optional subsampling.
    
    Args:
        dataset_name: Name of video dataset (e.g., 'kinetics400', 'ucf101', 'hmdb51')
        config_name: Configuration name (e.g. resolution '256x256', version '1.0.0')
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
            dataset_name,
            config_name,
            streaming=True,
            **kwargs
        )

    save_dir = None
    if dataset_path:
        dataset_path = Path(dataset_path).resolve()
        save_dir = dataset_path / f"{dataset_name}_{config_name.replace('/', '_')}"
        
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
            dataset_name,
            config_name,
            **kwargs
        )

        # Handle common video dataset split variations
        split_mapping = {
            "val": "validation",
            "test": "eval",
            "evaluation": "test"
        }
        ds = ds.rename_splits(split_mapping)

        if subset_size:
            ds["train"] = ds["train"].select(range(subset_size))
        
        if save_dir and not streaming:
            print(f"Saving dataset to: {save_dir}")
            ds.save_to_disk(str(save_dir), max_shard_size="2GB")  # Larger shards for video data

        return ds

    except Exception as e:
        raise RuntimeError(
            f"Failed to load {dataset_name} {config_name}.\n"
            f"1. Check dataset requirements at https://huggingface.co/datasets/{dataset_name}\n"
            f"2. Video datasets often require additional dependencies (e.g., decord, av)\n"
            f"3. Original error: {str(e)}"
        )