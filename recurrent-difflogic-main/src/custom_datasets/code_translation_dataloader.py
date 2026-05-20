from datasets import load_dataset, DatasetDict, Dataset
from pathlib import Path
import pandas as pd
import json
import os
from typing import Optional, Union, List, Dict
import requests
from io import StringIO

def load_code_translation_dataset(
    source_lang: str = "cpp",
    target_lang: str = "java", 
    dataset_path: Optional[Union[str, Path]] = None,
    streaming: bool = False,
    force_reload: bool = False,
    subset_size: Optional[int] = None,
    split_ratios: Dict[str, float] = {"train": 0.8, "validation": 0.1, "test": 0.1},
    **kwargs
) -> DatasetDict:
    """
    Load CodeTransOcean dataset for code translation.
    
    Args:
        source_lang: Source programming language (cpp, java, python, etc.)
        target_lang: Target programming language (java, cpp, python, etc.)
        dataset_path: Local path to store/load the dataset
        streaming: If True, streams the dataset without saving to disk
        force_reload: Force fresh download even if local copy exists
        subset_size: Optional int. If set, limits dataset to this size.
        split_ratios: Dictionary defining train/validation/test split ratios
        **kwargs: Additional arguments
        
    Returns:
        DatasetDict with train/validation/test splits containing code pairs
    """
    
    # Language mapping for CodeTransOcean dataset
    lang_mapping = {
        "cpp": "C++",
        "c": "C",
        "java": "Java", 
        "python": "Python",
        "javascript": "JavaScript",
        "csharp": "C#",
        "php": "PHP",
        "go": "Go",
        "vb": "VB"
    }
    
    # Validate language codes
    if source_lang not in lang_mapping or target_lang not in lang_mapping:
        raise ValueError(f"Supported languages: {list(lang_mapping.keys())}")
    
    if source_lang == target_lang:
        raise ValueError("Source and target languages must be different")
    
    src_col = lang_mapping[source_lang]
    tgt_col = lang_mapping[target_lang]
    
    # Create save directory if specified
    save_dir = None
    if dataset_path and not streaming:
        dataset_path = Path(dataset_path).resolve()
        save_dir = dataset_path / f"code_trans_{source_lang}_to_{target_lang}"
        
        if not force_reload and save_dir.exists():
            try:
                print(f"Loading existing dataset from: {save_dir}")
                ds = DatasetDict.load_from_disk(str(save_dir))
                if subset_size:
                    # Apply subset to all splits proportionally
                    for split in ds.keys():
                        split_size = int(subset_size * split_ratios.get(split, 0.33))
                        if split_size > 0:
                            ds[split] = ds[split].select(range(min(split_size, len(ds[split]))))
                return ds
            except Exception as e:
                print(f"Invalid existing dataset: {str(e)}")
                print("Attempting fresh download...")

        save_dir.mkdir(parents=True, exist_ok=True)

    try:
        print(f"Loading CodeTransOcean dataset for {source_lang} -> {target_lang}")
        
        # Load the multilingual translation subset specifically
        # We need to handle the data files manually due to schema inconsistencies
        data_files = {
            "train": "CodeTrans Datasets/MultilingualTrans/multilingual_train.json"
        }
        
        print("Loading multilingual translation data...")
        
        if streaming:
            # For streaming, load directly and filter
            ds = load_dataset(
                "WeixiangYan/CodeTransOcean",
                data_files=data_files,
                streaming=True,
                **kwargs
            )
            
            def filter_and_format_streaming(example):
                # Check if both required columns exist and have valid code
                if (src_col in example and tgt_col in example and 
                    example[src_col] and example[tgt_col] and
                    example[src_col].strip() and example[tgt_col].strip()):
                    return {
                        'translation': {
                            source_lang: example[src_col].strip(),
                            target_lang: example[tgt_col].strip()
                        }
                    }
                return None
            
            filtered_ds = ds.map(filter_and_format_streaming)
            return filtered_ds
        
        # For non-streaming, load and process manually
        print("Downloading and processing dataset...")
        
        # Load raw dataset
        raw_ds = load_dataset(
            "WeixiangYan/CodeTransOcean",
            data_files=data_files,
            **kwargs
        )
        
        # Process the data to extract our language pair
        translation_pairs = []
        
        for example in raw_ds['train']:
            # Check if both columns exist and contain valid code
            if (src_col in example and tgt_col in example and 
                example[src_col] and example[tgt_col]):
                
                src_code = example[src_col].strip()
                tgt_code = example[tgt_col].strip()
                
                # Skip empty or very short code snippets
                if len(src_code) > 10 and len(tgt_code) > 10:
                    translation_pairs.append({
                        'translation': {
                            source_lang: src_code,
                            target_lang: tgt_code
                        }
                    })
        
        if len(translation_pairs) == 0:
            raise ValueError(f"No valid translation pairs found for {source_lang} -> {target_lang}")
        
        print(f"Found {len(translation_pairs)} valid translation pairs")
        
        # Apply subset if specified
        if subset_size and len(translation_pairs) > subset_size:
            import random
            random.shuffle(translation_pairs)
            translation_pairs = translation_pairs[:subset_size]
            print(f"Limited to {len(translation_pairs)} examples")
        
        # Split into train/validation/test
        train_size = int(len(translation_pairs) * split_ratios['train'])
        val_size = int(len(translation_pairs) * split_ratios['validation'])
        
        train_data = translation_pairs[:train_size]
        val_data = translation_pairs[train_size:train_size + val_size]
        test_data = translation_pairs[train_size + val_size:]
        
        # Ensure we have data for all splits
        if len(val_data) == 0 and len(train_data) > 1:
            val_data = [train_data.pop()]
        if len(test_data) == 0 and len(train_data) > 1:
            test_data = [train_data.pop()]
        
        # Create DatasetDict
        dataset_dict = DatasetDict({
            'train': Dataset.from_list(train_data),
            'validation': Dataset.from_list(val_data),
            'test': Dataset.from_list(test_data)
        })
        
        print(f"Dataset splits: train={len(train_data)}, val={len(val_data)}, test={len(test_data)}")
        
        # Save if specified
        if save_dir and not streaming:
            print(f"Saving dataset to: {save_dir}")
            dataset_dict.save_to_disk(str(save_dir), max_shard_size="1GB")

        return dataset_dict

    except Exception as e:
        print(f"Failed to load from HuggingFace. Error: {str(e)}")
        print("Trying fallback approach with sample data...")
        
        # Fallback: Create a small sample dataset
        return create_sample_code_dataset(source_lang, target_lang, subset_size or 1000, split_ratios)


def create_sample_code_dataset(
    source_lang: str,
    target_lang: str, 
    num_samples: int = 1000,
    split_ratios: Dict[str, float] = {"train": 0.8, "validation": 0.1, "test": 0.1}
) -> DatasetDict:
    """
    Create a sample code translation dataset for testing.
    """
    print(f"Creating sample {source_lang} -> {target_lang} dataset with {num_samples} examples")
    
    # Sample code pairs (you can expand this)
    sample_pairs = []
    
    if source_lang == "cpp" and target_lang == "java":
        cpp_java_pairs = [
            {
                "cpp": "#include <iostream>\nusing namespace std;\n\nint main() {\n    cout << \"Hello World\" << endl;\n    return 0;\n}",
                "java": "public class Main {\n    public static void main(String[] args) {\n        System.out.println(\"Hello World\");\n    }\n}"
            },
            {
                "cpp": "#include <vector>\nusing namespace std;\n\nint sum(vector<int>& nums) {\n    int total = 0;\n    for(int num : nums) {\n        total += num;\n    }\n    return total;\n}",
                "java": "import java.util.List;\n\npublic class Solution {\n    public int sum(List<Integer> nums) {\n        int total = 0;\n        for(int num : nums) {\n            total += num;\n        }\n        return total;\n    }\n}"
            },
            {
                "cpp": "#include <iostream>\n#include <string>\nusing namespace std;\n\nclass Person {\npublic:\n    string name;\n    int age;\n    \n    Person(string n, int a) : name(n), age(a) {}\n    \n    void greet() {\n        cout << \"Hi, I'm \" << name << endl;\n    }\n};",
                "java": "public class Person {\n    private String name;\n    private int age;\n    \n    public Person(String name, int age) {\n        this.name = name;\n        this.age = age;\n    }\n    \n    public void greet() {\n        System.out.println(\"Hi, I'm \" + name);\n    }\n}"
            }
        ]
        sample_pairs = cpp_java_pairs
    
    # Replicate samples to reach desired size
    translation_pairs = []
    for i in range(num_samples):
        base_pair = sample_pairs[i % len(sample_pairs)]
        # Add some variation to make examples unique
        src_code = base_pair[source_lang]
        tgt_code = base_pair[target_lang]
        
        # Simple variation: add comments
        if i % 3 == 1:
            src_code = f"// Example {i+1}\n" + src_code
            tgt_code = f"// Example {i+1}\n" + tgt_code
        elif i % 3 == 2:
            src_code = src_code + f"\n// End of example {i+1}"
            tgt_code = tgt_code + f"\n// End of example {i+1}"
        
        translation_pairs.append({
            'translation': {
                source_lang: src_code,
                target_lang: tgt_code
            }
        })
    
    # Split the data
    train_size = int(len(translation_pairs) * split_ratios['train'])
    val_size = int(len(translation_pairs) * split_ratios['validation'])
    
    train_data = translation_pairs[:train_size]
    val_data = translation_pairs[train_size:train_size + val_size]
    test_data = translation_pairs[train_size + val_size:]
    
    return DatasetDict({
        'train': Dataset.from_list(train_data),
        'validation': Dataset.from_list(val_data),
        'test': Dataset.from_list(test_data)
    })


def load_custom_code_pairs(
    source_files: List[str],
    target_files: List[str],
    source_lang: str,
    target_lang: str,
    split_ratios: Dict[str, float] = {"train": 0.8, "validation": 0.1, "test": 0.1}
) -> DatasetDict:
    """
    Load custom code translation pairs from local files.
    
    Args:
        source_files: List of paths to source code files
        target_files: List of paths to target code files (must match source_files length)
        source_lang: Source language identifier
        target_lang: Target language identifier
        split_ratios: Train/validation/test split ratios
        
    Returns:
        DatasetDict with translation pairs
    """
    if len(source_files) != len(target_files):
        raise ValueError("Source and target file lists must have the same length")
    
    translation_pairs = []
    
    for src_file, tgt_file in zip(source_files, target_files):
        with open(src_file, 'r', encoding='utf-8') as f:
            src_code = f.read().strip()
        with open(tgt_file, 'r', encoding='utf-8') as f:
            tgt_code = f.read().strip()
        
        if src_code and tgt_code:  # Skip empty files
            translation_pairs.append({
                'translation': {source_lang: src_code, target_lang: tgt_code}
            })
    
    # Split the data
    train_size = int(len(translation_pairs) * split_ratios['train'])
    val_size = int(len(translation_pairs) * split_ratios['validation'])
    
    train_data = translation_pairs[:train_size]
    val_data = translation_pairs[train_size:train_size + val_size]
    test_data = translation_pairs[train_size + val_size:]
    
    return DatasetDict({
        'train': Dataset.from_list(train_data),
        'validation': Dataset.from_list(val_data),
        'test': Dataset.from_list(test_data)
    })