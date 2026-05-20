import random
import string
from typing import Optional, Union, List, Dict, Tuple
from pathlib import Path
import json
import os

def generate_rle_sequence(min_length: int = 5, max_length: int = 50, 
                         alphabet: str = string.ascii_uppercase) -> str:
    """Generate a random sequence with repeated characters"""
    length = random.randint(min_length, max_length)
    sequence = ""
    
    while len(sequence) < length:
        char = random.choice(alphabet)
        # Create runs of 1-9 characters
        run_length = random.randint(1, min(9, length - len(sequence)))
        sequence += char * run_length
    
    return sequence

def sequence_to_rle(sequence: str) -> str:
    """Convert sequence to RLE format: AAABBC → 3A2B1C"""
    if not sequence:
        return ""
    
    rle = ""
    current_char = sequence[0]
    count = 1
    
    for char in sequence[1:]:
        if char == current_char:
            count += 1
        else:
            rle += f"{count}{current_char}"
            current_char = char
            count = 1
    
    # Add the last run
    rle += f"{count}{current_char}"
    return rle

def rle_to_sequence(rle: str) -> str:
    """Convert RLE format to sequence: 3A2B1C → AAABBC"""
    if not rle:
        return ""
    
    sequence = ""
    i = 0
    
    while i < len(rle):
        # Extract count (can be multiple digits)
        count_str = ""
        while i < len(rle) and rle[i].isdigit():
            count_str += rle[i]
            i += 1
        
        if not count_str or i >= len(rle):
            break
            
        count = int(count_str)
        char = rle[i]
        sequence += char * count
        i += 1
    
    return sequence

def create_rle_dataset(
    num_samples: int = 10000,
    min_seq_length: int = 5,
    max_seq_length: int = 50,
    alphabet: str = string.ascii_uppercase,
    validation_split: float = 0.1,
    test_split: float = 0.1,
    compression_ratio: float = 0.5,  # Ratio of compression vs decompression tasks
    seed: Optional[int] = 42
) -> Dict[str, List[Dict[str, Dict[str, str]]]]:
    """
    Create RLE dataset with both compression and decompression tasks
    """
    if seed is not None:
        random.seed(seed)
    
    samples = []
    
    # Generate compression tasks (sequence → RLE)
    num_compression = int(num_samples * compression_ratio)
    for _ in range(num_compression):
        sequence = generate_rle_sequence(min_seq_length, max_seq_length, alphabet)
        rle = sequence_to_rle(sequence)
        
        samples.append({
            "translation": {
                "src": sequence,  # Original sequence
                "tgt": rle        # RLE compressed
            },
            "task_type": "compression"
        })
    
    # Generate decompression tasks (RLE → sequence)
    num_decompression = num_samples - num_compression
    for _ in range(num_decompression):
        sequence = generate_rle_sequence(min_seq_length, max_seq_length, alphabet)
        rle = sequence_to_rle(sequence)
        
        samples.append({
            "translation": {
                "src": rle,       # RLE compressed
                "tgt": sequence   # Original sequence
            },
            "task_type": "decompression"
        })
    
    # Shuffle samples
    random.shuffle(samples)
    
    # Split into train/validation/test
    total_samples = len(samples)
    test_size = int(total_samples * test_split)
    val_size = int(total_samples * validation_split)
    train_size = total_samples - test_size - val_size
    
    splits = {
        "train": samples[:train_size],
        "validation": samples[train_size:train_size + val_size],
        "test": samples[train_size + val_size:]
    }
    
    # Print statistics
    for split_name, split_data in splits.items():
        compression_tasks = sum(1 for item in split_data if item["task_type"] == "compression")
        decompression_tasks = len(split_data) - compression_tasks
        print(f"{split_name}: {len(split_data)} samples "
              f"({compression_tasks} compression, {decompression_tasks} decompression)")
    
    return splits

def load_rle_dataset(
    dataset_path: Optional[Union[str, Path]] = None,
    num_samples: int = 10000,
    min_seq_length: int = 5,
    max_seq_length: int = 50,
    alphabet: str = string.ascii_uppercase,
    validation_split: float = 0.1,
    test_split: float = 0.1,
    compression_ratio: float = 0.5,
    force_reload: bool = False,
    seed: Optional[int] = 42,
    **kwargs
) -> Dict[str, List[Dict[str, Dict[str, str]]]]:
    """
    Load or create RLE dataset with caching support
    """
    
    # Create cache filename based on parameters
    cache_filename = (f"rle_dataset_n{num_samples}_len{min_seq_length}-{max_seq_length}_"
                     f"alpha{len(alphabet)}_comp{compression_ratio}_seed{seed}.json")
    
    if dataset_path:
        dataset_path = Path(dataset_path).resolve()
        cache_path = dataset_path / cache_filename
        
        # Try to load existing dataset
        if not force_reload and cache_path.exists():
            try:
                print(f"Loading existing RLE dataset from: {cache_path}")
                with open(cache_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Failed to load cached dataset: {e}")
                print("Generating new dataset...")
        
        # Create directory if it doesn't exist
        dataset_path.mkdir(parents=True, exist_ok=True)
    
    # Generate new dataset
    print(f"Generating RLE dataset with {num_samples} samples...")
    dataset = create_rle_dataset(
        num_samples=num_samples,
        min_seq_length=min_seq_length,
        max_seq_length=max_seq_length,
        alphabet=alphabet,
        validation_split=validation_split,
        test_split=test_split,
        compression_ratio=compression_ratio,
        seed=seed
    )
    
    # Save dataset if path is provided
    if dataset_path:
        cache_path = dataset_path / cache_filename
        try:
            print(f"Saving RLE dataset to: {cache_path}")
            with open(cache_path, 'w') as f:
                json.dump(dataset, f, indent=2)
        except Exception as e:
            print(f"Warning: Failed to save dataset: {e}")
    
    return dataset

# Validation functions
def validate_rle_conversion(sequence: str, rle: str) -> bool:
    """Validate that RLE conversion is correct"""
    try:
        # Test round-trip conversion
        converted_rle = sequence_to_rle(sequence)
        converted_seq = rle_to_sequence(rle)
        
        return (converted_rle == rle and 
                converted_seq == sequence and 
                rle_to_sequence(converted_rle) == sequence and
                sequence_to_rle(converted_seq) == rle)
    except:
        return False

def analyze_rle_dataset(dataset: Dict[str, List]) -> Dict[str, any]:
    """Analyze RLE dataset statistics"""
    stats = {
        "total_samples": 0,
        "compression_tasks": 0,
        "decompression_tasks": 0,
        "avg_src_length": 0,
        "avg_tgt_length": 0,
        "max_src_length": 0,
        "max_tgt_length": 0,
        "compression_ratio": 0,
        "unique_characters": set()
    }
    
    all_samples = []
    for split_data in dataset.values():
        all_samples.extend(split_data)
    
    stats["total_samples"] = len(all_samples)
    
    src_lengths = []
    tgt_lengths = []
    
    for sample in all_samples:
        src = sample["translation"]["src"]
        tgt = sample["translation"]["tgt"]
        
        src_lengths.append(len(src))
        tgt_lengths.append(len(tgt))
        
        # Collect unique characters
        stats["unique_characters"].update(src)
        stats["unique_characters"].update(tgt)
        
        # Count task types
        if sample["task_type"] == "compression":
            stats["compression_tasks"] += 1
        else:
            stats["decompression_tasks"] += 1
    
    stats["avg_src_length"] = sum(src_lengths) / len(src_lengths)
    stats["avg_tgt_length"] = sum(tgt_lengths) / len(tgt_lengths)
    stats["max_src_length"] = max(src_lengths)
    stats["max_tgt_length"] = max(tgt_lengths)
    stats["compression_ratio"] = stats["avg_tgt_length"] / stats["avg_src_length"]
    stats["unique_characters"] = sorted(list(stats["unique_characters"]))
    
    return stats

if __name__ == "__main__":
    # Example usage
    dataset = load_rle_dataset(
        dataset_path="./data/rle_datasets",
        num_samples=1000,
        min_seq_length=5,
        max_seq_length=30,
        seed=42
    )
    
    # Analyze dataset
    stats = analyze_rle_dataset(dataset)
    print("\nDataset Statistics:")
    print(f"Total samples: {stats['total_samples']}")
    print(f"Compression tasks: {stats['compression_tasks']}")
    print(f"Decompression tasks: {stats['decompression_tasks']}")
    print(f"Average source length: {stats['avg_src_length']:.2f}")
    print(f"Average target length: {stats['avg_tgt_length']:.2f}")
    print(f"Compression ratio: {stats['compression_ratio']:.2f}")
    print(f"Unique characters: {stats['unique_characters']}")
    
    # Test some examples
    print("\nExample samples:")
    for i, sample in enumerate(dataset["train"][:3]):
        src = sample["translation"]["src"]
        tgt = sample["translation"]["tgt"]
        task = sample["task_type"]
        print(f"{i+1}. {task}: {src} → {tgt}")
        print(f"   Valid: {validate_rle_conversion(src if task == 'compression' else tgt, tgt if task == 'compression' else src)}")