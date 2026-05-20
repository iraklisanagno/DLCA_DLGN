import random
import string
from typing import Optional, Union, List, Dict, Tuple
from pathlib import Path
import json
import os

def generate_sequence(min_length: int = 3, max_length: int = 20, 
                     alphabet: str = string.ascii_uppercase) -> List[str]:
    """Generate a random sequence of characters"""
    length = random.randint(min_length, max_length)
    return [random.choice(alphabet) for _ in range(length)]

def apply_reverse_permutation(sequence: List[str]) -> List[str]:
    """Apply reverse permutation: [A, B, C, D] → [D, C, B, A]"""
    return sequence[::-1]

def apply_swap_pairs_permutation(sequence: List[str]) -> List[str]:
    """Apply swap pairs permutation: [A, B, C, D] → [B, A, D, C]"""
    result = sequence.copy()
    for i in range(0, len(result) - 1, 2):
        result[i], result[i + 1] = result[i + 1], result[i]
    return result

def apply_rotate_left_permutation(sequence: List[str]) -> List[str]:
    """Apply rotate left permutation: [A, B, C, D] → [B, C, D, A]"""
    if len(sequence) == 0:
        return sequence
    return sequence[1:] + [sequence[0]]

def apply_rotate_right_permutation(sequence: List[str]) -> List[str]:
    """Apply rotate right permutation: [A, B, C, D] → [D, A, B, C]"""
    if len(sequence) == 0:
        return sequence
    return [sequence[-1]] + sequence[:-1]

def apply_sort_ascending_permutation(sequence: List[str]) -> List[str]:
    """Apply sort ascending permutation: [C, A, D, B] → [A, B, C, D]"""
    return sorted(sequence)

def apply_sort_descending_permutation(sequence: List[str]) -> List[str]:
    """Apply sort descending permutation: [C, A, D, B] → [D, C, B, A]"""
    return sorted(sequence, reverse=True)

def apply_shuffle_permutation(sequence: List[str], seed: Optional[int] = None) -> List[str]:
    """Apply shuffle permutation with optional seed for reproducibility"""
    result = sequence.copy()
    if seed is not None:
        random.seed(seed)
    random.shuffle(result)
    return result

# Permutation operations mapping
PERMUTATION_OPS = {
    0: ("reverse", apply_reverse_permutation),
    1: ("swap_pairs", apply_swap_pairs_permutation),
    2: ("rotate_left", apply_rotate_left_permutation),
    3: ("rotate_right", apply_rotate_right_permutation),
    4: ("sort_asc", apply_sort_ascending_permutation),
    5: ("sort_desc", apply_sort_descending_permutation),
    6: ("shuffle", apply_shuffle_permutation)
}

def apply_permutation(control_token: int, sequence: List[str], seed: Optional[int] = None) -> Tuple[List[str], str]:
    """Apply permutation based on control token"""
    if control_token not in PERMUTATION_OPS:
        raise ValueError(f"Invalid control token: {control_token}")
    
    op_name, op_func = PERMUTATION_OPS[control_token]
    
    # Special handling for shuffle operation with seed
    if control_token == 6 and seed is not None:
        result = op_func(sequence, seed)
    else:
        result = op_func(sequence)
    
    return result, op_name

def create_permutation_dataset(
    num_samples: int = 10000,
    min_seq_length: int = 3,
    max_seq_length: int = 20,
    alphabet: str = string.ascii_uppercase,
    validation_split: float = 0.1,
    test_split: float = 0.1,
    operation_distribution: Optional[Dict[int, float]] = None,
    seed: Optional[int] = 42
) -> Dict[str, List[Dict[str, Union[str, int]]]]:
    """
    Create permutation dataset with different operations
    """
    if seed is not None:
        random.seed(seed)
    
    # Default uniform distribution across all operations
    if operation_distribution is None:
        num_ops = len(PERMUTATION_OPS)
        operation_distribution = {i: 1.0/num_ops for i in range(num_ops)}
    
    # Normalize distribution
    total_weight = sum(operation_distribution.values())
    operation_distribution = {k: v/total_weight for k, v in operation_distribution.items()}
    
    samples = []
    
    for sample_idx in range(num_samples):
        # Generate random sequence
        sequence = generate_sequence(min_seq_length, max_seq_length, alphabet)
        
        # Choose operation based on distribution
        control_token = random.choices(
            list(operation_distribution.keys()),
            weights=list(operation_distribution.values())
        )[0]
        
        # Apply permutation
        try:
            # Use sample index as seed for shuffle to ensure reproducibility
            shuffle_seed = seed + sample_idx if seed is not None else None
            result_sequence, op_name = apply_permutation(control_token, sequence, shuffle_seed)
            
            # Format input and output
            input_str = str(control_token) + " " + " ".join(sequence)
            output_str = " ".join(result_sequence)
            
            samples.append({
                "translation": {
                    "src": input_str,
                    "tgt": output_str
                },
                "control_token": control_token,
                "operation": op_name,
                "original_sequence": sequence,
                "result_sequence": result_sequence
            })
            
        except Exception as e:
            print(f"Error creating sample {sample_idx}: {e}")
            continue
    
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
        op_counts = {}
        for item in split_data:
            op = item["operation"]
            op_counts[op] = op_counts.get(op, 0) + 1
        
        print(f"{split_name}: {len(split_data)} samples")
        for op, count in sorted(op_counts.items()):
            print(f"  - {op}: {count} samples")
    
    return splits

def load_permutation_dataset(
    dataset_path: Optional[Union[str, Path]] = None,
    num_samples: int = 10000,
    min_seq_length: int = 3,
    max_seq_length: int = 20,
    alphabet: str = string.ascii_uppercase,
    validation_split: float = 0.1,
    test_split: float = 0.1,
    operation_distribution: Optional[Dict[int, float]] = None,
    force_reload: bool = False,
    seed: Optional[int] = 42,
    **kwargs
) -> Dict[str, List[Dict[str, Union[str, int]]]]:
    """
    Load or create permutation dataset with caching support
    """
    
    # Create cache filename based on parameters
    op_dist_hash = hash(str(sorted(operation_distribution.items()))) if operation_distribution else 0
    cache_filename = (f"permutation_dataset_n{num_samples}_len{min_seq_length}-{max_seq_length}_"
                     f"alpha{len(alphabet)}_dist{op_dist_hash}_seed{seed}.json")
    
    if dataset_path:
        dataset_path = Path(dataset_path).resolve()
        cache_path = dataset_path / cache_filename
        
        # Try to load existing dataset
        if not force_reload and cache_path.exists():
            try:
                print(f"Loading existing permutation dataset from: {cache_path}")
                with open(cache_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Failed to load cached dataset: {e}")
                print("Generating new dataset...")
        
        # Create directory if it doesn't exist
        dataset_path.mkdir(parents=True, exist_ok=True)
    
    # Generate new dataset
    print(f"Generating permutation dataset with {num_samples} samples...")
    dataset = create_permutation_dataset(
        num_samples=num_samples,
        min_seq_length=min_seq_length,
        max_seq_length=max_seq_length,
        alphabet=alphabet,
        validation_split=validation_split,
        test_split=test_split,
        operation_distribution=operation_distribution,
        seed=seed
    )
    
    # Save dataset if path is provided
    if dataset_path:
        cache_path = dataset_path / cache_filename
        try:
            print(f"Saving permutation dataset to: {cache_path}")
            with open(cache_path, 'w') as f:
                json.dump(dataset, f, indent=2)
        except Exception as e:
            print(f"Warning: Failed to save dataset: {e}")
    
    return dataset

# Validation functions
def validate_permutation(control_token: int, original_sequence: List[str], 
                        result_sequence: List[str], seed: Optional[int] = None) -> bool:
    """Validate that permutation is correct"""
    try:
        expected_result, _ = apply_permutation(control_token, original_sequence, seed)
        return expected_result == result_sequence
    except:
        return False

def analyze_permutation_dataset(dataset: Dict[str, List]) -> Dict[str, any]:
    """Analyze permutation dataset statistics"""
    stats = {
        "total_samples": 0,
        "operations": {},
        "avg_src_length": 0,
        "avg_tgt_length": 0,
        "max_src_length": 0,
        "max_tgt_length": 0,
        "unique_characters": set(),
        "sequence_length_distribution": {}
    }
    
    all_samples = []
    for split_data in dataset.values():
        all_samples.extend(split_data)
    
    stats["total_samples"] = len(all_samples)
    
    src_lengths = []
    tgt_lengths = []
    seq_lengths = []
    
    for sample in all_samples:
        src = sample["translation"]["src"]
        tgt = sample["translation"]["tgt"]
        operation = sample["operation"]
        original_seq = sample["original_sequence"]
        
        src_lengths.append(len(src.split()))
        tgt_lengths.append(len(tgt.split()))
        seq_lengths.append(len(original_seq))
        
        # Collect unique characters
        for char in original_seq:
            stats["unique_characters"].add(char)
        
        # Count operations
        stats["operations"][operation] = stats["operations"].get(operation, 0) + 1
        
        # Sequence length distribution
        seq_len = len(original_seq)
        stats["sequence_length_distribution"][seq_len] = stats["sequence_length_distribution"].get(seq_len, 0) + 1
    
    stats["avg_src_length"] = sum(src_lengths) / len(src_lengths)
    stats["avg_tgt_length"] = sum(tgt_lengths) / len(tgt_lengths)
    stats["max_src_length"] = max(src_lengths)
    stats["max_tgt_length"] = max(tgt_lengths)
    stats["avg_sequence_length"] = sum(seq_lengths) / len(seq_lengths)
    stats["unique_characters"] = sorted(list(stats["unique_characters"]))
    
    return stats

if __name__ == "__main__":
    # Example usage
    dataset = load_permutation_dataset(
        dataset_path="./data/permutation_datasets",
        num_samples=1000,
        min_seq_length=3,
        max_seq_length=10,
        seed=42
    )
    
    # Analyze dataset
    stats = analyze_permutation_dataset(dataset)
    print("\nDataset Statistics:")
    print(f"Total samples: {stats['total_samples']}")
    print(f"Average source length: {stats['avg_src_length']:.2f}")
    print(f"Average target length: {stats['avg_tgt_length']:.2f}")
    print(f"Average sequence length: {stats['avg_sequence_length']:.2f}")
    print(f"Unique characters: {stats['unique_characters']}")
    print("\nOperation distribution:")
    for op, count in sorted(stats['operations'].items()):
        print(f"  - {op}: {count} samples")
    
    # Test some examples
    print("\nExample samples:")
    for i, sample in enumerate(dataset["train"][:5]):
        src = sample["translation"]["src"]
        tgt = sample["translation"]["tgt"]
        operation = sample["operation"]
        control_token = sample["control_token"]
        original_seq = sample["original_sequence"]
        result_seq = sample["result_sequence"]
        
        print(f"{i+1}. {operation} (token {control_token}): {src} → {tgt}")
        is_valid = validate_permutation(control_token, original_seq, result_seq)
        print(f"   Valid: {is_valid}")
        print(f"   Original: {original_seq} → Result: {result_seq}")