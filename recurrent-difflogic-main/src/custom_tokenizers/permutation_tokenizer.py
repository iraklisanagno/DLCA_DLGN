import torch
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import numpy as np
from tqdm import tqdm
from tokenizers import Tokenizer, models, trainers, pre_tokenizers
import re
from typing import List, Dict, Optional, Tuple
import os
import multiprocessing as mp
import pickle
import time
import random
from functools import partial

# Import the permutation-specific functions
from custom_datasets.permutation_dataloader import (
    apply_permutation, validate_permutation, PERMUTATION_OPS
)

# Special tokens constants for permutation tasks
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
BOS_TOKEN = "<BOS>"
EOS_TOKEN = "<EOS>"
SEP_TOKEN = "<SEP>"

# Control tokens for operations
CONTROL_TOKENS = {
    0: "REV",    # reverse
    1: "SWAP",   # swap pairs
    2: "ROTL",   # rotate left
    3: "ROTR",   # rotate right
    4: "SORTASC", # sort ascending
    5: "SORTDSC", # sort descending
    6: "SHUF"    # shuffle
}

# ===== TOP-LEVEL FUNCTIONS FOR MULTIPROCESSING =====

def permutation_tokenizer(text, level, tokenizer_obj=None):
    """Permutation-specific tokenizer that handles control tokens and sequences"""
    if level == "char":
        return list(text)
    elif level == "word":
        # Split on spaces - perfect for our space-separated format
        return text.split()
    elif level == "perm_aware":
        # Special tokenization that understands permutation format
        tokens = text.split()
        processed_tokens = []
        
        for i, token in enumerate(tokens):
            if i == 0 and token.isdigit():
                # First token is control token - map to readable form
                control_num = int(token)
                if control_num in CONTROL_TOKENS:
                    processed_tokens.append(CONTROL_TOKENS[control_num])
                else:
                    processed_tokens.append(token)
            else:
                processed_tokens.append(token)
        
        return processed_tokens
    elif level == "byte":
        return list(text.encode("utf-8"))
    elif level == "subword":
        if not tokenizer_obj:
            raise ValueError("Subword tokenizer required")
        return tokenizer_obj.encode(text).tokens
    else:
        raise ValueError(f"Unsupported tokenization level: {level}")

def process_chunk_for_perm_vocab(chunk, level):
    """Process a chunk of permutation data for vocabulary building"""
    counter = Counter()
    for ex in chunk:
        # Process both source and target
        src_tokens = permutation_tokenizer(ex["translation"]["src"], level)
        tgt_tokens = permutation_tokenizer(ex["translation"]["tgt"], level)
        counter.update(src_tokens)
        counter.update(tgt_tokens)
    return counter

def tokenize_perm_example_fn(ex, level, tokenizer_obj, seq_length):
    """Tokenize a single permutation example"""
    src_text = ex["translation"]["src"]
    tgt_text = ex["translation"]["tgt"]
    operation = ex.get("operation", "unknown")
    control_token = ex.get("control_token", -1)
    
    # Check for empty or invalid text
    if not src_text or not tgt_text or not src_text.strip() or not tgt_text.strip():
        return None
    
    src_tokens = permutation_tokenizer(src_text, level, tokenizer_obj)
    tgt_tokens = permutation_tokenizer(tgt_text, level, tokenizer_obj)
    
    # Skip if too long or too short
    if len(src_tokens) > seq_length or len(tgt_tokens) > seq_length:
        return None
    if len(src_tokens) == 0 or len(tgt_tokens) == 0:
        return None
        
    return (src_tokens, tgt_tokens, operation, control_token)

def numericalize_perm_example_fn(tokens_tuple, vocab, seq_length, shift):
    """Numericalize a tokenized permutation example"""
    if not tokens_tuple:
        return None
        
    src_tokens, tgt_tokens, operation, control_token = tokens_tuple
    
    # Source sequence
    src_ids = [vocab.get(tok, vocab[UNK_TOKEN]) for tok in src_tokens[:seq_length]]
    src_ids += [vocab[PAD_TOKEN]] * (seq_length - len(src_ids))
    
    # Target sequence with optional shift
    shifted_tokens = [PAD_TOKEN] * shift + tgt_tokens
    shifted_tokens = shifted_tokens[:seq_length]
    tgt_ids = [vocab.get(tok, vocab[UNK_TOKEN]) for tok in shifted_tokens]
    tgt_ids += [vocab[PAD_TOKEN]] * (seq_length - len(tgt_ids))
    
    # Create tensors
    src_tensor = torch.tensor(src_ids, dtype=torch.long)
    tgt_tensor = torch.tensor(tgt_ids, dtype=torch.long)
    
    # Validate tensors
    pad_idx = vocab[PAD_TOKEN]
    if (src_tensor == pad_idx).all() or (tgt_tensor == pad_idx).all():
        return None
    
    if torch.isnan(src_tensor.float()).any() or torch.isnan(tgt_tensor.float()).any():
        return None
    
    return (src_tensor, tgt_tensor, operation, control_token)

def process_perm_batch_fn(batch, level, tokenizer_obj, vocab, seq_length, shift):
    """Process a batch of permutation examples"""
    results = []
    for ex in batch:
        tokens_tuple = tokenize_perm_example_fn(ex, level, tokenizer_obj, seq_length)
        if tokens_tuple:
            item = numericalize_perm_example_fn(tokens_tuple, vocab, seq_length, shift)
            if item:
                results.append(item)
    return results

# ===== END OF TOP-LEVEL FUNCTIONS =====

class PermutationCacheManager:
    """Cache manager specifically for permutation datasets"""
    def __init__(self, cache_dir="/data/permutation_tokenizer_cache", use_disk=True):
        self.cache_dir = cache_dir
        self.use_disk = use_disk
        self.memory_cache = {}
        
        if use_disk:
            try:
                os.makedirs(cache_dir, exist_ok=True)
                print(f"Permutation Cache directory created at: {cache_dir}")
                self.disk_available = True
            except Exception as e:
                print(f"Warning: Couldn't create cache directory: {e}")
                self.disk_available = False
    
    def get_cache_key(self, dataset_name, level, vocab_size, alphabet):
        return f"perm_{dataset_name}_{level}_{vocab_size}_{hash(alphabet)}"
    
    def save_vocab(self, vocab, dataset_name, level, vocab_size, alphabet):
        cache_key = self.get_cache_key(dataset_name, level, vocab_size, alphabet)
        
        if not self.use_disk or not self.disk_available:
            self.memory_cache[f"vocab_{cache_key}"] = vocab
            return cache_key
            
        path = os.path.join(self.cache_dir, f"{cache_key}_vocab.pkl")
        try:
            with open(path, 'wb') as f:
                pickle.dump(vocab, f)
            return path
        except Exception as e:
            print(f"Warning: Failed to save vocab to disk: {e}")
            self.memory_cache[f"vocab_{cache_key}"] = vocab
            return cache_key
    
    def load_vocab(self, dataset_name, level, vocab_size, alphabet):
        cache_key = self.get_cache_key(dataset_name, level, vocab_size, alphabet)
        
        # Check memory first
        if f"vocab_{cache_key}" in self.memory_cache:
            return self.memory_cache[f"vocab_{cache_key}"]
            
        # Check disk
        if self.use_disk and self.disk_available:
            path = os.path.join(self.cache_dir, f"{cache_key}_vocab.pkl")
            if os.path.exists(path):
                try:
                    with open(path, 'rb') as f:
                        return pickle.load(f)
                except Exception as e:
                    print(f"Warning: Failed to load vocab from disk: {e}")
        
        return None

def build_perm_vocab_parallel(dataset, level, max_size=1024, pad_idx=0, unk_idx=1, 
                             sample_size=None, num_workers=None, alphabet="ABCDEFGHIJKLMNOPQRSTUVWXYZ"):
    """Build vocabulary for permutation data using parallel processing"""
    if num_workers is None:
        num_workers = min(8, max(1, mp.cpu_count() - 1))
    
    # Sample if dataset is too large
    if sample_size and len(dataset) > sample_size:
        print(f"Sampling {sample_size} examples from {len(dataset)} for permutation vocabulary building")
        dataset = random.sample(dataset, sample_size)
    
    # Split into chunks
    chunk_size = max(1, len(dataset) // num_workers)
    chunks = [dataset[i:i+chunk_size] for i in range(0, len(dataset), chunk_size)]
    
    # Process in parallel
    start_time = time.time()
    print(f"Building permutation vocab with {num_workers} workers...")
    
    with mp.Pool(num_workers) as pool:
        process_fn = partial(process_chunk_for_perm_vocab, level=level)
        counters = list(tqdm(pool.imap(process_fn, chunks), total=len(chunks), desc="Building permutation vocab"))
    
    # Combine counters
    combined_counter = Counter()
    for counter in counters:
        combined_counter.update(counter)
    
    # Create vocabulary with special tokens
    vocab = {
        PAD_TOKEN: pad_idx,
        UNK_TOKEN: unk_idx,
        BOS_TOKEN: max(pad_idx, unk_idx) + 1,
        EOS_TOKEN: max(pad_idx, unk_idx) + 2,
        SEP_TOKEN: max(pad_idx, unk_idx) + 3
    }
    
    next_index = max(vocab.values()) + 1
    
    # Add control tokens
    for control_token in CONTROL_TOKENS.values():
        vocab[control_token] = next_index
        next_index += 1
    
    # Add digit tokens (for control numbers)
    for i in range(10):
        digit_token = str(i)
        if digit_token not in vocab:
            vocab[digit_token] = next_index
            next_index += 1
    
    # Add most common tokens
    for tok, _ in combined_counter.most_common(max_size - len(vocab)):
        if tok not in vocab:
            vocab[tok] = next_index
            next_index += 1
    
    elapsed = time.time() - start_time
    print(f"Permutation vocabulary built with {len(vocab)} tokens in {elapsed:.2f}s")
    
    # Print some statistics
    control_count = sum(1 for tok in vocab.keys() if isinstance(tok, str) and tok in CONTROL_TOKENS.values())
    digits_count = sum(1 for tok in vocab.keys() if isinstance(tok, str) and tok.isdigit())
    letters_count = sum(1 for tok in vocab.keys() if isinstance(tok, str) and tok.isalpha() and len(tok) == 1)
    
    print(f"Vocab composition: {control_count} control tokens, {digits_count} digits, {letters_count} letters")
    
    return vocab

class FastPermutationDataset(Dataset):
    """Permutation-specific dataset with validation"""
    def __init__(self, data, vocab, level, seq_length, shift, 
                 tokenizer_obj=None, max_preprocess_size=1000000, num_workers=4):
        
        self.data = data
        self.vocab = vocab
        self.seq_length = seq_length
        self.level = level
        self.tokenizer_obj = tokenizer_obj
        self.shift = shift
        self.num_workers = num_workers
        self.pad_idx = vocab[PAD_TOKEN]
        
        # Streaming vs preprocessing decision
        self.streaming_mode = len(data) > max_preprocess_size
        self.preprocessed_data = None
        
        # Calculate statistics
        self._calculate_perm_stats()
        
        # Preprocess if not streaming
        if not self.streaming_mode:
            print(f"Pre-processing {len(data)} permutation examples")
            self._preprocess_data()
    
    def _calculate_perm_stats(self):
        """Calculate permutation-specific statistics"""
        sample_size = min(1000, len(self.data))
        sample_indices = random.sample(range(len(self.data)), sample_size)
        
        src_lengths = []
        tgt_lengths = []
        operation_counts = {}
        
        for idx in tqdm(sample_indices, desc="Calculating permutation statistics"):
            ex = self.data[idx]
            src_tokens = permutation_tokenizer(ex["translation"]["src"], self.level, self.tokenizer_obj)
            tgt_tokens = permutation_tokenizer(ex["translation"]["tgt"], self.level, self.tokenizer_obj)
            
            src_lengths.append(len(src_tokens))
            tgt_lengths.append(len(tgt_tokens))
            
            # Count operations
            operation = ex.get("operation", "unknown")
            operation_counts[operation] = operation_counts.get(operation, 0) + 1
        
        self.avg_src_len = np.mean(src_lengths) if src_lengths else self.seq_length
        self.avg_tgt_len = np.mean(tgt_lengths) if tgt_lengths else self.seq_length
        self.operation_distribution = operation_counts
        
        print(f"Permutation Stats - Avg src: {self.avg_src_len:.2f}, Avg tgt: {self.avg_tgt_len:.2f}")
        print(f"Operation distribution: {operation_counts}")
    
    def _preprocess_data(self):
        """Preprocess permutation data with validation"""
        batch_size = 1000
        batches = [self.data[i:i+batch_size] for i in range(0, len(self.data), batch_size)]
        
        if self.num_workers <= 1:
            results = []
            for batch in tqdm(batches, desc="Processing permutation dataset"):
                batch_results = process_perm_batch_fn(
                    batch, self.level, self.tokenizer_obj, 
                    self.vocab, self.seq_length, self.shift
                )
                results.extend(batch_results)
        else:
            with mp.Pool(self.num_workers) as pool:
                process_fn = partial(
                    process_perm_batch_fn,
                    level=self.level,
                    tokenizer_obj=self.tokenizer_obj,
                    vocab=self.vocab,
                    seq_length=self.seq_length,
                    shift=self.shift
                )
                
                results = []
                for batch_result in tqdm(pool.imap(process_fn, batches), 
                                       total=len(batches), 
                                       desc="Processing permutation dataset"):
                    results.extend(batch_result)
        
        # Validate results
        valid_results = []
        for item in results:
            if item is not None:
                src_tensor, tgt_tensor, operation, control_token = item
                # Additional permutation validation could go here
                valid_results.append(item)
        
        self.preprocessed_data = valid_results
        print(f"Pre-processed {len(self.preprocessed_data)} valid permutation examples")

    def __len__(self):
        if self.preprocessed_data is not None:
            return len(self.preprocessed_data)
        return len(self.data)

    def __getitem__(self, idx):
        if self.preprocessed_data is not None:
            return self.preprocessed_data[idx]
        
        # On-the-fly processing
        ex = self.data[idx]
        tokens_tuple = tokenize_perm_example_fn(
            ex, self.level, self.tokenizer_obj, self.seq_length
        )
        
        if tokens_tuple is None:
            return None
            
        result = numericalize_perm_example_fn(
            tokens_tuple, self.vocab, self.seq_length, self.shift
        )
        
        return result

def perm_filter_collate(batch, pad_idx=0):
    """Enhanced collate function for permutation data"""
    # Filter out None values
    batch = [item for item in batch if item is not None]
    
    if len(batch) == 0:
        return None
    
    if len(batch) < 2:
        print(f"⚠️  Permutation: Filtering out batch with only {len(batch)} samples")
        return None
    
    try:
        # Separate by operation type if needed
        operation_groups = {}
        for item in batch:
            src_tensor, tgt_tensor, operation, control_token = item
            if operation not in operation_groups:
                operation_groups[operation] = []
            operation_groups[operation].append((src_tensor, tgt_tensor))
        
        # For now, collate all together regardless of operation
        all_items = [(src, tgt) for src, tgt, _, _ in batch]
        
        if len(all_items) == 0:
            return None
        
        collated = torch.utils.data.dataloader.default_collate(all_items)
        src_batch, tgt_batch = collated
        
        # Validate batch
        valid_indices = []
        for i in range(src_batch.size(0)):
            src_sample = src_batch[i]
            tgt_sample = tgt_batch[i]
            
            src_valid = (src_sample != pad_idx).any()
            tgt_valid = (tgt_sample != pad_idx).any()
            
            if src_valid and tgt_valid:
                valid_indices.append(i)
        
        if len(valid_indices) == 0:
            return None
        
        valid_indices = torch.tensor(valid_indices)
        return (src_batch[valid_indices], tgt_batch[valid_indices])
        
    except Exception as e:
        print(f"⚠️  Error in permutation collate function: {e}")
        return None

def permutation_tokenizer_pipeline(dataset, dataset_name="permutation", tk_level="perm_aware", 
                                  tokens_per_batch=25000, seq_length=64,
                                  max_vocab_size=10000, pad_idx=0, unk_idx=1, shift=0,
                                  cache_dir="/data/permutation_tokenizer_cache", use_disk_cache=True,
                                  vocab_sample_size=500000, max_preprocess_size=500000,
                                  alphabet="ABCDEFGHIJKLMNOPQRSTUVWXYZ"):
    """
    Permutation tokenization pipeline optimized for permutation tasks
    """
    # Initialize cache manager
    cache_manager = PermutationCacheManager(cache_dir=cache_dir, use_disk=use_disk_cache)
    
    # Set CPU count
    num_cpus = min(7, max(1, mp.cpu_count() - 1))
    print(f"Permutation tokenizer using {num_cpus} CPU cores")
    
    # Convert dataset to lists
    splits = {k: list(v) for k, v in dataset.items()}
    
    # Try to load cached vocabulary
    vocab = cache_manager.load_vocab(dataset_name, tk_level, max_vocab_size, alphabet)
    
    if vocab is None:
        print("Building new permutation vocabulary...")
        vocab = build_perm_vocab_parallel(
            splits['train'], tk_level, max_vocab_size, pad_idx, unk_idx,
            sample_size=vocab_sample_size, num_workers=num_cpus, alphabet=alphabet
        )
        
        # Save vocabulary
        cache_manager.save_vocab(vocab, dataset_name, tk_level, max_vocab_size, alphabet)
    else:
        print("Using cached permutation vocabulary")
    
    # Create datasets and dataloaders
    datasets = {}
    dataloaders = {}
    batch_sizes = {}
    
    for split_name in ['train', 'validation', 'test']:
        current_max_preprocess = max_preprocess_size
        if split_name != 'train':
            current_max_preprocess = max_preprocess_size * 10
        
        datasets[split_name] = FastPermutationDataset(
            splits[split_name],
            vocab, tk_level, seq_length, shift,
            tokenizer_obj=None,  # Permutation doesn't use subword tokenizers
            max_preprocess_size=current_max_preprocess,
            num_workers=num_cpus
        )
        
        # Calculate batch size
        avg_len = max(datasets[split_name].avg_src_len, datasets[split_name].avg_tgt_len)
        batch_size = max(1, int(tokens_per_batch / avg_len))
        batch_sizes[split_name] = batch_size
        
        # Create dataloader
        num_workers = 0 if datasets[split_name].streaming_mode else 2
        
        collate_fn = partial(perm_filter_collate, pad_idx=pad_idx)
        
        dataloaders[split_name] = DataLoader(
            datasets[split_name],
            batch_size=batch_size,
            shuffle=(split_name == 'train'),
            pin_memory=True,
            num_workers=num_workers,
            collate_fn=collate_fn,
            persistent_workers=num_workers > 0
        )
    
    # Build reverse vocabulary
    id_to_token = {idx: tok for tok, idx in vocab.items()}
    
    # Print summary
    print(f"Permutation tokenization completed for {tk_level} level")
    print(f"Vocabulary size: {len(vocab)}")
    print(f"Dynamic batch sizes:")
    for split, batch_size in batch_sizes.items():
        streaming_status = "streaming" if datasets[split].streaming_mode else "precomputed"
        print(f"  - {split}: {batch_size} examples "
              f"(avg len: {max(datasets[split].avg_src_len, datasets[split].avg_tgt_len):.1f}, "
              f"{streaming_status})")
    
    return {
        'model_interface': {
            'num_input_tokens': len(vocab),
            'num_classes': len(vocab),
            'vocab': vocab,
            'id_to_token': id_to_token,
            'padding_idx': pad_idx,
            'unk_idx': unk_idx,
            'seq_length': seq_length,
            'shift': shift,
            'tokenization_level': tk_level,
            'avg_src_len': {split: datasets[split].avg_src_len for split in splits},
            'avg_tgt_len': {split: datasets[split].avg_tgt_len for split in splits},
            'operation_distributions': {split: datasets[split].operation_distribution for split in splits},
            'batch_sizes': batch_sizes,
            'alphabet': alphabet,
            'control_tokens': CONTROL_TOKENS
        },
        'train': dataloaders['train'],
        'validation': dataloaders['validation'],
        'test': dataloaders['test'],
    }

# Utility functions for evaluation
def decode_permutation_prediction(prediction_ids, id_to_token, remove_special_tokens=True):
    """Decode permutation prediction back to text"""
    tokens = []
    for token_id in prediction_ids:
        if token_id in id_to_token:
            token = id_to_token[token_id]
            if remove_special_tokens and token in [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN]:
                continue
            tokens.append(token)
    
    return " ".join(tokens)

def evaluate_permutation_predictions(src_batch, tgt_batch, pred_batch, id_to_token):
    """Evaluate permutation predictions for correctness"""
    results = {
        'exact_matches': 0,
        'valid_permutations': 0,
        'correct_operations': 0,
        'total': len(src_batch)
    }
    
    for i in range(len(src_batch)):
        src_text = decode_permutation_prediction(src_batch[i], id_to_token)
        tgt_text = decode_permutation_prediction(tgt_batch[i], id_to_token)
        pred_text = decode_permutation_prediction(pred_batch[i], id_to_token)
        
        # Check exact match
        if pred_text == tgt_text:
            results['exact_matches'] += 1
        
        # Parse source to get control token and original sequence
        try:
            src_parts = src_text.split()
            if len(src_parts) >= 2:
                control_part = src_parts[0]
                original_sequence = src_parts[1:]
                
                # Map control token back to number
                control_token = None
                for num, token_name in CONTROL_TOKENS.items():
                    if control_part == token_name or control_part == str(num):
                        control_token = num
                        break
                
                if control_token is not None:
                    # Check if prediction is a valid permutation
                    pred_sequence = pred_text.split()
                    if len(pred_sequence) == len(original_sequence):
                        results['valid_permutations'] += 1
                        
                        # Check if it's the correct operation
                        try:
                            expected_result, _ = apply_permutation(control_token, original_sequence)
                            if pred_sequence == expected_result:
                                results['correct_operations'] += 1
                        except:
                            pass
        except:
            pass
    
    return results

if __name__ == "__main__":
    # Example usage
    from custom_datasets.permutation_dataloader import load_permutation_dataset
    
    # Load permutation dataset
    dataset = load_permutation_dataset(
        dataset_path="./data/permutation_datasets",
        num_samples=1000,
        min_seq_length=3,
        max_seq_length=10,
        seed=42
    )
    
    # Tokenize
    tokenized = permutation_tokenizer_pipeline(
        dataset,
        dataset_name="permutation_test",
        tk_level="perm_aware",
        seq_length=64,
        max_vocab_size=1000
    )
    
    print("Permutation tokenization complete!")
    print(f"Vocabulary size: {tokenized['model_interface']['num_input_tokens']}")
    print(f"Control tokens: {tokenized['model_interface']['control_tokens']}")
    
    # Test a batch
    train_loader = tokenized['train']
    for batch in train_loader:
        if batch is not None:
            src, tgt = batch
            print(f"Batch shape: src={src.shape}, tgt={tgt.shape}")
            
            # Decode first example
            src_text = decode_permutation_prediction(src[0], tokenized['model_interface']['id_to_token'])
            tgt_text = decode_permutation_prediction(tgt[0], tokenized['model_interface']['id_to_token'])
            print(f"Example: {src_text} → {tgt_text}")
            break