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

# Import the RLE-specific functions
from custom_datasets.rle_dataloader import sequence_to_rle, rle_to_sequence, validate_rle_conversion

# Special tokens constants for RLE
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
BOS_TOKEN = "<BOS>"
EOS_TOKEN = "<EOS>"
SEP_TOKEN = "<SEP>"  # For separating compression/decompression tasks

# ===== TOP-LEVEL FUNCTIONS FOR MULTIPROCESSING =====

def rle_tokenizer(text, level, tokenizer_obj=None):
    """RLE-specific tokenizer that handles digits and characters appropriately"""
    if level == "char":
        return list(text)
    elif level == "word":
        # For RLE, we want to split on character boundaries but keep digit-char pairs together
        tokens = []
        i = 0
        while i < len(text):
            if text[i].isdigit():
                # Collect consecutive digits
                num = ""
                while i < len(text) and text[i].isdigit():
                    num += text[i]
                    i += 1
                tokens.append(num)
            else:
                tokens.append(text[i])
                i += 1
        return tokens
    elif level == "rle_aware":
        # Special tokenization that understands RLE patterns
        tokens = []
        i = 0
        while i < len(text):
            if text[i].isdigit():
                # Collect number and following character as a unit
                num = ""
                while i < len(text) and text[i].isdigit():
                    num += text[i]
                    i += 1
                if i < len(text) and text[i].isalpha():
                    tokens.append(num + text[i])
                    i += 1
                else:
                    tokens.append(num)
            else:
                tokens.append(text[i])
                i += 1
        return tokens
    elif level == "byte":
        return list(text.encode("utf-8"))
    elif level == "subword":
        if not tokenizer_obj:
            raise ValueError("Subword tokenizer required")
        return tokenizer_obj.encode(text).tokens
    else:
        raise ValueError(f"Unsupported tokenization level: {level}")

def process_chunk_for_rle_vocab(chunk, level):
    """Process a chunk of RLE data for vocabulary building"""
    counter = Counter()
    for ex in chunk:
        # Process both source and target
        src_tokens = rle_tokenizer(ex["translation"]["src"], level)
        tgt_tokens = rle_tokenizer(ex["translation"]["tgt"], level)
        counter.update(src_tokens)
        counter.update(tgt_tokens)
    return counter

def tokenize_rle_example_fn(ex, level, tokenizer_obj, seq_length):
    """Tokenize a single RLE example"""
    src_text = ex["translation"]["src"]
    tgt_text = ex["translation"]["tgt"]
    task_type = ex.get("task_type", "unknown")
    
    # Check for empty or invalid text
    if not src_text or not tgt_text or not src_text.strip() or not tgt_text.strip():
        return None
    
    # Add task type information as a prefix token
    task_prefix = "COMP:" if task_type == "compression" else "DECOMP:"
    src_text_with_task = task_prefix + src_text
    
    src_tokens = rle_tokenizer(src_text_with_task, level, tokenizer_obj)
    tgt_tokens = rle_tokenizer(tgt_text, level, tokenizer_obj)
    
    # Skip if too long or too short
    if len(src_tokens) > seq_length or len(tgt_tokens) > seq_length:
        return None
    if len(src_tokens) == 0 or len(tgt_tokens) == 0:
        return None
        
    return (src_tokens, tgt_tokens, task_type)

def numericalize_rle_example_fn(tokens_tuple, vocab, seq_length, shift):
    """Numericalize a tokenized RLE example"""
    if not tokens_tuple:
        return None
        
    src_tokens, tgt_tokens, task_type = tokens_tuple
    
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
    
    return (src_tensor, tgt_tensor, task_type)

def process_rle_batch_fn(batch, level, tokenizer_obj, vocab, seq_length, shift):
    """Process a batch of RLE examples"""
    results = []
    for ex in batch:
        tokens_tuple = tokenize_rle_example_fn(ex, level, tokenizer_obj, seq_length)
        if tokens_tuple:
            item = numericalize_rle_example_fn(tokens_tuple, vocab, seq_length, shift)
            if item:
                results.append(item)
    return results

# ===== END OF TOP-LEVEL FUNCTIONS =====

class RLECacheManager:
    """Cache manager specifically for RLE datasets"""
    def __init__(self, cache_dir="/data/rle_tokenizer_cache", use_disk=True):
        self.cache_dir = cache_dir
        self.use_disk = use_disk
        self.memory_cache = {}
        
        if use_disk:
            try:
                os.makedirs(cache_dir, exist_ok=True)
                print(f"RLE Cache directory created at: {cache_dir}")
                self.disk_available = True
            except Exception as e:
                print(f"Warning: Couldn't create cache directory: {e}")
                self.disk_available = False
    
    def get_cache_key(self, dataset_name, level, vocab_size, alphabet):
        return f"rle_{dataset_name}_{level}_{vocab_size}_{hash(alphabet)}"
    
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

def build_rle_vocab_parallel(dataset, level, max_size=1024, pad_idx=0, unk_idx=1, 
                            sample_size=None, num_workers=None, alphabet="ABCDEFGHIJKLMNOPQRSTUVWXYZ"):
    """Build vocabulary for RLE data using parallel processing"""
    if num_workers is None:
        num_workers = min(8, max(1, mp.cpu_count() - 1))
    
    # Sample if dataset is too large
    if sample_size and len(dataset) > sample_size:
        print(f"Sampling {sample_size} examples from {len(dataset)} for RLE vocabulary building")
        dataset = random.sample(dataset, sample_size)
    
    # Split into chunks
    chunk_size = max(1, len(dataset) // num_workers)
    chunks = [dataset[i:i+chunk_size] for i in range(0, len(dataset), chunk_size)]
    
    # Process in parallel
    start_time = time.time()
    print(f"Building RLE vocab with {num_workers} workers...")
    
    with mp.Pool(num_workers) as pool:
        process_fn = partial(process_chunk_for_rle_vocab, level=level)
        counters = list(tqdm(pool.imap(process_fn, chunks), total=len(chunks), desc="Building RLE vocab"))
    
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
    
    # Add task prefixes
    vocab["COMP:"] = max(vocab.values()) + 1
    vocab["DECOMP:"] = max(vocab.values()) + 1
    
    next_index = max(vocab.values()) + 1
    
    # Add most common tokens
    for tok, _ in combined_counter.most_common(max_size - len(vocab)):
        if tok not in vocab:
            vocab[tok] = next_index
            next_index += 1
    
    elapsed = time.time() - start_time
    print(f"RLE vocabulary built with {len(vocab)} tokens in {elapsed:.2f}s")
    
    # Print some statistics
    digits_count = sum(1 for tok in vocab.keys() if isinstance(tok, str) and tok.isdigit())
    letters_count = sum(1 for tok in vocab.keys() if isinstance(tok, str) and tok.isalpha() and len(tok) == 1)
    rle_patterns_count = sum(1 for tok in vocab.keys() if isinstance(tok, str) and 
                           len(tok) > 1 and tok[:-1].isdigit() and tok[-1].isalpha())
    
    print(f"Vocab composition: {digits_count} digits, {letters_count} letters, {rle_patterns_count} RLE patterns")
    
    return vocab

class FastRLEDataset(Dataset):
    """RLE-specific dataset with validation"""
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
        self._calculate_rle_stats()
        
        # Preprocess if not streaming
        if not self.streaming_mode:
            print(f"Pre-processing {len(data)} RLE examples")
            self._preprocess_data()
    
    def _calculate_rle_stats(self):
        """Calculate RLE-specific statistics"""
        sample_size = min(1000, len(self.data))
        sample_indices = random.sample(range(len(self.data)), sample_size)
        
        src_lengths = []
        tgt_lengths = []
        compression_ratios = []
        
        for idx in tqdm(sample_indices, desc="Calculating RLE statistics"):
            ex = self.data[idx]
            src_tokens = rle_tokenizer(ex["translation"]["src"], self.level, self.tokenizer_obj)
            tgt_tokens = rle_tokenizer(ex["translation"]["tgt"], self.level, self.tokenizer_obj)
            
            src_lengths.append(len(src_tokens))
            tgt_lengths.append(len(tgt_tokens))
            
            # Calculate compression ratio for this example
            if ex["task_type"] == "compression":
                ratio = len(tgt_tokens) / len(src_tokens) if len(src_tokens) > 0 else 1.0
            else:
                ratio = len(src_tokens) / len(tgt_tokens) if len(tgt_tokens) > 0 else 1.0
            compression_ratios.append(ratio)
        
        self.avg_src_len = np.mean(src_lengths) if src_lengths else self.seq_length
        self.avg_tgt_len = np.mean(tgt_lengths) if tgt_lengths else self.seq_length
        self.avg_compression_ratio = np.mean(compression_ratios) if compression_ratios else 0.5
        
        print(f"RLE Stats - Avg src: {self.avg_src_len:.2f}, Avg tgt: {self.avg_tgt_len:.2f}, "
              f"Compression ratio: {self.avg_compression_ratio:.2f}")
    
    def _preprocess_data(self):
        """Preprocess RLE data with validation"""
        batch_size = 1000
        batches = [self.data[i:i+batch_size] for i in range(0, len(self.data), batch_size)]
        
        if self.num_workers <= 1:
            results = []
            for batch in tqdm(batches, desc="Processing RLE dataset"):
                batch_results = process_rle_batch_fn(
                    batch, self.level, self.tokenizer_obj, 
                    self.vocab, self.seq_length, self.shift
                )
                results.extend(batch_results)
        else:
            with mp.Pool(self.num_workers) as pool:
                process_fn = partial(
                    process_rle_batch_fn,
                    level=self.level,
                    tokenizer_obj=self.tokenizer_obj,
                    vocab=self.vocab,
                    seq_length=self.seq_length,
                    shift=self.shift
                )
                
                results = []
                for batch_result in tqdm(pool.imap(process_fn, batches), 
                                       total=len(batches), 
                                       desc="Processing RLE dataset"):
                    results.extend(batch_result)
        
        # Validate results
        valid_results = []
        for item in results:
            if item is not None:
                src_tensor, tgt_tensor, task_type = item
                # Additional RLE validation could go here
                valid_results.append(item)
        
        self.preprocessed_data = valid_results
        print(f"Pre-processed {len(self.preprocessed_data)} valid RLE examples")

    def __len__(self):
        if self.preprocessed_data is not None:
            return len(self.preprocessed_data)
        return len(self.data)

    def __getitem__(self, idx):
        if self.preprocessed_data is not None:
            return self.preprocessed_data[idx]
        
        # On-the-fly processing
        ex = self.data[idx]
        tokens_tuple = tokenize_rle_example_fn(
            ex, self.level, self.tokenizer_obj, self.seq_length
        )
        
        if tokens_tuple is None:
            return None
            
        result = numericalize_rle_example_fn(
            tokens_tuple, self.vocab, self.seq_length, self.shift
        )
        
        return result

def rle_filter_collate(batch, pad_idx=0):
    """Enhanced collate function for RLE data"""
    # Filter out None values
    batch = [item for item in batch if item is not None]
    
    if len(batch) == 0:
        return None
    
    if len(batch) < 2:
        print(f"⚠️  RLE: Filtering out batch with only {len(batch)} samples")
        return None
    
    try:
        # Separate task types
        compression_items = []
        decompression_items = []
        
        for item in batch:
            src_tensor, tgt_tensor, task_type = item
            if task_type == "compression":
                compression_items.append((src_tensor, tgt_tensor))
            else:
                decompression_items.append((src_tensor, tgt_tensor))
        
        # Collate each task type separately if needed, or together
        all_items = [(src, tgt) for src, tgt, _ in batch]
        
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
        print(f"⚠️  Error in RLE collate function: {e}")
        return None

def rle_tokenizer_pipeline(dataset, dataset_name="rle", tk_level="rle_aware", 
                          tokens_per_batch=25000, seq_length=64,
                          max_vocab_size=10000, pad_idx=0, unk_idx=1, shift=0,
                          cache_dir="/data/rle_tokenizer_cache", use_disk_cache=True,
                          vocab_sample_size=500000, max_preprocess_size=500000,
                          alphabet="ABCDEFGHIJKLMNOPQRSTUVWXYZ"):
    """
    RLE tokenization pipeline optimized for compression/decompression tasks
    """
    # Initialize cache manager
    cache_manager = RLECacheManager(cache_dir=cache_dir, use_disk=use_disk_cache)
    
    # Set CPU count
    num_cpus = min(7, max(1, mp.cpu_count() - 1))
    print(f"RLE tokenizer using {num_cpus} CPU cores")
    
    # Convert dataset to lists
    splits = {k: list(v) for k, v in dataset.items()}
    
    # Try to load cached vocabulary
    vocab = cache_manager.load_vocab(dataset_name, tk_level, max_vocab_size, alphabet)
    
    if vocab is None:
        print("Building new RLE vocabulary...")
        vocab = build_rle_vocab_parallel(
            splits['train'], tk_level, max_vocab_size, pad_idx, unk_idx,
            sample_size=vocab_sample_size, num_workers=num_cpus, alphabet=alphabet
        )
        
        # Save vocabulary
        cache_manager.save_vocab(vocab, dataset_name, tk_level, max_vocab_size, alphabet)
    else:
        print("Using cached RLE vocabulary")
    
    # Create datasets and dataloaders
    datasets = {}
    dataloaders = {}
    batch_sizes = {}
    
    for split_name in ['train', 'validation', 'test']:
        current_max_preprocess = max_preprocess_size
        if split_name != 'train':
            current_max_preprocess = max_preprocess_size * 10
        
        datasets[split_name] = FastRLEDataset(
            splits[split_name],
            vocab, tk_level, seq_length, shift,
            tokenizer_obj=None,  # RLE doesn't use subword tokenizers
            max_preprocess_size=current_max_preprocess,
            num_workers=num_cpus
        )
        
        # Calculate batch size
        avg_len = max(datasets[split_name].avg_src_len, datasets[split_name].avg_tgt_len)
        batch_size = max(1, int(tokens_per_batch / avg_len))
        batch_sizes[split_name] = batch_size
        
        # Create dataloader
        num_workers = 0 if datasets[split_name].streaming_mode else 2
        
        collate_fn = partial(rle_filter_collate, pad_idx=pad_idx)
        
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
    print(f"RLE tokenization completed for {tk_level} level")
    print(f"Vocabulary size: {len(vocab)}")
    print(f"Dynamic batch sizes:")
    for split, batch_size in batch_sizes.items():
        streaming_status = "streaming" if datasets[split].streaming_mode else "precomputed"
        print(f"  - {split}: {batch_size} examples "
              f"(avg len: {max(datasets[split].avg_src_len, datasets[split].avg_tgt_len):.1f}, "
              f"compression ratio: {datasets[split].avg_compression_ratio:.2f}, {streaming_status})")
    
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
            'compression_ratios': {split: datasets[split].avg_compression_ratio for split in splits},
            'batch_sizes': batch_sizes,
            'alphabet': alphabet
        },
        'train': dataloaders['train'],
        'validation': dataloaders['validation'],
        'test': dataloaders['test'],
    }

# Utility functions for evaluation
def decode_rle_prediction(prediction_ids, id_to_token, remove_special_tokens=True):
    """Decode RLE prediction back to text"""
    tokens = []
    for token_id in prediction_ids:
        if token_id in id_to_token:
            token = id_to_token[token_id]
            if remove_special_tokens and token in [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN]:
                continue
            tokens.append(token)
    
    return "".join(tokens)

def evaluate_rle_predictions(src_batch, tgt_batch, pred_batch, id_to_token):
    """Evaluate RLE predictions for correctness"""
    results = {
        'exact_matches': 0,
        'valid_rle_format': 0,
        'valid_round_trip': 0,
        'total': len(src_batch)
    }
    
    for i in range(len(src_batch)):
        src_text = decode_rle_prediction(src_batch[i], id_to_token)
        tgt_text = decode_rle_prediction(tgt_batch[i], id_to_token)
        pred_text = decode_rle_prediction(pred_batch[i], id_to_token)
        
        # Check exact match
        if pred_text == tgt_text:
            results['exact_matches'] += 1
        
        # Check if prediction is valid RLE format
        try:
            if "COMP:" in src_text:
                # Compression task: check if we can decompress the prediction
                test_seq = rle_to_sequence(pred_text)
                results['valid_rle_format'] += 1
            elif "DECOMP:" in src_text:
                # Decompression task: check if we can compress the prediction
                test_rle = sequence_to_rle(pred_text)
                results['valid_rle_format'] += 1
        except:
            pass
        
        # Check round-trip conversion
        try:
            if "COMP:" in src_text:
                original_seq = src_text.replace("COMP:", "")
                if validate_rle_conversion(original_seq, pred_text):
                    results['valid_round_trip'] += 1
            elif "DECOMP:" in src_text:
                original_rle = src_text.replace("DECOMP:", "")
                if validate_rle_conversion(pred_text, original_rle):
                    results['valid_round_trip'] += 1
        except:
            pass
    
    return results

if __name__ == "__main__":
    # Example usage
    from custom_datasets.rle_dataloader import load_rle_dataset
    
    # Load RLE dataset
    dataset = load_rle_dataset(
        dataset_path="./data/rle_datasets",
        num_samples=1000,
        min_seq_length=5,
        max_seq_length=30,
        seed=42
    )
    
    # Tokenize
    tokenized = rle_tokenizer_pipeline(
        dataset,
        dataset_name="rle_test",
        tk_level="rle_aware",
        seq_length=64,
        max_vocab_size=1000
    )
    
    print("RLE tokenization complete!")
    print(f"Vocabulary size: {tokenized['model_interface']['num_input_tokens']}")
    
    # Test a batch
    train_loader = tokenized['train']
    for batch in train_loader:
        if batch is not None:
            src, tgt = batch
            print(f"Batch shape: src={src.shape}, tgt={tgt.shape}")
            break