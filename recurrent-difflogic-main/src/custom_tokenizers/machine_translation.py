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

# Special tokens constants for consistency
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
BOS_TOKEN = "<BOS>"
EOS_TOKEN = "<EOS>"

# ===== DEFINE TOP-LEVEL FUNCTIONS FOR MULTIPROCESSING =====
# Functions must be at module level to be picklable for multiprocessing

def simple_tokenizer(text, level, tokenizer_obj=None):
    if level == "char":
        return list(text)
    elif level == "word":
        return re.findall(r"\w+|\S", text.strip())
    elif level == "byte":
        return list(text.encode("utf-8"))
    elif level == "subword":
        if not tokenizer_obj:
            raise ValueError("Subword tokenizer required")
        return tokenizer_obj.encode(text).tokens
    else:
        raise ValueError(f"Unsupported tokenization level: {level}")

def process_chunk_for_vocab(chunk, lang_key, level):
    """Process a chunk of data for vocabulary building"""
    counter = Counter()
    for ex in chunk:
        tokens = simple_tokenizer(ex["translation"][lang_key], level)
        counter.update(tokens)
    return counter

def tokenize_example_fn(ex, src_lang, tgt_lang, level, src_tok, tgt_tok, seq_length):
    """Tokenize a single example (must be top-level for pickling)"""
    src_text = ex["translation"][src_lang]
    tgt_text = ex["translation"][tgt_lang]
    
    # Check for empty or invalid text
    if not src_text or not tgt_text or not src_text.strip() or not tgt_text.strip():
        return None
    
    src_tokens = simple_tokenizer(src_text, level, src_tok)
    tgt_tokens = simple_tokenizer(tgt_text, level, tgt_tok)
    
    # Skip if too long or too short
    if len(src_tokens) > seq_length or len(tgt_tokens) > seq_length:
        return None
    if len(src_tokens) == 0 or len(tgt_tokens) == 0:
        return None
        
    return (src_tokens, tgt_tokens)

def numericalize_example_fn(tokens_pair, src_vocab, tgt_vocab, seq_length, shift):
    """Numericalize a tokenized example (must be top-level for pickling)"""
    if not tokens_pair:
        return None
        
    src_tokens, tgt_tokens = tokens_pair
    
    src_ids = [src_vocab.get(tok, src_vocab[UNK_TOKEN]) for tok in src_tokens[:seq_length]]
    src_ids += [src_vocab[PAD_TOKEN]] * (seq_length - len(src_ids))
    
    shifted_tokens = [PAD_TOKEN] * shift + tgt_tokens
    shifted_tokens = shifted_tokens[:seq_length]
    tgt_ids = [tgt_vocab.get(tok, tgt_vocab[UNK_TOKEN]) for tok in shifted_tokens]
    tgt_ids += [tgt_vocab[PAD_TOKEN]] * (seq_length - len(tgt_ids))
    
    # CRITICAL: Validate that we have non-padding tokens
    src_tensor = torch.tensor(src_ids, dtype=torch.long)
    tgt_tensor = torch.tensor(tgt_ids, dtype=torch.long)
    
    # Check for valid content (at least some non-padding tokens)
    pad_idx = src_vocab[PAD_TOKEN]
    if (src_tensor == pad_idx).all() or (tgt_tensor == pad_idx).all():
        return None  # Skip all-padding sequences
    
    # Check for NaN or invalid values
    if torch.isnan(src_tensor.float()).any() or torch.isnan(tgt_tensor.float()).any():
        return None
    
    return (src_tensor, tgt_tensor)

def process_batch_fn(batch, src_lang, tgt_lang, level, src_tok, tgt_tok, src_vocab, tgt_vocab, seq_length, shift):
    """Process a batch of examples (must be top-level for pickling)"""
    results = []
    for ex in batch:
        tokens_pair = tokenize_example_fn(ex, src_lang, tgt_lang, level, src_tok, tgt_tok, seq_length)
        if tokens_pair:  # Skip too long sequences
            item = numericalize_example_fn(tokens_pair, src_vocab, tgt_vocab, seq_length, shift)
            if item:
                results.append(item)
    return results

# ===== END OF TOP-LEVEL FUNCTIONS =====

class CacheManager:
    """Memory and disk cache manager with container-friendly paths"""
    def __init__(self, cache_dir="/data/tokenizer_cache", use_disk=True):
        self.cache_dir = cache_dir
        self.use_disk = use_disk
        self.memory_cache = {}
        
        # Create cache directory only if disk caching is enabled
        if use_disk:
            try:
                os.makedirs(cache_dir, exist_ok=True)
                print(f"Cache directory created at: {cache_dir}")
                self.disk_available = True
            except Exception as e:
                print(f"Warning: Couldn't create cache directory: {e}")
                print("Falling back to memory-only caching")
                self.disk_available = False
    
    def get_cache_path(self, dataset_name, split, lang, level, vocab_size):
        return os.path.join(self.cache_dir, f"{dataset_name}_{split}_{lang}_{level}_{vocab_size}.pkl")
    
    def save_tokenizer(self, tokenizer, dataset_name, lang, level, vocab_size):
        """Save tokenizer to disk or memory"""
        if not self.use_disk or not self.disk_available:
            cache_key = f"tokenizer_{dataset_name}_{lang}_{level}_{vocab_size}"
            self.memory_cache[cache_key] = tokenizer
            return cache_key
            
        path = os.path.join(self.cache_dir, f"{dataset_name}_{lang}_{level}_{vocab_size}_tokenizer.json")
        try:
            tokenizer.save(path)
            return path
        except Exception as e:
            print(f"Warning: Failed to save tokenizer to disk: {e}")
            # Fall back to memory caching
            cache_key = f"tokenizer_{dataset_name}_{lang}_{level}_{vocab_size}"
            self.memory_cache[cache_key] = tokenizer
            return cache_key
    
    def load_tokenizer(self, dataset_name, lang, level, vocab_size):
        """Try to load tokenizer from disk or memory"""
        # Check memory cache first
        cache_key = f"tokenizer_{dataset_name}_{lang}_{level}_{vocab_size}"
        if cache_key in self.memory_cache:
            return self.memory_cache[cache_key]
            
        # Then try disk if available
        if self.use_disk and self.disk_available:
            path = os.path.join(self.cache_dir, f"{dataset_name}_{lang}_{level}_{vocab_size}_tokenizer.json")
            if os.path.exists(path):
                try:
                    return Tokenizer.from_file(path)
                except Exception as e:
                    print(f"Warning: Failed to load tokenizer from disk: {e}")
        
        return None
    
    def save_vocab(self, vocab, dataset_name, lang, level, vocab_size):
        """Save vocabulary to disk or memory"""
        if not self.use_disk or not self.disk_available:
            cache_key = f"vocab_{dataset_name}_{lang}_{level}_{vocab_size}"
            self.memory_cache[cache_key] = vocab
            return cache_key
            
        path = os.path.join(self.cache_dir, f"{dataset_name}_{lang}_{level}_{vocab_size}_vocab.pkl")
        try:
            with open(path, 'wb') as f:
                pickle.dump(vocab, f)
            return path
        except Exception as e:
            print(f"Warning: Failed to save vocab to disk: {e}")
            # Fall back to memory caching
            cache_key = f"vocab_{dataset_name}_{lang}_{level}_{vocab_size}"
            self.memory_cache[cache_key] = vocab
            return cache_key
    
    def load_vocab(self, dataset_name, lang, level, vocab_size):
        """Try to load vocab from disk or memory"""
        # Check memory cache first
        cache_key = f"vocab_{dataset_name}_{lang}_{level}_{vocab_size}"
        if cache_key in self.memory_cache:
            return self.memory_cache[cache_key]
            
        # Then try disk if available
        if self.use_disk and self.disk_available:
            path = os.path.join(self.cache_dir, f"{dataset_name}_{lang}_{level}_{vocab_size}_vocab.pkl")
            if os.path.exists(path):
                try:
                    with open(path, 'rb') as f:
                        return pickle.load(f)
                except Exception as e:
                    print(f"Warning: Failed to load vocab from disk: {e}")
        
        return None

def build_vocab_parallel(dataset, lang_key, level, max_size=1024, pad_idx=0, unk_idx=1, 
                        sample_size=None, num_workers=None, verbose=True):
    """Build vocabulary using safe single-process approach for containers"""
    # FIXED: Use single worker to prevent multiprocessing issues in containers
    if num_workers is None:
        num_workers = 1  # Always use single process for safety
    
    # Sampling for very large datasets
    if sample_size and len(dataset) > sample_size:
        if verbose:
            print(f"Sampling {sample_size} examples from {len(dataset)} for vocabulary building")
        dataset = random.sample(dataset, sample_size)
    
    # Process all data in single process to avoid multiprocessing issues
    if verbose:
        print(f"Building {lang_key} vocab with single worker (container-safe mode)...")
    
    start_time = time.time()
    combined_counter = Counter()
    
    # Process data in chunks for progress reporting
    chunk_size = max(1000, len(dataset) // 10)
    chunks = [dataset[i:i+chunk_size] for i in range(0, len(dataset), chunk_size)]
    
    for i, chunk in enumerate(chunks):
        counter = process_chunk_for_vocab(chunk, lang_key, level)
        combined_counter.update(counter)
        
        if verbose and len(chunks) > 5 and (i + 1) % max(1, len(chunks) // 5) == 0:
            print(f"Vocab building progress: {i + 1}/{len(chunks)} chunks")
    
    # Create vocabulary
    vocab = {
        PAD_TOKEN: pad_idx,
        UNK_TOKEN: unk_idx,
        BOS_TOKEN: max(pad_idx, unk_idx) + 1,
        EOS_TOKEN: max(pad_idx, unk_idx) + 2
    }
    next_index = max(vocab.values()) + 1
    
    for tok, _ in combined_counter.most_common(max_size - len(vocab)):
        if tok not in vocab:
            vocab[tok] = next_index
            next_index += 1
    
    elapsed = time.time() - start_time
    if verbose:
        print(f"Vocabulary built with {len(vocab)} tokens in {elapsed:.2f}s")
    return vocab

def get_subword_training_sample(dataset, lang_key, sample_size=100000):
    """Get representative sample for subword tokenizer training"""
    if len(dataset) > sample_size:
        indices = np.random.choice(len(dataset), sample_size, replace=False)
        return [dataset[i]["translation"][lang_key] for i in indices]
    return [ex["translation"][lang_key] for ex in dataset]

def train_subword_tokenizer(texts, vocab_size=1024, special_tokens=None, cache_manager=None, 
                           dataset_name="dataset", lang="lang", level="subword", verbose=True):
    """Train BPE tokenizer with proper special tokens and caching"""
    # Try loading cached tokenizer
    if cache_manager:
        cached_tokenizer = cache_manager.load_tokenizer(dataset_name, lang, level, vocab_size)
        if cached_tokenizer:
            if verbose:
                print(f"Using cached tokenizer for {lang}")
            return cached_tokenizer, cached_tokenizer.get_vocab()
    
    if special_tokens is None:
        special_tokens = [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN]
    
    if verbose:
        print(f"Training BPE tokenizer on {len(texts)} texts...")
    start_time = time.time()
    
    tokenizer = Tokenizer(models.BPE(unk_token=UNK_TOKEN))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        min_frequency=2
    )
    
    # Train in batches for memory efficiency
    tokenizer.train_from_iterator(texts, trainer)
    vocab = tokenizer.get_vocab()
    
    elapsed = time.time() - start_time
    if verbose:
        print(f"Trained BPE tokenizer with {len(vocab)} tokens in {elapsed:.2f}s")
    
    # Cache tokenizer if cache manager is available
    if cache_manager:
        cache_manager.save_tokenizer(tokenizer, dataset_name, lang, level, vocab_size)
        cache_manager.save_vocab(vocab, dataset_name, lang, level, vocab_size)
    
    return tokenizer, vocab

class FastStreamingDataset(Dataset):
    """Memory-efficient dataset with safe multiprocessing support"""
    def __init__(self, data, src_lang, tgt_lang, src_vocab, tgt_vocab,
                 level, seq_length, shift, src_tok=None, tgt_tok=None,
                 max_preprocess_size=1000000, num_workers=4, verbose=True):
        
        self.data = data
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.seq_length = seq_length
        self.level = level
        self.src_tok = src_tok
        self.tgt_tok = tgt_tok
        self.shift = shift
        self.num_workers = num_workers
        self.pad_idx = src_vocab[PAD_TOKEN]
        self.verbose = verbose
        
        # For large datasets, use streaming mode to avoid OOM
        # For smaller datasets, pre-tokenize everything
        self.streaming_mode = len(data) > max_preprocess_size
        self.preprocessed_data = None
        
        # Calculate sequence length statistics from a sample
        self._calculate_sequence_stats()
        
        # Pre-process smaller datasets
        if not self.streaming_mode:
            if self.verbose:
                print(f"Pre-processing {len(data)} examples (smaller than threshold {max_preprocess_size})")
            self._preprocess_data()
    
    def _calculate_sequence_stats(self):
        """Get statistics on sequence lengths"""
        sample_size = min(1000, len(self.data))
        sample_indices = random.sample(range(len(self.data)), sample_size)
        
        sample_lengths = []
        # Disable tqdm for sequence length estimation
        for idx in sample_indices:
            ex = self.data[idx]["translation"]
            src_tokens = simple_tokenizer(ex[self.src_lang], self.level, self.src_tok)
            tgt_tokens = simple_tokenizer(ex[self.tgt_lang], self.level, self.tgt_tok)
            sample_lengths.append(max(len(src_tokens), len(tgt_tokens)))
        
        self.avg_seq_len = np.mean(sample_lengths) if sample_lengths else self.seq_length
        if self.verbose:
            print(f"Estimated average sequence length: {self.avg_seq_len:.2f}")
    
    def _preprocess_data(self):
        """Process dataset in batches with minimal logging - ALWAYS use single process"""
        # Create data batches for parallel processing
        batch_size = 1000
        batches = [self.data[i:i+batch_size] for i in range(0, len(self.data), batch_size)]
        
        # FIXED: Always use single process version to prevent multiprocessing issues
        results = []
        for i, batch in enumerate(batches):
            batch_results = process_batch_fn(
                batch,
                self.src_lang,
                self.tgt_lang,
                self.level,
                self.src_tok,
                self.tgt_tok,
                self.src_vocab,
                self.tgt_vocab,
                self.seq_length,
                self.shift
            )
            results.extend(batch_results)
            
            # Progress every 20% instead of every batch
            if self.verbose and (i + 1) % max(1, len(batches) // 5) == 0:
                print(f"Processing progress: {i + 1}/{len(batches)} batches")
        
        self.preprocessed_data = results
        if self.verbose:
            print(f"Pre-processed {len(self.preprocessed_data)} valid examples")

    def __len__(self):
        if self.preprocessed_data is not None:
            return len(self.preprocessed_data)
        return len(self.data)

    def __getitem__(self, idx):
        # Return preprocessed data if available
        if self.preprocessed_data is not None:
            return self.preprocessed_data[idx]
        
        # Otherwise, process on-the-fly (streaming mode)
        ex = self.data[idx]
        tokens_pair = tokenize_example_fn(
            ex, self.src_lang, self.tgt_lang, 
            self.level, self.src_tok, self.tgt_tok, 
            self.seq_length
        )
        
        if tokens_pair is None:
            # Return None for invalid sequences - will be filtered by collate function
            return None
            
        result = numericalize_example_fn(
            tokens_pair, 
            self.src_vocab, self.tgt_vocab, 
            self.seq_length, self.shift
        )
        
        return result  # May be None, which will be filtered

def validate_batch(src_batch, tgt_batch, pad_idx, debug_step_count):
    """Validate batch and filter out problematic samples"""
    if src_batch is None or tgt_batch is None:
        return None, None, True
    
    if src_batch.numel() == 0 or tgt_batch.numel() == 0:
        print(f"⚠️  BATCH {debug_step_count}: Empty batch detected")
        return None, None, True
    
    # Check for NaN values
    if torch.isnan(src_batch.float()).any() or torch.isnan(tgt_batch.float()).any():
        print(f"⚠️  BATCH {debug_step_count}: NaN values detected in batch")
        return None, None, True
    
    # Check for all-padding batches
    src_non_pad = (src_batch != pad_idx).any(dim=1)  # [batch_size] - True if row has non-padding
    tgt_non_pad = (tgt_batch != pad_idx).any(dim=1)  # [batch_size] - True if row has non-padding
    
    # Keep only samples that have at least some non-padding tokens in both src and tgt
    valid_mask = src_non_pad & tgt_non_pad
    
    if not valid_mask.any():
        print(f"⚠️  BATCH {debug_step_count}: All samples are padding-only")
        return None, None, True
    
    # Filter out invalid samples
    valid_src = src_batch[valid_mask]
    valid_tgt = tgt_batch[valid_mask]
    
    # Check if we filtered out too many samples
    original_size = src_batch.size(0)
    filtered_size = valid_src.size(0)
    
    if filtered_size < original_size * 0.5:  # More than 50% filtered
        print(f"⚠️  BATCH {debug_step_count}: Filtered {original_size - filtered_size}/{original_size} samples")
    
    # Final validation
    if valid_src.size(0) == 0:
        return None, None, True
    
    return valid_src, valid_tgt, False

# Enhanced collate function with comprehensive filtering
def enhanced_filter_collate(batch, pad_idx=0):
    """Enhanced collate function that filters and validates batches"""
    # Filter out None values
    batch = [item for item in batch if item is not None]
    
    if len(batch) == 0:
        return None
    
    # ENHANCED: Filter out very small batches before collating
    if len(batch) < 2:
        print(f"⚠️  Filtering out batch with only {len(batch)} samples")
        return None
    
    try:
        collated = torch.utils.data.dataloader.default_collate(batch)
        src_batch, tgt_batch = collated
        
        # Validate each sample in the batch
        valid_indices = []
        for i in range(src_batch.size(0)):
            src_sample = src_batch[i]
            tgt_sample = tgt_batch[i]
            
            # Check if source has valid content
            src_valid = (src_sample != pad_idx).any()
            # Check if target has valid content (especially after BOS removal)
            tgt_valid = (tgt_sample[1:] != pad_idx).any() if tgt_sample.size(0) > 1 else False
            
            if src_valid and tgt_valid:
                valid_indices.append(i)
        
        # Keep only valid samples
        if len(valid_indices) == 0:
            print(f"⚠️  All samples in batch are invalid")
            return None
        
        if len(valid_indices) < len(batch) * 0.5:  # Lost more than 50%
            print(f"⚠️  Filtered {len(batch) - len(valid_indices)}/{len(batch)} samples")
        
        # Return filtered batch
        valid_indices = torch.tensor(valid_indices)
        return (src_batch[valid_indices], tgt_batch[valid_indices])
        
    except Exception as e:
        print(f"⚠️  Error in collate function: {e}")
        return None

# Custom collate function for filtering None values
def filter_none_collate(batch):
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        # Return empty batch with correct shapes if all items were filtered
        return (
            torch.zeros((0, 0), dtype=torch.long),
            torch.zeros((0, 0), dtype=torch.long)
        )
    return torch.utils.data.dataloader.default_collate(batch)

def mt_tokenizer(dataset, dataset_name="wmt14", tk_level="word", tokens_per_batch=25000, seq_length=32,
                         src_lang="de", tgt_lang="en", max_vocab_size=10000,
                         pad_idx=0, unk_idx=1, shift=0, shared_vocab=False,
                         cache_dir="/data/tokenizer_cache", use_disk_cache=True,
                         vocab_sample_size=500000, max_preprocess_size=500000, verbose=True):
    """
    Optimized tokenization pipeline for Singularity/HPC environments:
    
    1. Uses multiprocessing-safe approach with top-level functions
    2. Has both memory and disk caching options
    3. Efficiently samples data for large datasets
    4. Optimizes memory usage through streaming when needed
    5. Handles pickling issues that occur in containerized environments
    6. ENHANCED: Comprehensive filtering of invalid batches
    7. NEW: Much quieter progress reporting with verbose control
    8. FIXED: Uses safe single-process mode for all tokenization levels
    """
    # Initialize cache manager with container-friendly paths
    cache_manager = CacheManager(cache_dir=cache_dir, use_disk=use_disk_cache)
    
    # FIXED: Use conservative CPU count - single process for safety
    if verbose:
        print(f"Using container-safe single-process mode for tokenization")
    
    # Disable tokenizers parallelism to prevent deadlocks for all levels
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    # Convert dataset to lists for easier processing
    splits = {k: list(v) for k, v in dataset.items()}
    
    # Process based on tokenization level
    if tk_level == 'subword':
        special_tokens = [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN]
        
        if shared_vocab:
            # Sample data for vocabulary training
            sample_texts = get_subword_training_sample(
                splits['train'], src_lang, vocab_sample_size//2)
            sample_texts += get_subword_training_sample(
                splits['train'], tgt_lang, vocab_sample_size//2)
            
            # Train shared tokenizer
            tokenizer, vocab = train_subword_tokenizer(
                sample_texts, vocab_size=max_vocab_size,
                special_tokens=special_tokens, cache_manager=cache_manager,
                dataset_name=dataset_name, lang="shared", level=tk_level, verbose=verbose
            )
            src_tok = tgt_tok = tokenizer
            src_vocab = tgt_vocab = vocab
        else:
            # Train separate tokenizers
            src_sample = get_subword_training_sample(
                splits['train'], src_lang, vocab_sample_size)
            tgt_sample = get_subword_training_sample(
                splits['train'], tgt_lang, vocab_sample_size)
            
            src_tok, src_vocab = train_subword_tokenizer(
                src_sample, vocab_size=max_vocab_size,
                special_tokens=special_tokens, cache_manager=cache_manager,
                dataset_name=dataset_name, lang=src_lang, level=tk_level, verbose=verbose
            )
            tgt_tok, tgt_vocab = train_subword_tokenizer(
                tgt_sample, vocab_size=max_vocab_size,
                special_tokens=special_tokens, cache_manager=cache_manager,
                dataset_name=dataset_name, lang=tgt_lang, level=tk_level, verbose=verbose
            )
    else:
        # Try to load cached vocabulary
        src_vocab = tgt_vocab = None
        if cache_manager:
            if shared_vocab:
                src_vocab = tgt_vocab = cache_manager.load_vocab(
                    dataset_name, "shared", tk_level, max_vocab_size)
            else:
                src_vocab = cache_manager.load_vocab(
                    dataset_name, src_lang, tk_level, max_vocab_size)
                tgt_vocab = cache_manager.load_vocab(
                    dataset_name, tgt_lang, tk_level, max_vocab_size)
        
        # Build vocabulary if not loaded from cache
        if shared_vocab and not src_vocab:
            # For shared vocab, create a combined dataset
            combined_dataset = []
            for ex in splits['train']:
                combined_dataset.append({
                    "translation": {
                        "combined": ex['translation'][src_lang] + " " + ex['translation'][tgt_lang]
                    }
                })
            
            src_vocab = tgt_vocab = build_vocab_parallel(
                combined_dataset, "combined", tk_level,
                max_vocab_size, pad_idx, unk_idx,
                sample_size=vocab_sample_size, 
                num_workers=1, verbose=verbose  # FIXED: Always use 1 worker
            )
            
            # Save vocabulary to cache
            if cache_manager:
                cache_manager.save_vocab(src_vocab, dataset_name, "shared", tk_level, max_vocab_size)
                
        elif not shared_vocab:
            if not src_vocab:
                src_vocab = build_vocab_parallel(
                    splits['train'], src_lang, tk_level,
                    max_vocab_size, pad_idx, unk_idx,
                    sample_size=vocab_sample_size, 
                    num_workers=1, verbose=verbose  # FIXED: Always use 1 worker
                )
                if cache_manager:
                    cache_manager.save_vocab(src_vocab, dataset_name, src_lang, tk_level, max_vocab_size)
                    
            if not tgt_vocab:
                tgt_vocab = build_vocab_parallel(
                    splits['train'], tgt_lang, tk_level,
                    max_vocab_size, pad_idx, unk_idx,
                    sample_size=vocab_sample_size, 
                    num_workers=1, verbose=verbose  # FIXED: Always use 1 worker
                )
                if cache_manager:
                    cache_manager.save_vocab(tgt_vocab, dataset_name, tgt_lang, tk_level, max_vocab_size)
        
        # No tokenizer objects for non-subword tokenization
        src_tok = tgt_tok = None

    # Create datasets and dataloaders
    datasets = {}
    dataloaders = {}
    batch_sizes = {}
    
    for split_name in ['train', 'validation', 'test']:
        # For test/validation we can use a higher preprocessing threshold
        current_max_preprocess = max_preprocess_size
        if split_name != 'train':
            current_max_preprocess = max_preprocess_size * 10
        
        # Use our streaming dataset implementation - FIXED: Always use 1 worker
        datasets[split_name] = FastStreamingDataset(
            splits[split_name],
            src_lang, tgt_lang,
            src_vocab, tgt_vocab,
            tk_level, seq_length, shift,
            src_tok, tgt_tok,
            max_preprocess_size=current_max_preprocess,
            num_workers=1,  # FIXED: Always use single worker for all tokenization levels
            verbose=verbose
        )
        
        # Calculate optimal batch size
        avg_seq_len = datasets[split_name].avg_seq_len
        batch_size = max(1, int(tokens_per_batch / avg_seq_len))
        batch_sizes[split_name] = batch_size
        
        # FIXED: Always use 0 workers in DataLoader for container safety
        num_workers = 0
        
        # Create enhanced collate function with pad_idx
        collate_fn = partial(enhanced_filter_collate, pad_idx=pad_idx)
        
        # Create dataloader - FIXED: Safe settings for all tokenization levels
        dataloaders[split_name] = DataLoader(
            datasets[split_name],
            batch_size=batch_size,
            shuffle=(split_name == 'train'),
            pin_memory=False,  # FIXED: Disable pin_memory for container safety
            num_workers=num_workers,
            collate_fn=collate_fn,
            persistent_workers=False  # FIXED: Disable persistent workers
        )
    
    # Build reverse vocabulary lookups
    src_id_to_token = {idx: tok for tok, idx in src_vocab.items()}
    tgt_id_to_token = {idx: tok for tok, idx in tgt_vocab.items()}
    
    # Print summary information
    if verbose:
        vocab_status = "shared" if shared_vocab else "separate"
        print(f"Container-safe tokenization completed for {tk_level} level with {vocab_status} vocabulary")
        print(f"Dynamic batch sizes:")
        for split, batch_size in batch_sizes.items():
            streaming_status = "streaming" if datasets[split].streaming_mode else "precomputed"
            print(f"  - {split}: {batch_size} examples (avg seq length: {datasets[split].avg_seq_len:.1f}, {streaming_status})")
    
    return {
        'model_interface': {
            'num_input_tokens': len(src_vocab),
            'num_classes': len(tgt_vocab),
            'src_vocab': src_vocab,
            'tgt_vocab': tgt_vocab,
            'src_id_to_token': src_id_to_token,
            'tgt_id_to_token': tgt_id_to_token,
            'padding_idx': pad_idx,
            'unk_idx': unk_idx,
            'seq_length': seq_length,
            'shift': shift,
            'shared_vocab': shared_vocab,
            'avg_seq_len': {split: datasets[split].avg_seq_len for split in splits},
            'batch_sizes': batch_sizes
        },
        'train': dataloaders['train'],
        'validation': dataloaders['validation'],
        'test': dataloaders['test'],
    }