import torch
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import numpy as np
from tqdm import tqdm
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, normalizers
import re
from typing import List, Dict, Optional, Tuple
import os
import multiprocessing as mp
import pickle
import time
import random
from functools import partial
import ast
import keyword

# Special tokens for code translation
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
BOS_TOKEN = "<BOS>"
EOS_TOKEN = "<EOS>"
INDENT_TOKEN = "<INDENT>"
DEDENT_TOKEN = "<DEDENT>"
NEWLINE_TOKEN = "<NEWLINE>"
COMMENT_TOKEN = "<COMMENT>"

# Language-specific keywords and operators
LANGUAGE_KEYWORDS = {
    'cpp': ['int', 'float', 'double', 'char', 'bool', 'void', 'string', 'vector', 'map', 'set',
           'if', 'else', 'for', 'while', 'do', 'switch', 'case', 'break', 'continue', 'return',
           'class', 'struct', 'public', 'private', 'protected', 'virtual', 'static', 'const',
           'include', 'namespace', 'using', 'std', 'cout', 'cin', 'endl'],
    'java': ['int', 'float', 'double', 'char', 'boolean', 'void', 'String', 'List', 'Map', 'Set',
            'if', 'else', 'for', 'while', 'do', 'switch', 'case', 'break', 'continue', 'return',
            'class', 'interface', 'public', 'private', 'protected', 'static', 'final', 'abstract',
            'import', 'package', 'extends', 'implements', 'new', 'this', 'super'],
    'python': ['int', 'float', 'str', 'bool', 'list', 'dict', 'set', 'tuple', 'None', 'True', 'False',
              'if', 'elif', 'else', 'for', 'while', 'break', 'continue', 'return', 'yield',
              'class', 'def', 'import', 'from', 'as', 'with', 'try', 'except', 'finally',
              'and', 'or', 'not', 'in', 'is', 'lambda', 'global', 'nonlocal'],
    'javascript': ['var', 'let', 'const', 'function', 'return', 'if', 'else', 'for', 'while', 'do',
                  'switch', 'case', 'break', 'continue', 'try', 'catch', 'finally', 'throw',
                  'new', 'this', 'typeof', 'instanceof', 'true', 'false', 'null', 'undefined'],
}

def code_tokenizer(text: str, level: str, lang: str = None, tokenizer_obj=None) -> List[str]:
    """
    Tokenize code with language-aware preprocessing.
    """
    if not text or not text.strip():
        return []
    
    if level == "char":
        return list(text)
    
    elif level == "byte":
        return list(text.encode("utf-8"))
    
    elif level == "word":
        # Simple word-based tokenization with better handling
        tokens = re.findall(r'\w+|[^\w\s]', text.strip())
        return [tok for tok in tokens if tok.strip()]
    
    elif level == "subword":
        if not tokenizer_obj:
            raise ValueError("Subword tokenizer required")
        return tokenizer_obj.encode(text).tokens
    
    elif level == "code_aware":
        return code_aware_tokenize(text, lang)
    
    else:
        raise ValueError(f"Unsupported tokenization level: {level}")

def code_aware_tokenize(code: str, lang: str = None) -> List[str]:
    """
    Improved code-aware tokenization that preserves structure but is less aggressive.
    """
    if not code or not code.strip():
        return []
    
    tokens = []
    
    # Use a more lenient approach - split by common code delimiters
    # This regex captures important code elements without being too aggressive
    code_pattern = r'''
        (?P<STRING>["'][^"']*["'])|                    # String literals
        (?P<MULTICHAR><=|>=|==|!=|&&|\|\||->|::|\+\+|--|\+=|-=|\*=|/=|%=)|  # Multi-char operators
        (?P<NUMBER>\d+\.?\d*[fFlL]?)|                  # Numbers with suffixes
        (?P<IDENTIFIER>[a-zA-Z_][a-zA-Z0-9_]*)|       # Identifiers and keywords
        (?P<PUNCT>[{}()\[\];,.])|                     # Important punctuation
        (?P<OPERATOR>[+\-*/%=<>!&|^~])|               # Single operators
        (?P<WHITESPACE>\s+)|                          # Whitespace (to be filtered)
        (?P<OTHER>[^\s])                              # Anything else
    '''
    
    for match in re.finditer(code_pattern, code, re.VERBOSE):
        token = match.group()
        group_type = match.lastgroup
        
        # Skip pure whitespace but preserve important whitespace indicators
        if group_type == 'WHITESPACE':
            if '\n' in token:
                tokens.append(NEWLINE_TOKEN)
        else:
            tokens.append(token)
    
    # Remove empty tokens and limit length
    tokens = [tok for tok in tokens if tok.strip()]
    return tokens

# Top-level functions for multiprocessing
def process_code_chunk_for_vocab(chunk, lang_key, level, lang=None):
    """Process a chunk of code data for vocabulary building"""
    counter = Counter()
    for ex in chunk:
        try:
            code_text = ex["translation"][lang_key]
            if code_text and code_text.strip():
                tokens = code_tokenizer(code_text, level, lang)
                counter.update(tokens)
        except Exception as e:
            continue  # Skip problematic examples
    return counter

def tokenize_code_example_fn(ex, src_lang, tgt_lang, level, src_code_lang, tgt_code_lang, src_tok, tgt_tok, seq_length):
    """Tokenize a single code translation example with better error handling"""
    try:
        src_code = ex["translation"][src_lang]
        tgt_code = ex["translation"][tgt_lang]
        
        # Check for empty or invalid code
        if not src_code or not tgt_code or not src_code.strip() or not tgt_code.strip():
            return None
        
        # Limit input length before tokenization to prevent excessive tokens
        max_chars = seq_length * 8  # Rough estimate: 8 chars per token on average
        if len(src_code) > max_chars:
            src_code = src_code[:max_chars] + "..."
        if len(tgt_code) > max_chars:
            tgt_code = tgt_code[:max_chars] + "..."
        
        src_tokens = code_tokenizer(src_code, level, src_code_lang, src_tok)
        tgt_tokens = code_tokenizer(tgt_code, level, tgt_code_lang, tgt_tok)
        
        # More lenient length checking - allow 90% of seq_length
        max_allowed = int(seq_length * 0.9)
        
        if len(src_tokens) > max_allowed:
            src_tokens = src_tokens[:max_allowed]
        if len(tgt_tokens) > max_allowed:
            tgt_tokens = tgt_tokens[:max_allowed]
        
        # Only skip if both are empty or too short
        if len(src_tokens) < 3 or len(tgt_tokens) < 3:
            return None
            
        return (src_tokens, tgt_tokens)
        
    except Exception as e:
        return None

def numericalize_code_example_fn(tokens_pair, src_vocab, tgt_vocab, seq_length, shift):
    """Numericalize a tokenized code example with robust handling"""
    if not tokens_pair:
        return None
        
    try:
        src_tokens, tgt_tokens = tokens_pair
        
        # Handle vocabulary lookup with fallback
        src_ids = []
        for tok in src_tokens[:seq_length-2]:  # Leave room for special tokens
            if tok in src_vocab:
                src_ids.append(src_vocab[tok])
            else:
                src_ids.append(src_vocab[UNK_TOKEN])
        
        tgt_ids = []
        shifted_tokens = [PAD_TOKEN] * shift + tgt_tokens
        for tok in shifted_tokens[:seq_length-2]:
            if tok in tgt_vocab:
                tgt_ids.append(tgt_vocab[tok])
            else:
                tgt_ids.append(tgt_vocab[UNK_TOKEN])
        
        # Pad sequences
        src_pad_len = seq_length - len(src_ids)
        tgt_pad_len = seq_length - len(tgt_ids)
        
        src_ids += [src_vocab[PAD_TOKEN]] * src_pad_len
        tgt_ids += [tgt_vocab[PAD_TOKEN]] * tgt_pad_len
        
        # Create tensors
        src_tensor = torch.tensor(src_ids[:seq_length], dtype=torch.long)
        tgt_tensor = torch.tensor(tgt_ids[:seq_length], dtype=torch.long)
        
        # Validation - ensure we have some meaningful content
        pad_idx = src_vocab[PAD_TOKEN]
        unk_idx = src_vocab[UNK_TOKEN]
        
        # Count non-padding, non-UNK tokens
        src_meaningful = ((src_tensor != pad_idx) & (src_tensor != unk_idx)).sum().item()
        tgt_meaningful = ((tgt_tensor != pad_idx) & (tgt_tensor != unk_idx)).sum().item()
        
        # Require at least 3 meaningful tokens
        if src_meaningful < 3 or tgt_meaningful < 3:
            return None
        
        # Check for NaN or invalid values
        if torch.isnan(src_tensor.float()).any() or torch.isnan(tgt_tensor.float()).any():
            return None
        
        return (src_tensor, tgt_tensor)
        
    except Exception as e:
        return None

def process_code_batch_fn(batch, src_lang, tgt_lang, level, src_code_lang, tgt_code_lang, src_tok, tgt_tok, src_vocab, tgt_vocab, seq_length, shift):
    """Process a batch of code examples"""
    results = []
    for ex in batch:
        tokens_pair = tokenize_code_example_fn(ex, src_lang, tgt_lang, level, src_code_lang, tgt_code_lang, src_tok, tgt_tok, seq_length)
        if tokens_pair:
            item = numericalize_code_example_fn(tokens_pair, src_vocab, tgt_vocab, seq_length, shift)
            if item:
                results.append(item)
    return results

def build_code_vocab_parallel(dataset, lang_key, level, code_lang=None, max_size=8192, 
                             pad_idx=0, unk_idx=1, sample_size=None, num_workers=None):
    """Build vocabulary for code with better error handling"""
    if num_workers is None:
        num_workers = min(8, max(1, mp.cpu_count() - 1))
    
    # Sampling for very large datasets
    if sample_size and len(dataset) > sample_size:
        print(f"Sampling {sample_size} code examples from {len(dataset)} for vocabulary building")
        dataset = random.sample(dataset, sample_size)
    
    # Split data into chunks
    chunk_size = max(1, len(dataset) // num_workers)
    chunks = [dataset[i:i+chunk_size] for i in range(0, len(dataset), chunk_size)]
    
    start_time = time.time()
    print(f"Building {lang_key} code vocab with {num_workers} workers...")
    
    with mp.Pool(num_workers) as pool:
        process_fn = partial(process_code_chunk_for_vocab, lang_key=lang_key, level=level, lang=code_lang)
        counters = list(tqdm(pool.imap(process_fn, chunks), total=len(chunks), desc=f"Building {lang_key} vocab"))
    
    # Combine counters
    combined_counter = Counter()
    for counter in counters:
        combined_counter.update(counter)
    
    # Create vocabulary with code-specific special tokens
    special_tokens = [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN, 
                     INDENT_TOKEN, DEDENT_TOKEN, NEWLINE_TOKEN, COMMENT_TOKEN]
    
    # Add language-specific keywords
    if code_lang and code_lang in LANGUAGE_KEYWORDS:
        special_tokens.extend(LANGUAGE_KEYWORDS[code_lang])
    
    vocab = {}
    for i, token in enumerate(special_tokens):
        vocab[token] = i
    
    next_index = len(vocab)
    
    # Add most common tokens
    remaining_slots = max_size - len(vocab)
    for tok, count in combined_counter.most_common(remaining_slots):
        if tok not in vocab and tok.strip():  # Only add non-empty tokens
            vocab[tok] = next_index
            next_index += 1
    
    elapsed = time.time() - start_time
    print(f"Code vocabulary built with {len(vocab)} tokens in {elapsed:.2f}s")
    return vocab

class CodeStreamingDataset(Dataset):
    """Improved dataset with better preprocessing and validation"""
    def __init__(self, data, src_lang, tgt_lang, src_vocab, tgt_vocab,
                 level, seq_length, shift, src_code_lang, tgt_code_lang, 
                 src_tok=None, tgt_tok=None,
                 max_preprocess_size=200000, num_workers=4):
        
        self.data = data
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.src_code_lang = src_code_lang
        self.tgt_code_lang = tgt_code_lang
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.seq_length = seq_length
        self.level = level
        self.src_tok = src_tok
        self.tgt_tok = tgt_tok
        self.shift = shift
        self.num_workers = min(num_workers, 4)  # Limit workers to prevent issues
        self.pad_idx = src_vocab[PAD_TOKEN]
        
        # More conservative preprocessing threshold
        self.streaming_mode = len(data) > max_preprocess_size
        self.preprocessed_data = None
        
        # Calculate sequence length statistics
        self._calculate_sequence_stats()
        
        # Pre-process smaller datasets
        if not self.streaming_mode:
            print(f"Pre-processing {len(data)} code examples (smaller than threshold {max_preprocess_size})")
            self._preprocess_data()
        else:
            print(f"Using streaming mode for {len(data)} examples (larger than threshold {max_preprocess_size})")
    
    def _calculate_sequence_stats(self):
        """Get statistics on sequence lengths with better sampling"""
        sample_size = min(1000, max(100, len(self.data) // 10))
        sample_indices = random.sample(range(len(self.data)), sample_size)
        
        sample_lengths = []
        valid_samples = 0
        
        for idx in tqdm(sample_indices, desc="Estimating sequence lengths"):
            try:
                ex = self.data[idx]["translation"]
                src_text = ex.get(self.src_lang, "")
                tgt_text = ex.get(self.tgt_lang, "")
                
                if not src_text or not tgt_text:
                    continue
                
                # Limit text length for estimation
                max_chars = self.seq_length * 8
                src_text = src_text[:max_chars]
                tgt_text = tgt_text[:max_chars]
                
                src_tokens = code_tokenizer(src_text, self.level, self.src_code_lang, self.src_tok)
                tgt_tokens = code_tokenizer(tgt_text, self.level, self.tgt_code_lang, self.tgt_tok)
                
                if src_tokens and tgt_tokens:
                    sample_lengths.append(max(len(src_tokens), len(tgt_tokens)))
                    valid_samples += 1
            except Exception:
                continue
        
        if sample_lengths:
            self.avg_seq_len = np.mean(sample_lengths)
            self.max_seq_len = np.percentile(sample_lengths, 95)  # 95th percentile
        else:
            self.avg_seq_len = self.seq_length * 0.7  # Conservative estimate
            self.max_seq_len = self.seq_length
        
        print(f"Estimated average sequence length: {self.avg_seq_len:.2f}")
        print(f"95th percentile sequence length: {self.max_seq_len:.2f}")
        print(f"Valid samples for estimation: {valid_samples}/{sample_size}")
    
    def _preprocess_data(self):
        """Process dataset with better error handling and progress tracking"""
        batch_size = 500  # Smaller batches for better memory management
        batches = [self.data[i:i+batch_size] for i in range(0, len(self.data), batch_size)]
        
        # Single process version to avoid multiprocessing issues
        results = []
        total_processed = 0
        
        for batch in tqdm(batches, desc="Processing dataset"):
            batch_results = process_code_batch_fn(
                batch,
                self.src_lang,
                self.tgt_lang,
                self.level,
                self.src_code_lang,
                self.tgt_code_lang,
                self.src_tok,
                self.tgt_tok,
                self.src_vocab,
                self.tgt_vocab,
                self.seq_length,
                self.shift
            )
            results.extend(batch_results)
            total_processed += len(batch)
            
            # Print progress every few batches
            if len(batches) > 10 and (len(results) % 1000 == 0 or total_processed % 5000 == 0):
                success_rate = len(results) / total_processed if total_processed > 0 else 0
                print(f"   Processed {total_processed}/{len(self.data)} examples, "
                      f"kept {len(results)} valid examples (success rate: {success_rate:.1%})")
        
        self.preprocessed_data = results
        success_rate = len(results) / len(self.data) if len(self.data) > 0 else 0
        print(f"Pre-processed {len(self.preprocessed_data)} valid examples "
              f"from {len(self.data)} total (success rate: {success_rate:.1%})")

    def __len__(self):
        if self.preprocessed_data is not None:
            return len(self.preprocessed_data)
        return len(self.data)

    def __getitem__(self, idx):
        if self.preprocessed_data is not None:
            return self.preprocessed_data[idx]
        
        # Streaming mode - process on the fly
        ex = self.data[idx]
        tokens_pair = tokenize_code_example_fn(
            ex, self.src_lang, self.tgt_lang, 
            self.level, self.src_code_lang, self.tgt_code_lang,
            self.src_tok, self.tgt_tok, 
            self.seq_length
        )
        
        if tokens_pair is None:
            return None
            
        result = numericalize_code_example_fn(
            tokens_pair, 
            self.src_vocab, self.tgt_vocab, 
            self.seq_length, self.shift
        )
        
        return result

def enhanced_filter_collate(batch, pad_idx=0):
    """Enhanced collate function with better filtering and debugging"""
    # Filter out None values
    valid_batch = [item for item in batch if item is not None]
    
    if len(valid_batch) == 0:
        print(f"⚠️  All {len(batch)} examples in batch were None, returning None")
        return None
    
    if len(valid_batch) < len(batch):
        print(f"⚠️  Filtered batch: {len(valid_batch)}/{len(batch)} examples kept")
    
    if len(valid_batch) == 1:
        print(f"⚠️  Batch size reduced to 1, considering increase tokens_per_batch")
    
    try:
        # Use default collate function
        collated = torch.utils.data.dataloader.default_collate(valid_batch)
        src_batch, tgt_batch = collated
        
        # Additional validation
        if src_batch.size(0) == 0 or tgt_batch.size(0) == 0:
            print(f"⚠️  Empty batch after collation")
            return None
        
        return (src_batch, tgt_batch)
        
    except Exception as e:
        print(f"⚠️  Error in collate function: {e}")
        return None

def print_example_batch(dataloader, tokenizer_interface, num_examples=3):
    """Print example batch to understand the data"""
    print(f"\n{'='*60}")
    print("EXAMPLE BATCH ANALYSIS")
    print(f"{'='*60}")
    
    src_vocab = tokenizer_interface['src_vocab']
    tgt_vocab = tokenizer_interface['tgt_vocab']
    src_id_to_token = tokenizer_interface['src_id_to_token']
    tgt_id_to_token = tokenizer_interface['tgt_id_to_token']
    pad_idx = tokenizer_interface['padding_idx']
    
    try:
        # Get a batch
        batch = next(iter(dataloader))
        if batch is None:
            print("❌ First batch is None - this indicates a problem with data processing")
            return
        
        src_batch, tgt_batch = batch
        
        print(f"Batch shape: src={src_batch.shape}, tgt={tgt_batch.shape}")
        print(f"Vocabulary sizes: src={len(src_vocab)}, tgt={len(tgt_vocab)}")
        print(f"Padding index: {pad_idx}")
        
        # Analyze examples
        batch_size = min(src_batch.size(0), num_examples)
        
        for i in range(batch_size):
            src_tokens = src_batch[i]
            tgt_tokens = tgt_batch[i]
            
            # Find non-padding tokens
            src_valid = src_tokens[src_tokens != pad_idx]
            tgt_valid = tgt_tokens[tgt_tokens != pad_idx]
            
            print(f"\n--- Example {i+1} ---")
            print(f"Source length: {len(src_valid)} / {len(src_tokens)} tokens")
            print(f"Target length: {len(tgt_valid)} / {len(tgt_tokens)} tokens")
            
            # Decode first few tokens
            src_decoded = []
            for token_id in src_valid[:20]:  # First 20 tokens
                token = src_id_to_token.get(token_id.item(), f"<UNK:{token_id.item()}>")
                src_decoded.append(token)
            
            tgt_decoded = []
            for token_id in tgt_valid[:20]:  # First 20 tokens
                token = tgt_id_to_token.get(token_id.item(), f"<UNK:{token_id.item()}>")
                tgt_decoded.append(token)
            
            print(f"Source tokens: {' '.join(src_decoded)}")
            if len(src_valid) > 20:
                print(f"   ... and {len(src_valid) - 20} more tokens")
                
            print(f"Target tokens: {' '.join(tgt_decoded)}")
            if len(tgt_valid) > 20:
                print(f"   ... and {len(tgt_valid) - 20} more tokens")
        
        # Overall statistics
        print(f"\n--- Batch Statistics ---")
        src_non_pad = (src_batch != pad_idx).sum(dim=1).float()
        tgt_non_pad = (tgt_batch != pad_idx).sum(dim=1).float()
        
        print(f"Average source length: {src_non_pad.mean().item():.1f}")
        print(f"Average target length: {tgt_non_pad.mean().item():.1f}")
        print(f"Source length range: {src_non_pad.min().item():.0f} - {src_non_pad.max().item():.0f}")
        print(f"Target length range: {tgt_non_pad.min().item():.0f} - {tgt_non_pad.max().item():.0f}")
        
    except Exception as e:
        print(f"❌ Error analyzing batch: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"{'='*60}\n")

def code_translation_tokenizer(dataset, dataset_name="code_trans", tk_level="code_aware", 
                              tokens_per_batch=2048, seq_length=512,  # Increased defaults
                              src_lang="cpp", tgt_lang="java", max_vocab_size=8192,
                              pad_idx=0, unk_idx=1, shift=0, shared_vocab=False,
                              cache_dir="/data/tokenizer_cache", use_disk_cache=True,
                              vocab_sample_size=50000, max_preprocess_size=10000):  # More conservative
    """
    Improved tokenization pipeline for code translation with better error handling.
    """
    # Simple cache manager for now
    class SimpleCacheManager:
        def __init__(self, cache_dir, use_disk):
            self.cache_dir = cache_dir
            self.use_disk = use_disk
            if use_disk:
                os.makedirs(cache_dir, exist_ok=True)
        
        def load_tokenizer(self, *args): return None
        def save_tokenizer(self, *args): pass
        def load_vocab(self, *args): return None
        def save_vocab(self, *args): pass
    
    # Initialize cache manager
    cache_manager = SimpleCacheManager(cache_dir=cache_dir, use_disk=use_disk_cache)
    
    # Set CPU count
    num_cpus = min(7, max(1, mp.cpu_count() - 1))
    print(f"Using {num_cpus} CPU cores for code tokenization")
    
    # Convert dataset to lists
    splits = {k: list(v) for k, v in dataset.items()}
    
    # Print initial dataset statistics
    print(f"Dataset sizes: train={len(splits['train'])}, val={len(splits['validation'])}, test={len(splits['test'])}")
    
    # For now, use simple vocabulary building (not subword)
    if shared_vocab:
        # Create combined samples for shared vocab
        combined_samples = []
        sample_size = min(vocab_sample_size, len(splits['train']))
        sampled_data = random.sample(splits['train'], sample_size)
        
        for ex in sampled_data:
            src_code = ex["translation"].get(src_lang, "")
            tgt_code = ex["translation"].get(tgt_lang, "")
            if src_code and tgt_code:
                combined_samples.append({
                    "translation": {"combined": src_code + " " + tgt_code}
                })
        
        vocab = build_code_vocab_parallel(
            combined_samples, "combined", tk_level, code_lang="shared",
            max_size=max_vocab_size, pad_idx=pad_idx, unk_idx=unk_idx,
            sample_size=None, num_workers=num_cpus
        )
        src_vocab = tgt_vocab = vocab
    else:
        src_vocab = build_code_vocab_parallel(
            splits['train'], src_lang, tk_level, code_lang=src_lang,
            max_size=max_vocab_size, pad_idx=pad_idx, unk_idx=unk_idx,
            sample_size=vocab_sample_size, num_workers=num_cpus
        )
        tgt_vocab = build_code_vocab_parallel(
            splits['train'], tgt_lang, tk_level, code_lang=tgt_lang,
            max_size=max_vocab_size, pad_idx=pad_idx, unk_idx=unk_idx,
            sample_size=vocab_sample_size, num_workers=num_cpus
        )
    
    src_tok = tgt_tok = None  # No subword tokenizers for now
    
    # Create datasets and dataloaders
    datasets = {}
    dataloaders = {}
    batch_sizes = {}
    
    for split_name in ['train', 'validation', 'test']:
        # More conservative preprocessing for all splits
        current_max_preprocess = max_preprocess_size
        
        datasets[split_name] = CodeStreamingDataset(
            splits[split_name],
            src_lang, tgt_lang,
            src_vocab, tgt_vocab,
            tk_level, seq_length, shift,
            src_lang, tgt_lang,  # Use language names as code languages
            src_tok, tgt_tok,
            max_preprocess_size=current_max_preprocess,
            num_workers=1  # Single worker to avoid issues
        )
        
        # Calculate batch size more conservatively
        avg_seq_len = max(datasets[split_name].avg_seq_len, 50)  # Minimum assumption
        batch_size = max(1, int(tokens_per_batch / avg_seq_len))
        batch_sizes[split_name] = batch_size
        
        # Create dataloader
        collate_fn = partial(enhanced_filter_collate, pad_idx=pad_idx)
        
        dataloaders[split_name] = DataLoader(
            datasets[split_name],
            batch_size=batch_size,
            shuffle=(split_name == 'train'),
            pin_memory=True,
            num_workers=0,  # No multiprocessing for dataloaders
            collate_fn=collate_fn,
            persistent_workers=False
        )
    
    # Build reverse vocabulary lookups
    src_id_to_token = {idx: tok for tok, idx in src_vocab.items()}
    tgt_id_to_token = {idx: tok for tok, idx in tgt_vocab.items()}
    
    # Create tokenizer interface
    tokenizer_interface = {
        'num_input_tokens': len(src_vocab),
        'num_classes': len(tgt_vocab),
        'src_vocab': src_vocab,
        'tgt_vocab': tgt_vocab,
        'vocab': src_vocab,  # For compatibility
        'id_to_token': src_id_to_token,  # For compatibility
        'src_id_to_token': src_id_to_token,
        'tgt_id_to_token': tgt_id_to_token,
        'padding_idx': pad_idx,
        'unk_idx': unk_idx,
        'seq_length': seq_length,
        'shift': shift,
        'shared_vocab': shared_vocab,
        'src_lang': src_lang,
        'tgt_lang': tgt_lang,
        'tokenization_level': tk_level,
        'avg_seq_len': {split: datasets[split].avg_seq_len for split in splits},
        'batch_sizes': batch_sizes
    }
    
    # Print summary
    vocab_status = "shared" if shared_vocab else "separate"
    print(f"\nCode translation tokenization completed for {tk_level} level")
    print(f"Languages: {src_lang} -> {tgt_lang} with {vocab_status} vocabulary")
    print(f"Vocabulary sizes: src={len(src_vocab)}, tgt={len(tgt_vocab)}")
    print(f"Dynamic batch sizes:")
    for split, batch_size in batch_sizes.items():
        streaming_status = "streaming" if datasets[split].streaming_mode else "precomputed"
        dataset_size = len(datasets[split])
        print(f"  - {split}: {batch_size} examples (avg seq length: {datasets[split].avg_seq_len:.1f}, {streaming_status}, {dataset_size} total)")
    
    # Print example batch
    print("\n🔍 Analyzing example batch from training data...")
    print_example_batch(dataloaders['train'], tokenizer_interface, num_examples=2)
    
    return {
        'model_interface': tokenizer_interface,
        'train': dataloaders['train'],
        'validation': dataloaders['validation'],
        'test': dataloaders['test'],
    }

if __name__ == "__main__":
    # Test the tokenizer
    print("Code translation tokenizer module loaded successfully")