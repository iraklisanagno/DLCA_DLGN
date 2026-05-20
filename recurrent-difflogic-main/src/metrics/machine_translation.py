# src/metrics/machine_translation.py
import numpy as np
import sacrebleu
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from nltk.translate.nist_score import sentence_nist
from rouge_score import rouge_scorer
from nltk.tokenize import word_tokenize

def _safe_tokenize(text):
    """Tokenize text and handle empty strings"""
    return word_tokenize(text) if text.strip() else []

def _build_idx2token(vocab: dict):
    return {idx: tok for tok, idx in vocab.items()}

def _remove_padding(tokens, pad_index):
    return [t for t in tokens if t != pad_index]

def calculate_bleu(logits, targets, vocab, pad_index=0):
    """Calculate BLEU score using NLTK's implementation with smoothing."""
    idx2token = _build_idx2token(vocab)
    smooth_fn = SmoothingFunction().method4
    pred_ids = logits.argmax(-1).cpu().numpy()
    targ_ids = targets.cpu().numpy()

    bleu_scores = []
    for pred, targ in zip(pred_ids, targ_ids):
        pred_tokens = [idx2token[i] for i in _remove_padding(pred, pad_index)]
        targ_tokens = [idx2token[i] for i in _remove_padding(targ, pad_index)]
        if pred_tokens and targ_tokens:
            bleu_scores.append(sentence_bleu([targ_tokens], pred_tokens, smoothing_function=smooth_fn))
    return np.mean(bleu_scores) if bleu_scores else 0.0

def calculate_sacrebleu(logits, targets, vocab, pad_index=0):
    """Calculate BLEU score using sacreBLEU (standard in MT)."""
    idx2token = _build_idx2token(vocab)
    pred_ids = logits.argmax(-1).cpu().numpy()
    targ_ids = targets.cpu().numpy()

    pred_strs = []
    targ_strs = []
    for pred, targ in zip(pred_ids, targ_ids):
        pred_tokens = _remove_padding(pred, pad_index)
        targ_tokens = _remove_padding(targ, pad_index)
        pred_strs.append(' '.join(idx2token[i] for i in pred_tokens))
        targ_strs.append(' '.join(idx2token[i] for i in targ_tokens))
    
    return sacrebleu.corpus_bleu(pred_strs, [targ_strs]).score if pred_strs else 0.0


def calculate_meteor(logits, targets, vocab, pad_index=0):
    """Calculate METEOR score with simplified token handling."""
    idx2token = _build_idx2token(vocab)
    pred_ids = logits.argmax(-1).cpu().numpy()
    targ_ids = targets.cpu().numpy()

    meteor_scores = []
    for pred, targ in zip(pred_ids, targ_ids):
        pred_tokens = [str(idx2token[i]) for i in _remove_padding(pred, pad_index)]
        targ_tokens = [str(idx2token[i]) for i in _remove_padding(targ, pad_index)]
        
        if pred_tokens and targ_tokens:
            try:
                # Use the tokens directly (already tokenized at word level)
                meteor_scores.append(meteor_score([targ_tokens], pred_tokens))
            except:
                meteor_scores.append(0.0)  # Fallback to worst score
    
    return np.mean(meteor_scores) if meteor_scores else 0.0

def calculate_nist(logits, targets, vocab, pad_index=0, n=5):
    """Calculate NIST score (variant of BLEU that weights n-grams by importance)."""
    idx2token = _build_idx2token(vocab)
    pred_ids = logits.argmax(-1).cpu().numpy()
    targ_ids = targets.cpu().numpy()

    nist_scores = []
    for pred, targ in zip(pred_ids, targ_ids):
        pred_tokens = [idx2token[i] for i in _remove_padding(pred, pad_index)]
        targ_tokens = [idx2token[i] for i in _remove_padding(targ, pad_index)]
        
        # Skip calculation if either sequence is empty
        if not pred_tokens or not targ_tokens:
            continue
            
        try:
            # Catch potential ZeroDivisionError from NLTK's NIST implementation
            score = sentence_nist([targ_tokens], pred_tokens, n)
            nist_scores.append(score)
        except ZeroDivisionError:
            # This happens when there are no matching n-grams
            # We can either skip this example or assign a default score of 0
            continue  # or: nist_scores.append(0.0)
            
    return np.mean(nist_scores) if nist_scores else 0.0


def calculate_rouge_l(logits, targets, vocab, pad_index=0):
    """Calculate ROUGE-L F1 score."""
    idx2token = _build_idx2token(vocab)
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    pred_ids = logits.argmax(-1).cpu().numpy()
    targ_ids = targets.cpu().numpy()

    scores = []
    for pred, targ in zip(pred_ids, targ_ids):
        pred_tokens = _remove_padding(pred, pad_index)
        targ_tokens = _remove_padding(targ, pad_index)
        if pred_tokens and targ_tokens:
            pred_str = ' '.join(idx2token[i] for i in pred_tokens)
            targ_str = ' '.join(idx2token[i] for i in targ_tokens)
            scores.append(scorer.score(targ_str, pred_str)['rougeL'].fmeasure)
    return np.mean(scores) if scores else 0.0

def calculate_chrf(logits, targets, vocab, pad_index=0):
    """Calculate chrF score (character n-gram F-score)."""
    idx2token = _build_idx2token(vocab)
    pred_ids = logits.argmax(-1).cpu().numpy()
    targ_ids = targets.cpu().numpy()

    pred_strs = []
    targ_strs = []
    for pred, targ in zip(pred_ids, targ_ids):
        pred_tokens = _remove_padding(pred, pad_index)
        targ_tokens = _remove_padding(targ, pad_index)
        pred_strs.append(' '.join(idx2token[i] for i in pred_tokens))
        targ_strs.append(' '.join(idx2token[i] for i in targ_tokens))
    
    return sacrebleu.corpus_chrf(pred_strs, [targ_strs]).score if pred_strs else 0.0