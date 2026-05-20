import torch
from torch.nn.functional import cross_entropy

def calculate_accuracy(logits, targets, pad_index=0):
    preds = logits.argmax(dim=-1)
    mask = targets != pad_index
    correct = (preds[mask] == targets[mask]).float().sum()
    return (correct / mask.sum()).item() if mask.sum() > 0 else 0.0

def calculate_top_k_accuracy(logits, targets, k=5, pad_index=0):
    """
    Calculate top-k accuracy by checking if the correct label is in the top k predictions.
    
    Args:
        logits: Model output logits of shape (batch_size, seq_len, vocab_size)
        targets: Target indices of shape (batch_size, seq_len)
        k: Number of top predictions to consider
        pad_index: Index used for padding tokens to ignore
        
    Returns:
        Top-k accuracy as a float
    """
    # Get top-k predictions
    _, top_k_preds = logits.topk(k=k, dim=-1)
    
    # Create a mask to ignore pad tokens
    mask = targets.unsqueeze(-1).expand_as(top_k_preds) != pad_index
    
    # Check if the correct label is in the top-k predictions
    correct = (top_k_preds == targets.unsqueeze(-1).expand_as(top_k_preds)) & mask
    correct_per_position = correct.any(dim=-1)
    
    # Count total correct predictions (ignoring padding)
    total_correct = correct_per_position.float().sum()
    total_tokens = (targets != pad_index).float().sum()
    
    return (total_correct / total_tokens).item() if total_tokens > 0 else 0.0

def calculate_top_5_accuracy(logits, targets, pad_index=0):
    """Calculate top-5 accuracy"""
    return calculate_top_k_accuracy(logits, targets, k=5, pad_index=pad_index)

def calculate_top_10_accuracy(logits, targets, pad_index=0):
    """Calculate top-10 accuracy"""
    return calculate_top_k_accuracy(logits, targets, k=10, pad_index=pad_index)

def calculate_perplexity(logits, targets, pad_index=0):
    loss = cross_entropy(logits.view(-1, logits.size(-1)),
                         targets.view(-1), 
                         ignore_index=pad_index)
    return torch.exp(loss).item()

def calculate_precision(logits, targets, pad_index=0):
    preds = logits.argmax(dim=-1)  # Get predicted indices
    mask = targets != pad_index    # Mask out padding tokens
    
    tp = ((preds == targets) & mask).sum().item()  # True positives
    fp = ((preds != targets) & mask).sum().item()  # False positives
    
    precision = tp / (tp + fp + 1e-8)  # Avoid division by zero
    return precision  # Return precision as a scalar

def calculate_recall(logits, targets, pad_index=0):
    preds = logits.argmax(dim=-1)  # Get predicted indices
    mask = targets != pad_index    # Mask out padding tokens
    
    tp = ((preds == targets) & mask).sum().item()  # True positives
    fn = ((preds != targets) & mask).sum().item()  # False negatives
    
    recall = tp / (tp + fn + 1e-8)  # Avoid division by zero
    return recall  # Return recall as a scalar

def calculate_f1(logits, targets, pad_index=0):
    preds = logits.argmax(dim=-1)  # Get predicted indices
    mask = targets != pad_index    # Mask out padding tokens
    
    tp = ((preds == targets) & mask).sum().item()  # True positives
    fp = ((preds != targets) & mask).sum().item()  # False positives
    fn = ((preds != targets) & mask).sum().item()  # False negatives
    
    precision = tp / (tp + fp + 1e-8)  # Avoid division by zero
    recall = tp / (tp + fn + 1e-8)     # Avoid division by zero
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)  # F1 score
    
    return f1  # Return F1 score as a scalar