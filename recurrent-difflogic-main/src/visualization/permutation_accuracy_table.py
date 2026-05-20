import matplotlib.pyplot as plt
import torch
from typing import List, Dict
from torch.utils.data import DataLoader
import wandb
import numpy as np
from custom_datasets.permutation_dataloader import apply_permutation, PERMUTATION_OPS

def generate_permutation_accuracy_table(
    model: torch.nn.Module,
    batch: dict,
    tokenizer_interface: dict,
    device: torch.device,
    visualization_config: dict,
) -> List[wandb.Image]:
    """
    Generates visualization of model predictions for permutation tasks.
    Shows control token, original sequence, expected result, and model prediction.
    
    Args:
        model: The permutation model
        batch: A batch of data containing src and tgt sequences
        tokenizer_interface: Dictionary with tokenization information
        device: Computation device
        visualization_config: Configuration for visualization
    
    Returns:
        List of wandb.Image objects containing visualizations
    """
    id_to_token = tokenizer_interface["id_to_token"]
    padding_idx = tokenizer_interface["padding_idx"]
    max_examples = visualization_config.get("max_examples", 8)
    max_positions = visualization_config.get("max_positions", 20)
    shift = tokenizer_interface.get("shift", 0)
    
    # Control token mappings
    CONTROL_TOKENS = {
        0: "REV",    # reverse
        1: "SWAP",   # swap pairs
        2: "ROTL",   # rotate left
        3: "ROTR",   # rotate right
        4: "SORTASC", # sort ascending
        5: "SORTDSC", # sort descending
        6: "SHUF"    # shuffle
    }
    
    model.eval()
    images = []
    
    # Move batch to device and get model outputs
    with torch.no_grad():
        if isinstance(batch, dict):
            src = batch["input_ids"].to(device)
            tgt = batch["labels"].to(device)
        else:
            src, tgt = [t.to(device) for t in batch]
        
        # Determine model type and call appropriately
        is_synced = getattr(model, 'is_synced', True)
        
        if is_synced:
            outputs = model(src)
            display_target = tgt
            raw_logits = outputs[0] if isinstance(outputs, tuple) else outputs
            aligned_logits = raw_logits
            aligned_target = display_target
        else:
            outputs = model(src, tgt)
            raw_logits = outputs[0] if isinstance(outputs, tuple) else outputs
            
            # Align logits and targets
            if raw_logits.size(1) == tgt.size(1):
                aligned_logits = raw_logits[:, :-1, :].contiguous()
                aligned_target = tgt[:, 1:].contiguous()
            elif raw_logits.size(1) == tgt.size(1) - 1:
                aligned_logits = raw_logits
                aligned_target = tgt[:, 1:].contiguous()
            else:
                min_len = min(raw_logits.size(1), tgt.size(1) - 1)
                aligned_logits = raw_logits[:, :min_len, :].contiguous()
                aligned_target = tgt[:, 1:1+min_len].contiguous()
            
            display_target = aligned_target
            
        batch_size = src.size(0)
        probs = torch.softmax(aligned_logits, dim=-1)
        predictions = torch.argmax(probs, dim=-1)

    # Process each example in batch
    for example_idx in range(min(batch_size, max_examples)):
        # Decode source sequence
        src_token_ids = [idx.item() for idx in src[example_idx] if idx != padding_idx]
        src_tokens = [id_to_token.get(idx, "<UNK>") for idx in src_token_ids]
        
        # Decode target sequence
        tgt_token_ids = [idx.item() for idx in display_target[example_idx]]
        tgt_tokens = [id_to_token.get(idx, "<UNK>") for idx in tgt_token_ids if idx != padding_idx]
        
        # Decode predictions
        pred_token_ids = [idx.item() for idx in predictions[example_idx]]
        pred_tokens = [id_to_token.get(idx, "<UNK>") for idx in pred_token_ids if idx != padding_idx]
        
        # Parse the permutation task
        task_info = parse_permutation_task(src_tokens, tgt_tokens, pred_tokens, CONTROL_TOKENS)
        
        if task_info is None:
            continue
            
        # Create figure
        fig = plt.figure(figsize=(16, 12), dpi=100)
        
        # Create main subplot for the table
        ax_main = fig.add_subplot(2, 1, 1)
        ax_main.axis('off')
        
        # Create position-by-position comparison table
        table_data = []
        cols = ["Position", "Target", "Predicted", "Correct?", "Probability"]
        
        seq_len = min(len(tgt_tokens), len(pred_tokens), max_positions)
        correct_positions = 0
        
        for pos in range(seq_len):
            true_token = tgt_tokens[pos] if pos < len(tgt_tokens) else "<PAD>"
            pred_token = pred_tokens[pos] if pos < len(pred_tokens) else "<PAD>"
            
            is_correct = true_token == pred_token
            correct_symbol = "✓" if is_correct else "✗"
            
            if is_correct:
                correct_positions += 1
            
            # Get probability for predicted token
            if pos < len(pred_token_ids) and pos < probs.size(1):
                pred_prob = probs[example_idx, pos, pred_token_ids[pos]].item()
                prob_str = f"{pred_prob:.3f}"
            else:
                prob_str = "N/A"
            
            table_data.append([f"{pos+1}", true_token, pred_token, correct_symbol, prob_str])

        # Create position table
        if table_data:
            table = ax_main.table(
                cellText=table_data,
                colLabels=cols,
                loc='center',
                cellLoc='center',
                colWidths=[0.1, 0.2, 0.2, 0.1, 0.15]
            )
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1, 1.8)
            
            # Color code the "Correct?" column
            for i, row in enumerate(table_data):
                cell_color = '#90EE90' if row[3] == "✓" else '#FFB6C1'
                table[(i+1, 3)].set_facecolor(cell_color)

        # Calculate accuracies
        position_accuracy = correct_positions / seq_len if seq_len > 0 else 0.0
        sequence_accuracy = 1.0 if correct_positions == seq_len else 0.0
        
        # Add title with task information
        sync_status = "synced" if is_synced else "unsynced"
        title_text = f"Example {example_idx+1} ({sync_status}) - Position Acc: {position_accuracy:.1%}, Sequence Acc: {sequence_accuracy:.1%}"
        
        ax_main.set_title(title_text, fontsize=12, pad=20)
        
        # Create summary subplot
        ax_summary = fig.add_subplot(2, 1, 2)
        ax_summary.axis('off')
        
        # Create summary table with task details
        summary_data = [
            ["Control Token", task_info["control_display"]],
            ["Operation", task_info["operation_name"]],
            ["Original Sequence", " ".join(task_info["original_sequence"])],
            ["Expected Result", " ".join(task_info["expected_result"])],
            ["Model Prediction", " ".join(task_info["predicted_result"])],
            ["Task Correct", "✓" if task_info["task_correct"] else "✗"],
            ["Valid Permutation", "✓" if task_info["valid_permutation"] else "✗"]
        ]
        
        summary_table = ax_summary.table(
            cellText=summary_data,
            colLabels=["Property", "Value"],
            loc='center',
            cellLoc='left',
            colWidths=[0.25, 0.7]
        )
        summary_table.auto_set_font_size(False)
        summary_table.set_fontsize(10)
        summary_table.scale(1, 2)
        
        # Color code task correctness
        task_color = '#90EE90' if task_info["task_correct"] else '#FFB6C1'
        summary_table[(6, 1)].set_facecolor(task_color)
        
        perm_color = '#90EE90' if task_info["valid_permutation"] else '#FFB6C1'
        summary_table[(7, 1)].set_facecolor(perm_color)
        
        ax_summary.set_title("Permutation Task Analysis", fontsize=12, pad=10)
        
        plt.tight_layout()
        
        # Convert to wandb image
        images.append(wandb.Image(fig))
        plt.close(fig)
    
    # Create operation summary visualization
    if len(images) > 0:
        summary_image = create_operation_summary(model, batch, tokenizer_interface, device, CONTROL_TOKENS)
        if summary_image:
            images.append(summary_image)
    
    model.train()
    return images


def parse_permutation_task(src_tokens, tgt_tokens, pred_tokens, control_tokens):
    """Parse permutation task from tokens"""
    try:
        if len(src_tokens) < 2:
            return None
            
        # Extract control token
        control_token_str = src_tokens[0]
        
        # Try to map control token
        control_num = None
        for num, token_name in control_tokens.items():
            if control_token_str == token_name or control_token_str == str(num):
                control_num = num
                break
        
        if control_num is None:
            return None
            
        # Extract original sequence (skip control token)
        original_sequence = src_tokens[1:]
        
        # Apply expected permutation
        try:
            expected_result, operation_name = apply_permutation(control_num, original_sequence)
        except:
            return None
        
        # Check if prediction matches expected result
        task_correct = pred_tokens == expected_result
        
        # Check if prediction is a valid permutation (same elements, different order)
        valid_permutation = (
            len(pred_tokens) == len(original_sequence) and
            sorted(pred_tokens) == sorted(original_sequence)
        )
        
        return {
            "control_num": control_num,
            "control_display": f"{control_num} ({control_tokens[control_num]})",
            "operation_name": operation_name,
            "original_sequence": original_sequence,
            "expected_result": expected_result,
            "predicted_result": pred_tokens,
            "task_correct": task_correct,
            "valid_permutation": valid_permutation
        }
        
    except Exception as e:
        print(f"Error parsing permutation task: {e}")
        return None


def create_operation_summary(model, batch, tokenizer_interface, device, control_tokens):
    """Create a summary visualization showing accuracy by operation type"""
    try:
        id_to_token = tokenizer_interface["id_to_token"]
        padding_idx = tokenizer_interface["padding_idx"]
        
        with torch.no_grad():
            if isinstance(batch, dict):
                src = batch["input_ids"].to(device)
                tgt = batch["labels"].to(device)
            else:
                src, tgt = [t.to(device) for t in batch]
            
            is_synced = getattr(model, 'is_synced', True)
            
            if is_synced:
                outputs = model(src)
                aligned_logits = outputs[0] if isinstance(outputs, tuple) else outputs
                aligned_target = tgt
            else:
                outputs = model(src, tgt)
                raw_logits = outputs[0] if isinstance(outputs, tuple) else outputs
                
                if raw_logits.size(1) == tgt.size(1):
                    aligned_logits = raw_logits[:, :-1, :].contiguous()
                    aligned_target = tgt[:, 1:].contiguous()
                else:
                    aligned_logits = raw_logits
                    aligned_target = tgt[:, 1:].contiguous()
            
            predictions = torch.argmax(aligned_logits, dim=-1)
        
        # Analyze accuracy by operation
        operation_stats = {op_name: {"correct": 0, "total": 0} for op_name in PERMUTATION_OPS.values()}
        operation_stats = {op[0]: {"correct": 0, "total": 0} for op in operation_stats}
        
        batch_size = src.size(0)
        
        for example_idx in range(batch_size):
            src_token_ids = [idx.item() for idx in src[example_idx] if idx != padding_idx]
            src_tokens = [id_to_token.get(idx, "<UNK>") for idx in src_token_ids]
            
            tgt_token_ids = [idx.item() for idx in aligned_target[example_idx]]
            tgt_tokens = [id_to_token.get(idx, "<UNK>") for idx in tgt_token_ids if idx != padding_idx]
            
            pred_token_ids = [idx.item() for idx in predictions[example_idx]]
            pred_tokens = [id_to_token.get(idx, "<UNK>") for idx in pred_token_ids if idx != padding_idx]
            
            task_info = parse_permutation_task(src_tokens, tgt_tokens, pred_tokens, control_tokens)
            
            if task_info:
                op_name = task_info["operation_name"]
                if op_name in operation_stats:
                    operation_stats[op_name]["total"] += 1
                    if task_info["task_correct"]:
                        operation_stats[op_name]["correct"] += 1
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Bar chart of accuracy by operation
        ops = list(operation_stats.keys())
        accuracies = [
            (stats["correct"] / stats["total"]) if stats["total"] > 0 else 0 
            for stats in operation_stats.values()
        ]
        counts = [stats["total"] for stats in operation_stats.values()]
        
        bars = ax1.bar(ops, accuracies, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8'])
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Accuracy by Permutation Operation')
        ax1.set_ylim(0, 1)
        
        # Add count labels on bars
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'n={count}', ha='center', va='bottom', fontsize=8)
        
        plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
        
        # Summary statistics table
        ax2.axis('off')
        
        total_correct = sum(stats["correct"] for stats in operation_stats.values())
        total_examples = sum(stats["total"] for stats in operation_stats.values())
        overall_accuracy = total_correct / total_examples if total_examples > 0 else 0
        
        summary_data = [
            ["Overall Accuracy", f"{overall_accuracy:.1%} ({total_correct}/{total_examples})"],
            ["Total Examples", str(total_examples)],
            ["Operations Tested", str(len([op for op, stats in operation_stats.items() if stats["total"] > 0]))],
        ]
        
        for op, stats in operation_stats.items():
            if stats["total"] > 0:
                acc = stats["correct"] / stats["total"]
                summary_data.append([f"{op} Accuracy", f"{acc:.1%} ({stats['correct']}/{stats['total']})"])
        
        summary_table = ax2.table(
            cellText=summary_data,
            colLabels=["Metric", "Value"],
            loc='center',
            cellLoc='left',
            colWidths=[0.4, 0.6]
        )
        summary_table.auto_set_font_size(False)
        summary_table.set_fontsize(10)
        summary_table.scale(1, 1.5)
        
        ax2.set_title('Batch Summary Statistics', fontsize=12)
        
        plt.tight_layout()
        
        return wandb.Image(fig)
        
    except Exception as e:
        print(f"Error creating operation summary: {e}")
        return None
    finally:
        plt.close('all')