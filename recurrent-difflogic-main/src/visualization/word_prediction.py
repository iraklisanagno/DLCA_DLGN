import matplotlib.pyplot as plt
import torch
from typing import List, Dict
from torch.utils.data import DataLoader
import wandb

def generate_tranlation_table(
    model: torch.nn.Module,
    batch: dict,
    tokenizer_interface: dict,
    device: torch.device,
    visualization_config: dict,
) -> List[wandb.Image]:
    """
    Generates visualization of model predictions for a batch of examples.
    
    Args:
        model: The translation model
        batch: A batch of data containing src and tgt sequences
        tokenizer_interface: Dictionary with tokenization information
        device: Computation device
        visualization_config: Configuration for visualization
    
    Returns:
        List of wandb.Image objects containing visualizations
    """
    src_id_to_token = tokenizer_interface["src_id_to_token"]
    tgt_id_to_token = tokenizer_interface["tgt_id_to_token"]
    padding_idx = tokenizer_interface["padding_idx"]
    max_examples = visualization_config["max_examples"]
    max_positions = visualization_config["max_positions"]
    shift = tokenizer_interface.get("shift", 0)  # Get shift value if available
    
    model.eval()
    images = []
    
    # Move batch to device and get model outputs
    with torch.no_grad():
        if isinstance(batch, dict):
            src = batch["input_ids"].to(device)
            tgt = batch["labels"].to(device)
        else:
            src, tgt = [t.to(device) for t in batch]
        
        # **FIXED: Determine model type and call appropriately**
        is_synced = getattr(model, 'is_synced', True)  # Default to synced if not specified
        
        if is_synced:
            # Synced models: only pass source tokens
            outputs = model(src)
            display_target = tgt
            raw_logits = outputs[0] if isinstance(outputs, tuple) else outputs
            aligned_logits = raw_logits
            aligned_target = display_target
        else:
            # Unsynced models: pass both source and target (encoder-decoder)
            outputs = model(src, tgt)
            raw_logits = outputs[0] if isinstance(outputs, tuple) else outputs
            
            # **FIXED: Align logits and targets same as in training**
            if raw_logits.size(1) == tgt.size(1):
                # Model outputs same length as target, remove last logit position
                aligned_logits = raw_logits[:, :-1, :].contiguous()
                aligned_target = tgt[:, 1:].contiguous()  # Skip BOS token
            elif raw_logits.size(1) == tgt.size(1) - 1:
                # Model already outputs correct length
                aligned_logits = raw_logits
                aligned_target = tgt[:, 1:].contiguous()  # Skip BOS token
            else:
                # Fallback alignment
                min_len = min(raw_logits.size(1), tgt.size(1) - 1)
                aligned_logits = raw_logits[:, :min_len, :].contiguous()
                aligned_target = tgt[:, 1:1+min_len].contiguous()
                print(f"Warning in visualization: Unusual shape alignment - logits: {raw_logits.shape}, tgt: {tgt.shape}")
            
            display_target = aligned_target
            
        batch_size = src.size(0)
        probs = torch.softmax(aligned_logits, dim=-1)
        top_probs, top_tokens = torch.topk(probs, k=8, dim=-1)

    # Process each example in batch
    for example_idx in range(min(batch_size, max_examples)):
        # Convert token IDs to strings, filtering padding for source
        src_tokens = [
            src_id_to_token.get(idx.item(), "<UNK>") 
            for idx in src[example_idx]
            if idx != padding_idx
        ]
        
        # Use aligned target tokens for display
        target_tokens_for_display = [
            tgt_id_to_token.get(idx.item(), "<UNK>") 
            for idx in display_target[example_idx]
        ]

        # Create figure
        fig = plt.figure(figsize=(16, 10), dpi=100)
        ax = fig.add_subplot(111)
        ax.axis('off')
        
        # Create table data
        table_data = []
        cols = ["Position", "Source Token", "Target Truth", "Correct?"] + [f"Top {i+1}" for i in range(8)]
        
        # Get sequence length (use aligned target length)
        seq_len = min(len(target_tokens_for_display), max_positions, aligned_logits.size(1))
        
        # Track accuracy
        correct_predictions = 0
        total_predictions = 0
        
        for pos in range(seq_len):
            # Skip padding positions in target
            true_token_id = display_target[example_idx, pos].item()
            if true_token_id == padding_idx:
                continue
                
            # Get predictions for this position
            pred_tokens = [
                tgt_id_to_token.get(idx.item(), "<UNK>") 
                for idx in top_tokens[example_idx, pos].cpu()
            ]
            pred_probs = [f"{p:.2f}" for p in top_probs[example_idx, pos].cpu().numpy()]
            
            # **FIXED: Check if prediction is correct**
            predicted_token_id = top_tokens[example_idx, pos, 0].item()  # Top prediction
            is_correct = true_token_id == predicted_token_id
            correct_symbol = "✓" if is_correct else "✗"
            
            # Update accuracy counters
            if is_correct:
                correct_predictions += 1
            total_predictions += 1
            
            # Combine tokens with probabilities
            preds = [f"{t}\n({p})" for t, p in zip(pred_tokens, pred_probs)]
            
            # Get source token (handle case where source is shorter)
            src_token = src_tokens[pos] if pos < len(src_tokens) else "<PAD>"
            tgt_token = target_tokens_for_display[pos]
            
            # Position label with shift information
            position_label = f"{pos+1}"
            if not is_synced and shift > 0:
                position_label = f"{pos+1}"  # Keep simple for now
            
            table_data.append([position_label, src_token, tgt_token, correct_symbol] + preds)

        # Create table with correct styling
        if table_data:  # Only create table if we have data
            table = ax.table(
                cellText=table_data,
                colLabels=cols,
                loc='center',
                cellLoc='center',
                colWidths=[0.05, 0.12, 0.12, 0.05] + [0.08]*8
            )
            table.auto_set_font_size(False)
            table.set_fontsize(8)
            table.scale(1, 2)
            
            # Color code the "Correct?" column
            for i, row in enumerate(table_data):
                cell_color = '#90EE90' if row[3] == "✓" else '#FFB6C1'  # Light green or light red
                table[(i+1, 3)].set_facecolor(cell_color)
        
        # Calculate accuracy for this example
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
        
        # Add title with accuracy and model type information
        sync_status = "synced" if is_synced else "unsynced"
        title_text = f"Example {example_idx+1} ({sync_status}) - Accuracy: {accuracy:.1%} ({correct_predictions}/{total_predictions})"
        
        # Add source and target info (truncated for display)
        max_display_tokens = 15
        src_display = ' '.join(src_tokens[:max_display_tokens])
        if len(src_tokens) > max_display_tokens:
            src_display += "..."
            
        tgt_display = ' '.join(target_tokens_for_display[:max_display_tokens])
        if len(target_tokens_for_display) > max_display_tokens:
            tgt_display += "..."
        
        title_text += f"\nSource: {src_display}"
        title_text += f"\nTarget: {tgt_display}"
        
        # Add shape information for debugging
        if not is_synced:
            title_text += f"\nShapes - Logits: {aligned_logits.shape[1:]}, Target: {aligned_target.shape[1:]}"
        
        plt.title(title_text, fontsize=10, pad=20)
        
        # Convert to wandb image
        images.append(wandb.Image(fig))
        plt.close(fig)
    
    model.train()
    return images