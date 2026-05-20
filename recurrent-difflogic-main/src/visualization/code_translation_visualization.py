import matplotlib.pyplot as plt
import torch
from typing import List, Dict
from torch.utils.data import DataLoader
import wandb
import textwrap

def generate_code_translation_table(
    model: torch.nn.Module,
    batch: dict,
    tokenizer_interface: dict,
    device: torch.device,
    visualization_config: dict,
) -> List[wandb.Image]:
    """
    Generates visualization of model predictions for code translation examples.
    
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
    max_examples = visualization_config.get("max_examples", 3)
    max_positions = visualization_config.get("max_positions", 30)
    shift = tokenizer_interface.get("shift", 0)
    src_lang = tokenizer_interface.get("src_lang", "source")
    tgt_lang = tokenizer_interface.get("tgt_lang", "target")
    
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
            
            # Align logits and targets same as in training
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
                print(f"Warning in code visualization: Unusual shape alignment - logits: {raw_logits.shape}, tgt: {tgt.shape}")
            
            display_target = aligned_target
            
        batch_size = src.size(0)
        probs = torch.softmax(aligned_logits, dim=-1)
        top_probs, top_tokens = torch.topk(probs, k=5, dim=-1)  # Fewer predictions for code

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

        # Create figure with larger size for code
        fig = plt.figure(figsize=(20, 12), dpi=100)
        
        # Create subplots: top for code snippets, bottom for prediction table
        gs = fig.add_gridspec(3, 1, height_ratios=[1, 1, 2], hspace=0.3)
        
        # Top subplot: Source code
        ax_src = fig.add_subplot(gs[0])
        ax_src.axis('off')
        
        # Middle subplot: Target code
        ax_tgt = fig.add_subplot(gs[1])
        ax_tgt.axis('off')
        
        # Bottom subplot: Prediction table
        ax_table = fig.add_subplot(gs[2])
        ax_table.axis('off')
        
        # Reconstruct source and target code from tokens
        src_code = reconstruct_code_from_tokens(src_tokens, src_lang)
        tgt_code = reconstruct_code_from_tokens(target_tokens_for_display, tgt_lang)
        
        # Display source code
        ax_src.text(0.02, 0.95, f"Source Code ({src_lang.upper()}):", 
                   fontsize=12, weight='bold', transform=ax_src.transAxes, va='top')
        ax_src.text(0.02, 0.05, src_code, fontsize=9, family='monospace',
                   transform=ax_src.transAxes, va='bottom',
                   bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', alpha=0.7))
        
        # Display target code
        ax_tgt.text(0.02, 0.95, f"Target Code ({tgt_lang.upper()}):", 
                   fontsize=12, weight='bold', transform=ax_tgt.transAxes, va='top')
        ax_tgt.text(0.02, 0.05, tgt_code, fontsize=9, family='monospace',
                   transform=ax_tgt.transAxes, va='bottom',
                   bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgreen', alpha=0.7))
        
        # Create prediction table
        table_data = []
        cols = ["Pos", "Source Token", "Target Token", "✓", "Top 1", "Top 2", "Top 3", "Top 4", "Top 5"]
        
        # Get sequence length
        seq_len = min(len(target_tokens_for_display), max_positions, aligned_logits.size(1))
        
        # Track accuracy and special token counts
        correct_predictions = 0
        total_predictions = 0
        special_tokens = {"<PAD>", "<UNK>", "<BOS>", "<EOS>", "<INDENT>", "<DEDENT>", "<NEWLINE>", "<COMMENT>"}
        
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
            
            # Check if prediction is correct
            predicted_token_id = top_tokens[example_idx, pos, 0].item()
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
            
            # Truncate long tokens for display
            src_token_display = truncate_token(src_token)
            tgt_token_display = truncate_token(tgt_token)
            preds_display = [truncate_token(p) for p in preds]
            
            table_data.append([f"{pos+1}", src_token_display, tgt_token_display, correct_symbol] + preds_display)

        # Create table with code-appropriate styling
        if table_data:
            table = ax_table.table(
                cellText=table_data,
                colLabels=cols,
                loc='center',
                cellLoc='center',
                colWidths=[0.04, 0.15, 0.15, 0.04] + [0.12]*5
            )
            table.auto_set_font_size(False)
            table.set_fontsize(7)
            table.scale(1, 1.8)
            
            # Color code cells based on token types and correctness
            for i, row in enumerate(table_data):
                # Color the correctness column
                cell_color = '#90EE90' if row[3] == "✓" else '#FFB6C1'
                table[(i+1, 3)].set_facecolor(cell_color)
                
                # Highlight special tokens
                src_token = row[1].split('\n')[0]  # Get token without probability
                tgt_token = row[2].split('\n')[0]
                
                if src_token in special_tokens:
                    table[(i+1, 1)].set_facecolor('#FFFFE0')  # Light yellow
                if tgt_token in special_tokens:
                    table[(i+1, 2)].set_facecolor('#FFFFE0')
        
        # Calculate accuracy
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
        
        # Add overall title
        sync_status = "synced" if is_synced else "unsynced"
        title_text = f"Code Translation Example {example_idx+1} ({sync_status}) - Token Accuracy: {accuracy:.1%} ({correct_predictions}/{total_predictions})"
        
        plt.suptitle(title_text, fontsize=14, y=0.98)
        
        # Convert to wandb image
        images.append(wandb.Image(fig))
        plt.close(fig)
    
    model.train()
    return images

def reconstruct_code_from_tokens(tokens: List[str], lang: str) -> str:
    """
    Reconstruct readable code from tokens, handling special tokens appropriately.
    """
    if not tokens:
        return "No tokens"
    
    # Special token mappings
    special_token_map = {
        "<NEWLINE>": "\n",
        "<INDENT>": "    ",  # 4 spaces
        "<DEDENT>": "",  # Handle dedent by reducing indentation (simplified)
        "<COMMENT>": "// Comment",
        "<PAD>": "",
        "<UNK>": "<??>",
        "<BOS>": "",
        "<EOS>": ""
    }
    
    lines = []
    current_line = []
    indent_level = 0
    
    for token in tokens:
        if token == "<NEWLINE>":
            # Finish current line
            if current_line:
                line_text = "".join(current_line)
                lines.append("    " * indent_level + line_text)
                current_line = []
        elif token == "<INDENT>":
            indent_level += 1
        elif token == "<DEDENT>":
            indent_level = max(0, indent_level - 1)
        elif token in special_token_map:
            replacement = special_token_map[token]
            if replacement:
                current_line.append(replacement)
        else:
            # Regular token - add appropriate spacing
            if current_line and needs_space_before(token, current_line[-1]):
                current_line.append(" ")
            current_line.append(token)
    
    # Add final line if exists
    if current_line:
        line_text = "".join(current_line)
        lines.append("    " * indent_level + line_text)
    
    # Join lines and wrap for display
    code_text = "\n".join(lines)
    
    # Limit length for display
    max_chars = 800
    if len(code_text) > max_chars:
        code_text = code_text[:max_chars] + "\n... (truncated)"
    
    return code_text if code_text.strip() else "Empty code"

def needs_space_before(current_token: str, previous_token: str) -> bool:
    """
    Determine if a space is needed between tokens when reconstructing code.
    """
    # Punctuation that typically doesn't need spaces
    no_space_before = {"(", ")", "[", "]", "{", "}", ";", ",", ".", ":", "++", "--"}
    no_space_after = {"(", "[", "{", ".", "->", "::", "++", "--"}
    
    if current_token in no_space_before or previous_token in no_space_after:
        return False
    
    # Operators typically need spaces
    if current_token in {"=", "==", "!=", "<=", ">=", "+", "-", "*", "/", "%", "&&", "||"}:
        return True
    
    # Keywords and identifiers typically need spaces
    if (previous_token.isalnum() or previous_token.endswith("_")) and (current_token.isalnum() or current_token.startswith("_")):
        return True
    
    return False

def truncate_token(token: str, max_length: int = 12) -> str:
    """
    Truncate long tokens for table display.
    """
    if len(token) <= max_length:
        return token
    return token[:max_length-2] + ".."

def generate_code_translation_examples(
    model: torch.nn.Module,
    batch: dict,
    tokenizer_interface: dict,
    device: torch.device,
    visualization_config: dict,
) -> List[wandb.Image]:
    """
    Generates side-by-side comparison of source and predicted target code.
    """
    src_id_to_token = tokenizer_interface["src_id_to_token"]
    tgt_id_to_token = tokenizer_interface["tgt_id_to_token"]
    padding_idx = tokenizer_interface["padding_idx"]
    num_examples = visualization_config.get("num_examples", 3)
    src_lang = tokenizer_interface.get("src_lang", "source")
    tgt_lang = tokenizer_interface.get("tgt_lang", "target")
    
    model.eval()
    images = []
    
    with torch.no_grad():
        if isinstance(batch, dict):
            src = batch["input_ids"].to(device)
            tgt = batch["labels"].to(device)
        else:
            src, tgt = [t.to(device) for t in batch]
        
        # Get model predictions
        is_synced = getattr(model, 'is_synced', True)
        
        if is_synced:
            outputs = model(src)
            raw_logits = outputs[0] if isinstance(outputs, tuple) else outputs
            predictions = torch.argmax(raw_logits, dim=-1)
        else:
            outputs = model(src, tgt)
            raw_logits = outputs[0] if isinstance(outputs, tuple) else outputs
            predictions = torch.argmax(raw_logits, dim=-1)
        
        batch_size = src.size(0)

    # Process each example
    for example_idx in range(min(batch_size, num_examples)):
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 8))
        
        # Source code
        src_tokens = [
            src_id_to_token.get(idx.item(), "<UNK>") 
            for idx in src[example_idx]
            if idx != padding_idx
        ]
        src_code = reconstruct_code_from_tokens(src_tokens, src_lang)
        
        # Ground truth target
        tgt_tokens = [
            tgt_id_to_token.get(idx.item(), "<UNK>") 
            for idx in tgt[example_idx]
            if idx != padding_idx
        ]
        tgt_code = reconstruct_code_from_tokens(tgt_tokens, tgt_lang)
        
        # Predicted target
        pred_tokens = [
            tgt_id_to_token.get(idx.item(), "<UNK>") 
            for idx in predictions[example_idx]
        ]
        pred_code = reconstruct_code_from_tokens(pred_tokens, tgt_lang)
        
        # Display source
        ax1.text(0.02, 0.98, f"Source ({src_lang.upper()})", fontsize=12, weight='bold', 
                transform=ax1.transAxes, va='top')
        ax1.text(0.02, 0.02, src_code, fontsize=8, family='monospace',
                transform=ax1.transAxes, va='bottom', wrap=True)
        ax1.set_facecolor('#E6F3FF')
        
        # Display ground truth
        ax2.text(0.02, 0.98, f"Ground Truth ({tgt_lang.upper()})", fontsize=12, weight='bold', 
                transform=ax2.transAxes, va='top')
        ax2.text(0.02, 0.02, tgt_code, fontsize=8, family='monospace',
                transform=ax2.transAxes, va='bottom', wrap=True)
        ax2.set_facecolor('#E6FFE6')
        
        # Display prediction
        ax3.text(0.02, 0.98, f"Prediction ({tgt_lang.upper()})", fontsize=12, weight='bold', 
                transform=ax3.transAxes, va='top')
        ax3.text(0.02, 0.02, pred_code, fontsize=8, family='monospace',
                transform=ax3.transAxes, va='bottom', wrap=True)
        ax3.set_facecolor('#FFE6E6')
        
        # Remove axes
        for ax in [ax1, ax2, ax3]:
            ax.set_xticks([])
            ax.set_yticks([])
            
        plt.suptitle(f"Code Translation Example {example_idx+1}: {src_lang} → {tgt_lang}", 
                    fontsize=14, y=0.95)
        plt.tight_layout()
        
        images.append(wandb.Image(fig))
        plt.close(fig)
    
    model.train()
    return images