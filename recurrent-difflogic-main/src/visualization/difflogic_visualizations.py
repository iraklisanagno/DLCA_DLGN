import io
import matplotlib.pyplot as plt
import numpy as np
import wandb
import torch
from PIL import Image

def calculate_entropy(probs):
    """Calculate Shannon entropy of the probability distribution"""
    # Calculate entropy
    ent = -sum(p * np.log2(p) if p > 0 else 0 for p in probs)
    # Normalize by maximum possible entropy (log2(16) for 16 functions)
    max_entropy = np.log2(len(probs))
    normalized_entropy = ent / max_entropy if max_entropy > 0 else 0
    return ent, normalized_entropy

def logic_distribution_visualization(model, batch=None, tokenizer_interface=None, 
                                    device=None, visualization_config=None):
    """
    Creates visualization of softmax probability distribution of logic functions across model layers.
    
    Args:
        model: The RecurrentDiffLogicModel to analyze
        batch: Not used for this visualization
        tokenizer_interface: Not used for this visualization
        device: Not used for this visualization
        visualization_config: Optional configuration for the visualization
        
    Returns:
        list: List of wandb.Image objects
    """
    # Get distribution data from model
    distribution = model.analyze_logic_function_distribution()
    
    # Set up parameters from config
    config = visualization_config or {}
    fig_width = config.get("fig_width", 12)
    fig_height_per_layer = config.get("fig_height_per_layer", 2.5)
    
    # Prepare wandb images list
    wandb_images = []
    
    # Create plots for each layer group
    for group_name, layers_info in distribution.items():
        if not layers_info:
            continue
            
        # Create figure with subplots - one per layer
        fig, axes = plt.subplots(len(layers_info), 1, 
                                 figsize=(fig_width, fig_height_per_layer * len(layers_info)),
                                 squeeze=False)
        fig.suptitle(f"Logic Function Distribution (Softmax): {group_name.upper().replace('_', ' ')}", 
                     fontsize=16)
        
        # Plot each layer's distribution
        for i, layer_info in enumerate(layers_info):
            ax = axes[i, 0]
            
            # Use softmax probabilities instead of argmax counts
            probs = layer_info['avg_argmax_probs']
            function_names = layer_info['function_names']
            x = np.arange(16)
            bars = ax.bar(x, probs)
            
            # Get softmax entropy - mean of per-neuron entropies
            entropy_softmax = layer_info['entropy_softmax']
            max_entropy = np.log2(16)  # 16 possible functions
            norm_entropy = entropy_softmax / max_entropy
            
            # Highlight the most probable function
            most_probable_idx = np.argmax(probs)
            bars[most_probable_idx].set_color('blue')
            
            # Add annotations
            most_prob_name = function_names[most_probable_idx]
            most_prob_value = probs[most_probable_idx]
            ax.set_title(f"Layer {i} (size {layer_info['out_dim']}) - " + 
                         f"Highest prob: {most_prob_name} ({most_prob_value:.3f}) - " +
                         f"Avg Neuron Entropy: {entropy_softmax:.2f} bits", fontsize=10)
            ax.set_xlabel("Logic Function")
            ax.set_ylabel("Average Probability")
            ax.set_xticks(x)
            ax.set_xticklabels(function_names, rotation=90)
            
            # Add entropy as text in the plot with clearer description
            ax.text(0.02, 0.95, 
                   f"Avg Neuron Entropy: {entropy_softmax:.2f} bits\n"
                   f"Normalized (÷ max 4 bits): {norm_entropy:.2f}",
                   transform=ax.transAxes,
                   horizontalalignment='left',
                   verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # Add probability labels for significant values
            threshold = max(probs) * 0.3  # Higher threshold for readability
            for j, prob in enumerate(probs):
                if prob > threshold:
                    ax.text(j, prob, f"{prob:.3f}", ha='center', va='bottom', fontsize=8)
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        # Convert figure to PIL Image and then to wandb.Image
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=100)
        buf.seek(0)
        pil_img = Image.open(buf)
        wandb_images.append(wandb.Image(pil_img, caption=f"{group_name} Softmax Distribution"))
        plt.close(fig)
        buf.close()
    
    return wandb_images

def neuron_weight_histogram(model, batch=None, tokenizer_interface=None, 
                          device=None, visualization_config=None):
    """
    Creates histogram visualizations of the weight distributions for each layer type.
    Shows the average per-neuron entropy for the softmax weights.
    
    Args:
        model: The RecurrentDiffLogicModel to analyze
        batch: Not used for this visualization
        tokenizer_interface: Not used for this visualization
        device: Not used for this visualization
        visualization_config: Optional configuration for the visualization
        
    Returns:
        list: List of wandb.Image objects
    """
    wandb_images = []
    
    # Get distribution data from model for entropy values
    distribution = model.analyze_logic_function_distribution()
    
    # Analyze weights for each layer type
    layer_groups = {}
    
    # Dynamically determine available layer groups based on model structure
    if hasattr(model, 'n_layers'):
        layer_groups['N Layers (Input Processing)'] = model.n_layers
    if hasattr(model, 'k_layers'):
        layer_groups['K Layers (Recurrent Memory)'] = model.k_layers
    if hasattr(model, 'l_layers'):
        layer_groups['L Layers (Backward Processing)'] = model.l_layers
    if hasattr(model, 'm_layers'):
        layer_groups['M Layers (Output Generation)'] = model.m_layers
    if hasattr(model, 'p_layers'):
        layer_groups['P Layers (Combined Processing)'] = model.p_layers
    if hasattr(model, 'logic_layers'):
        layer_groups['Logic Layers'] = model.logic_layers
    
    # Map layer groups to distribution keys
    layer_to_dist = {
        'N Layers (Input Processing)': 'n_layers',
        'K Layers (Recurrent Memory)': 'k_layers',
        'L Layers (Backward Processing)': 'l_layers',
        'M Layers (Output Generation)': 'm_layers',
        'P Layers (Combined Processing)': 'p_layers',
        'Logic Layers': 'logic_layers'
    }
    
    for group_name, layers in layer_groups.items():
        if not layers:
            continue
            
        dist_key = layer_to_dist.get(group_name)
        
        fig, axes = plt.subplots(len(layers), 1, figsize=(10, 3*len(layers)))
        if len(layers) == 1:
            axes = [axes]
            
        fig.suptitle(f"Weight Distribution: {group_name}", fontsize=16)
        
        for i, layer in enumerate(layers):
            # Get the weights and convert to numpy for plotting
            weights = layer.weights.detach().cpu().numpy()
            
            # Try to get entropy from distribution data
            avg_entropy = None
            if dist_key and dist_key in distribution and i < len(distribution[dist_key]):
                avg_entropy = distribution[dist_key][i].get('entropy_softmax')
            
            # If not available, calculate it directly
            if avg_entropy is None:
                softmax_weights = np.exp(weights) / np.exp(weights).sum(axis=1, keepdims=True)
                entropies = [-np.sum(w * np.log2(w + 1e-10)) for w in softmax_weights]
                avg_entropy = np.mean(entropies)
                
            max_possible = np.log2(16)  # 16 possible functions
            norm_entropy = avg_entropy / max_possible
            
            # Plot histogram of weight values
            ax = axes[i]
            ax.hist(weights.flatten(), bins=50, alpha=0.7)
            ax.set_title(f"Layer {i} - Weights - Avg Neuron Entropy: {avg_entropy:.3f} bits", 
                       fontsize=10)
            ax.set_xlabel("Weight Value")
            ax.set_ylabel("Count")
            
            # Add statistics with clearer entropy description
            ax.text(0.98, 0.95, 
                   f"Mean: {weights.mean():.3f}\nStd: {weights.std():.3f}\n"
                   f"Min: {weights.min():.3f}\nMax: {weights.max():.3f}\n"
                   f"Avg Neuron Entropy: {avg_entropy:.3f} bits\n"
                   f"Normalized (÷ max 4 bits): {norm_entropy:.3f}",
                   transform=ax.transAxes,
                   horizontalalignment='right',
                   verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        # Convert figure to PIL Image then to wandb.Image
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=100)
        buf.seek(0)
        pil_img = Image.open(buf)
        wandb_images.append(wandb.Image(pil_img, caption=f"{group_name} Weight Distribution"))
        plt.close(fig)
        buf.close()
        
    return wandb_images