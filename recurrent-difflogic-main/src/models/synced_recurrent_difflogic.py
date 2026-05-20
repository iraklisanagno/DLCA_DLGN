import torch
import torch.nn as nn
import torch.nn.functional as F
from difflogic import LogicLayer as OriginalLogicLayer, GroupSum, PackBitsTensor
from utils.channel_logic_layer import ChannelLogicLayer, get_cuda_capability 
from models.base_model import BaseModel
from utils.gumbel_logic_wrapper import patch_logic_layers_with_gumbel


class SyncedRecurrentDiffLogicModel(BaseModel):
    def __init__(
        self, num_input_tokens, embedding_dim, seq_length,
        n_layers_sizes, k_layers_sizes, m_layers_sizes, num_classes,
        group_factor=1,
        device="cuda", grad_factor=1.0,
        connections='random', predefined_indices=None,
        difflogic_init_type='noisy_residual',  # 'gaussian',"noisy_residual" or 'residual'
        noise_factor=1.0,         # New parameter for Gumbel noise scaling
        hidden_state_init_type='zero',   # 'zero', 'one', 'gaussian', 'uniform', 'learnable'
        seed=None,  # New parameter for setting the random seed
        group_sum_tau=1.0,  # New parameter for softmax tau
        gumbel_tau=None,  # New parameter for Gumbel-Softmax temperature
        use_st_estimator=True,  # Whether to use straight-through estimator with Gumbel-Softmax
        dropout_prob=0.2,  # New parameter for dropout probability
        padding_idx=0, bos_token_id=2, eos_token_id=3,
        frozen_layers=None,
        noise_layers=None,
        noise_std=0.01,
        use_channel_logic=False,
        channel_logic_k=2,
        channel_regularization_lambda=0.0,
        use_compiled_layers=True,
        gradient_rescaling=None,
        adaptive_gradient_clipping=False,
    ):
        super().__init__()

        # Set random seed for reproducibility
        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        self.device = device
        self.seq_length = seq_length
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.num_input_tokens = num_input_tokens
        self.group_factor = group_factor
        self.connections = connections
        self.difflogic_init_type = difflogic_init_type
        self.noise_factor = noise_factor  # Store the noise factor
        self.hidden_state_init_type = hidden_state_init_type
        self.group_sum_tau = group_sum_tau  # Store the tau parameter for softmax in GroupSum
        self.gumbel_tau = gumbel_tau  # Store the gumbel tau parameter
        self.use_st_estimator = use_st_estimator  # Store the ST estimator flag
        self.grad_factor = grad_factor
        self.dropout_prob = dropout_prob  # Store dropout probability
        self.padding_idx = padding_idx
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        
        # Channel logic parameters
        self.use_channel_logic = use_channel_logic
        self.channel_logic_k = channel_logic_k
        self.channel_regularization_lambda = channel_regularization_lambda
        self.use_compiled_layers = use_compiled_layers

        # Layer freezing and noise injection parameters
        self.frozen_layers = frozen_layers or {}
        self.noise_layers = noise_layers or {}
        self.noise_std = noise_std
        self.gradient_hooks = []

        # Gradient management (no LR multipliers)
        self.gradient_rescaling = gradient_rescaling or {}
        self.adaptive_gradient_clipping = adaptive_gradient_clipping
        self.gradient_stats = {}  # Track gradient statistics per layer group
        self.rescaling_hooks = []  # Hooks for gradient rescaling
        
        # Validate layer sizes
        assert len(n_layers_sizes) > 0 and len(k_layers_sizes) > 0 and len(m_layers_sizes) > 0, \
            "All layer groups must have at least one layer"

        # Prepare final M-layer size
        final_m_out = num_classes * group_factor
        m_layers_sizes.append(final_m_out)

        # Choose logic layer based on use_channel_logic
        if use_channel_logic:
            LogicLayer = ChannelLogicLayer
            print(f"Using fast ChannelLogic layers with k={channel_logic_k}")
            if use_compiled_layers:
                print("Attempting to compile ChannelLogic layers for maximum speed...")
        else:
            LogicLayer = OriginalLogicLayer
            print(f"Using standard DiffLogic layers")

        # Embedding layer
        self.embedding = nn.Embedding(num_input_tokens, embedding_dim, padding_idx=padding_idx)
        
        # Dropout layers
        self.emb_dropout = nn.Dropout(dropout_prob)  # Dropout after embedding
        self.n_dropouts = nn.ModuleList([nn.Dropout(dropout_prob) for _ in n_layers_sizes])  # Dropouts after N layers
        self.k_dropouts = nn.ModuleList([nn.Dropout(dropout_prob) for _ in k_layers_sizes])  # Dropouts after K layers
        self.m_dropouts = nn.ModuleList([nn.Dropout(dropout_prob) for _ in m_layers_sizes])  # Dropouts after M layers

        # Create all layers first with regular LogicLayer
        # N Layers
        self.n_layers = nn.ModuleList()
        prev_dim = embedding_dim
        for size in n_layers_sizes:
            layer = self._create_logic_layer(LogicLayer, prev_dim, size, layer_type='n')
            self.n_layers.append(layer)
            prev_dim = size
        self.n_out = prev_dim

        # K Layers
        self.k_layers = nn.ModuleList()
        k_in_dim = self.n_out + k_layers_sizes[-1]
        for i, size in enumerate(k_layers_sizes):
            # Check LogicLayer input/output dimension condition
            if i == 0:
                required_min_out = (self.n_out + k_layers_sizes[-1]) // 2
                assert size >= required_min_out, (
                    f"First K layer size must be at least {required_min_out} to satisfy "
                    f"2*out_dim >= in_dim (in_dim={self.n_out + k_layers_sizes[-1]})."
                )
            layer = self._create_logic_layer(LogicLayer, k_in_dim, size, layer_type='k')
            self.k_layers.append(layer)
            k_in_dim = size
        self.hidden_dim = k_layers_sizes[-1]

        # Initialize persistent initial hidden state
        if self.hidden_state_init_type == 'learnable':
            # Apply sigmoid to ensure values are between 0 and 1
            self.initial_hidden = nn.Parameter(
                torch.sigmoid(torch.randn(self.hidden_dim, device=self.device))
            )
        else:
            if self.hidden_state_init_type == 'zero':
                buf = torch.zeros(self.hidden_dim, device=self.device)
            elif self.hidden_state_init_type == 'one':
                buf = torch.ones(self.hidden_dim, device=self.device)
            elif self.hidden_state_init_type == 'gaussian':
                # Apply sigmoid to ensure values are between 0 and 1
                buf = torch.sigmoid(torch.randn(self.hidden_dim, device=self.device))
            elif self.hidden_state_init_type == 'uniform':
                buf = torch.rand(self.hidden_dim, device=self.device)
            else:
                raise ValueError(f"Unknown hidden_state_init_type: {self.hidden_state_init_type}")
            self.register_buffer('initial_hidden', buf)

        # M Layers
        self.m_layers = nn.ModuleList()
        m_in_dim = self.n_out + self.hidden_dim
        for size in m_layers_sizes:
            layer = self._create_logic_layer(LogicLayer, m_in_dim, size, layer_type='m')
            self.m_layers.append(layer)
            m_in_dim = size

        # Final grouping
        self.final_sum = GroupSum(k=num_classes, tau=group_sum_tau, device=device)
        
        # Apply Gumbel-Softmax patching if gumbel_tau is provided
        if gumbel_tau is not None and not use_channel_logic:
            patch_logic_layers_with_gumbel(
                self, tau=gumbel_tau, use_st_estimator=use_st_estimator, verbose=False
            )

        # Compile layers if requested and using PyTorch 2.0+
        if use_channel_logic and use_compiled_layers:
            self._compile_channel_layers()

        # Apply freezing and noise injection
        self._apply_layer_modifications()
        
        # Setup gradient management (no LR multipliers)
        self._setup_gradient_management()
            
        self.set_mode('train')

    def _create_logic_layer(self, LogicLayer, in_dim, out_dim, layer_type=None):
        """Create a logic layer with appropriate parameters."""
        if self.use_channel_logic:
            layer = LogicLayer(
                in_dim=in_dim, 
                out_dim=out_dim, 
                k=self.channel_logic_k,
                device=self.device,
                grad_factor=self.grad_factor, 
                connections=self.connections,
                regularization_lambda=self.channel_regularization_lambda
            )
        else:
            layer = LogicLayer(
                in_dim=in_dim, 
                out_dim=out_dim, 
                device=self.device,
                grad_factor=self.grad_factor, 
                implementation='cuda', 
                connections=self.connections
            )
            self._init_logic_layer(layer, layer_type)
        
        return layer

    def _compile_channel_layers(self):
        """Compile ChannelLogic layers for maximum performance with GPU capability detection."""
        try:
            # Check CUDA capability
            cuda_capability = get_cuda_capability()
            min_required_capability = 7.0  # Triton requires CUDA Capability >= 7.0
            
            if cuda_capability < min_required_capability:
                print(f"GPU CUDA Capability {cuda_capability} < {min_required_capability} required for compilation.")
                print("Falling back to standard (non-compiled) ChannelLogic implementation.")
                print("Performance will still be excellent with the optimized k=2 implementation!")
                return
            
            # Compile all channel logic layers
            compiled_count = 0
            for layer_group in [self.n_layers, self.k_layers, self.m_layers]:
                for layer in layer_group:
                    if isinstance(layer, ChannelLogicLayer):
                        layer.forward = torch.compile(layer.forward, mode='max-autotune')
                        compiled_count += 1
            
            print(f"Successfully compiled {compiled_count} ChannelLogic layers for maximum performance!")
            
        except AttributeError:
            print("PyTorch compile not available, using standard ChannelLogic implementation")
            print("Performance will still be excellent with the optimized k=2 implementation!")
        except Exception as e:
            print(f"Warning: Could not compile layers: {e}")
            print("Falling back to standard (non-compiled) ChannelLogic implementation.")
            print("Performance will still be excellent with the optimized k=2 implementation!")

    def _setup_gradient_management(self):
        """Setup gradient rescaling and adaptive clipping."""
        if self.gradient_rescaling or self.adaptive_gradient_clipping:
            print("🔧 Setting up gradient management...")
        
        # Apply gradient rescaling if specified
        if self.gradient_rescaling:
            print("🎯 Layer-wise gradient rescaling:")
            for layer_group, scale_factor in self.gradient_rescaling.items():
                print(f"  {layer_group}: {scale_factor:.1f}x")
                self._apply_gradient_rescaling_to_layer_group(layer_group, scale_factor)
        
        # Setup adaptive gradient clipping if enabled
        if self.adaptive_gradient_clipping:
            print("⚡ Adaptive gradient clipping enabled")
            self._setup_adaptive_gradient_clipping()

    def _apply_gradient_rescaling_to_layer_group(self, layer_group_name, scale_factor):
        """Apply gradient rescaling to a specific layer group."""
        if hasattr(self, layer_group_name):
            layers = getattr(self, layer_group_name)
            
            # Handle single layers (like embedding) vs layer collections
            if isinstance(layers, (nn.ModuleList, list)):
                # Multiple layers in a group
                for layer in layers:
                    for param in layer.parameters():
                        if param.requires_grad:
                            hook = param.register_hook(self._create_gradient_rescaling_hook(scale_factor))
                            self.rescaling_hooks.append(hook)
            else:
                # Single layer (like embedding)
                for param in layers.parameters():
                    if param.requires_grad:
                        hook = param.register_hook(self._create_gradient_rescaling_hook(scale_factor))
                        self.rescaling_hooks.append(hook)

    def _create_gradient_rescaling_hook(self, scale_factor):
        """Create a gradient rescaling hook."""
        def rescaling_hook(grad):
            if grad is not None:
                return grad * scale_factor
            return grad
        return rescaling_hook

    def _setup_adaptive_gradient_clipping(self):
        """Setup adaptive gradient clipping based on layer gradient statistics."""
        layer_groups = ['embedding', 'n_layers', 'k_layers', 'm_layers']
        
        for layer_group_name in layer_groups:
            if hasattr(self, layer_group_name):
                layers = getattr(self, layer_group_name)
                
                # Handle single layers (like embedding) vs layer collections
                if isinstance(layers, (nn.ModuleList, list)):
                    # Multiple layers in a group
                    for layer in layers:
                        for param in layer.parameters():
                            if param.requires_grad:
                                hook = param.register_hook(
                                    self._create_adaptive_clipping_hook(layer_group_name)
                                )
                                self.rescaling_hooks.append(hook)
                else:
                    # Single layer (like embedding)
                    for param in layers.parameters():
                        if param.requires_grad:
                            hook = param.register_hook(
                                self._create_adaptive_clipping_hook(layer_group_name)
                            )
                            self.rescaling_hooks.append(hook)

    def _create_adaptive_clipping_hook(self, layer_group_name):
        """Create adaptive gradient clipping hook that adjusts based on gradient statistics."""
        def adaptive_clipping_hook(grad):
            if grad is not None:
                # Track gradient statistics
                if layer_group_name not in self.gradient_stats:
                    self.gradient_stats[layer_group_name] = {
                        'mean_norm': 0.0,
                        'std_norm': 0.0,
                        'count': 0
                    }
                
                grad_norm = grad.norm().item()
                stats = self.gradient_stats[layer_group_name]
                
                # Update running statistics
                stats['count'] += 1
                alpha = 0.1  # Exponential moving average factor
                stats['mean_norm'] = (1 - alpha) * stats['mean_norm'] + alpha * grad_norm
                stats['std_norm'] = (1 - alpha) * stats['std_norm'] + alpha * (grad_norm - stats['mean_norm']) ** 2
                
                # Adaptive clipping threshold
                threshold = stats['mean_norm'] + 2 * (stats['std_norm'] ** 0.5)
                if grad_norm > threshold and threshold > 0:
                    grad = grad * (threshold / grad_norm)
                
                return grad
            return grad
        return adaptive_clipping_hook

    def get_layerwise_gradient_stats(self):
        """Get current gradient statistics for all layer groups."""
        stats = {}
        layer_groups = ['n_layers', 'k_layers', 'm_layers', 'embedding']
        
        for layer_group_name in layer_groups:
            if hasattr(self, layer_group_name):
                layers = getattr(self, layer_group_name) if isinstance(getattr(self, layer_group_name), (list, nn.ModuleList)) else [getattr(self, layer_group_name)]
                
                total_grad_norm = 0.0
                param_count = 0
                
                for layer in layers:
                    for param in layer.parameters():
                        if param.requires_grad and param.grad is not None:
                            total_grad_norm += param.grad.norm().item() ** 2
                            param_count += 1
                
                if param_count > 0:
                    avg_grad_norm = (total_grad_norm / param_count) ** 0.5
                    stats[layer_group_name] = {
                        'avg_grad_norm': avg_grad_norm,
                        'param_count': param_count
                    }
        
        return stats

    def set_gradient_rescaling(self, gradient_rescaling):
        """Dynamically set gradient rescaling factors."""
        # Clear existing hooks
        for hook in self.rescaling_hooks:
            hook.remove()
        self.rescaling_hooks = []
        
        # Update rescaling
        self.gradient_rescaling = gradient_rescaling or {}
        
        # Reapply
        self._setup_gradient_management()

    def apply_vanishing_gradient_fix(self, gradient_ratios=None):
        """
        Apply automatic vanishing gradient fix based on gradient analysis.
        
        Args:
            gradient_ratios: Dict with layer_group -> ratio (e.g., {'n_layers': 0.03, 'k_layers': 0.03})
        """
        if gradient_ratios is None:
            # Default ratios based on the provided analysis
            gradient_ratios = {
                'n_layers': 0.03,
                'k_layers': 0.03, 
                'm_layers': 0.65
            }
        
        print("🚨 Applying vanishing gradient fix based on gradient ratios:")
        
        # Apply gradient rescaling for severely affected layers
        rescaling = {}
        for layer_group, ratio in gradient_ratios.items():
            if ratio < 0.05:  # Very small gradients
                rescaling[layer_group] = 10.0  # 10x gradient boost
                print(f"  {layer_group}: ratio={ratio:.3f} -> 10x gradient rescaling applied")
            elif ratio < 0.1:  # Small gradients
                rescaling[layer_group] = 5.0  # 5x gradient boost
                print(f"  {layer_group}: ratio={ratio:.3f} -> 5x gradient rescaling applied")
        
        if rescaling:
            self.set_gradient_rescaling(rescaling)

    def _apply_layer_modifications(self):
        """Apply freezing and noise injection to specified layers"""
        # Clear existing hooks
        for hook in self.gradient_hooks:
            hook.remove()
        self.gradient_hooks = []

        # Apply freezing - layers stay frozen as configured
        for layer_group, layer_indices in self.frozen_layers.items():
            if hasattr(self, layer_group):
                layers = getattr(self, layer_group)
                for idx in layer_indices:
                    if idx < len(layers):
                        # Freeze all parameters in this layer
                        for param in layers[idx].parameters():
                            param.requires_grad = False
                        print(f"❄️  Frozen {layer_group}[{idx}] - parameters will remain frozen")

        # Apply noise injection
        for layer_group, layer_indices in self.noise_layers.items():
            if hasattr(self, layer_group):
                layers = getattr(self, layer_group)
                for idx in layer_indices:
                    if idx < len(layers):
                        # Register backward hooks for noise injection
                        for param in layers[idx].parameters():
                            if param.requires_grad:
                                hook = param.register_hook(self._create_noise_hook())
                                self.gradient_hooks.append(hook)
                        print(f"🔊 Noise injection applied to {layer_group}[{idx}]")

    def _create_noise_hook(self):
        """Create a hook function that replaces gradients with noise"""
        def noise_hook(grad):
            if grad is not None:
                noise = torch.randn_like(grad) * self.noise_std
                return noise
            return grad
        return noise_hook

    def set_frozen_layers(self, frozen_layers):
        """Set which layers should be frozen (not trainable)"""
        self.frozen_layers = frozen_layers or {}
        self._apply_layer_modifications()

    def set_noise_layers(self, noise_layers, noise_std=None):
        """Set which layers should have their gradients replaced with noise"""
        self.noise_layers = noise_layers or {}
        if noise_std is not None:
            self.noise_std = noise_std
        self._apply_layer_modifications()

    def unfreeze_all_layers(self):
        """Unfreeze all layers and remove noise injection"""
        # Clear hooks
        for hook in self.gradient_hooks:
            hook.remove()
        self.gradient_hooks = []
        
        # Clear rescaling hooks
        for hook in self.rescaling_hooks:
            hook.remove()
        self.rescaling_hooks = []
        
        # Unfreeze all parameters
        for param in self.parameters():
            param.requires_grad = True
            
        # Clear frozen and noise layer configurations
        self.frozen_layers = {}
        self.noise_layers = {}

    def get_layer_training_status(self):
        """Get the training status of all layers"""
        status = {}
        
        for layer_group_name in ['n_layers', 'k_layers', 'm_layers']:
            if hasattr(self, layer_group_name):
                layers = getattr(self, layer_group_name)
                group_status = {
                    'total_layers': len(layers),
                    'frozen': [],
                    'noise': [],
                    'trainable': [],
                    'gradient_rescaling': self.gradient_rescaling.get(layer_group_name, 1.0)
                }
                
                for i, layer in enumerate(layers):
                    is_frozen = not any(p.requires_grad for p in layer.parameters())
                    is_noise = layer_group_name in self.noise_layers and i in self.noise_layers[layer_group_name]
                    
                    if is_frozen:
                        group_status['frozen'].append(i)
                    elif is_noise:
                        group_status['noise'].append(i)
                    else:
                        group_status['trainable'].append(i)
                
                status[layer_group_name] = group_status
        
        return status

    def compute_logic_entropy_loss(self, target_entropy=None, entropy_direction='minimize'):
        """
        Compute entropy loss for logic layer weights - works for both DiffLogic and ChannelLogic
        """
        if self.use_channel_logic:
            # For ChannelLogic layers, entropy loss is not applicable in the same way
            # Return zero loss to maintain compatibility
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        
        total_entropy = 0.0
        total_neurons = 0
        
        for layer_group in [self.n_layers, self.k_layers, self.m_layers]:
            for layer in layer_group:
                if hasattr(layer, 'weights'):
                    # For standard layers, compute entropy over function weights
                    # Shape: [out_dim, 16] -> each neuron has 16 function weights
                    probs = F.softmax(layer.weights, dim=1)  # [out_dim, 16]
                    # Compute entropy for each neuron (each row)
                    neuron_entropies = -torch.sum(probs * torch.log(probs + 1e-10), dim=1)  # [out_dim]
                    total_entropy += torch.sum(neuron_entropies)
                    total_neurons += layer.weights.size(0)  # out_dim
                else:
                    continue
        
        if total_neurons == 0:
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        
        # PROPERLY NORMALIZED: Average entropy across ALL neurons in ALL layers
        avg_entropy = total_entropy / total_neurons
        
        if target_entropy is not None:
            # Return absolute difference from target
            result = torch.abs(avg_entropy - target_entropy)
        elif entropy_direction == 'minimize':
            # Return entropy directly (minimizing it)
            result = avg_entropy
        else:  # maximize
            # Return negative entropy (minimizing negative entropy = maximizing entropy)
            result = -avg_entropy
        
        # The entropy is now properly normalized per neuron, so less aggressive scaling needed
        # For 16 functions, max entropy = log2(16) = 4.0, so this is already well-scaled
        return result / 4.0  # Normalize by theoretical maximum entropy

    def compute_weight_magnitude_loss(self):
        """
        Compute loss that specifically penalizes high weights for indices 3 and 15 (FALSE and TRUE functions).
        Only applicable to DiffLogic layers.
        """
        if self.use_channel_logic:
            # For ChannelLogic layers, this loss is not applicable
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        
        total_penalty = torch.tensor(0.0, device=self.device)
        total_neurons = 0
        
        # Indices to penalize: 3 = FALSE (constant 0), 15 = TRUE (constant 1)
        bias_indices = [3, 15]
        
        for layer_group in [self.n_layers, self.k_layers, self.m_layers]:
            for layer in layer_group:
                if hasattr(layer, 'weights') and layer.weights.size(-1) == 16:
                    # Standard difflogic layers with 16 weights per neuron
                    # Shape: [out_dim, 16]
                    
                    # Extract weights for bias functions (indices 3 and 15)
                    bias_weights = layer.weights[:, bias_indices]  # [out_dim, 2]
                    
                    # Penalize high positive values for these specific indices
                    # Take mean across the bias indices for each neuron, then ReLU to only penalize positive values
                    neuron_bias_penalty = torch.relu(torch.mean(bias_weights, dim=1))  # [out_dim]
                    
                    # Sum penalty across all neurons in this layer
                    total_penalty += torch.sum(neuron_bias_penalty)
                    total_neurons += layer.weights.size(0)  # out_dim
        
        if total_neurons == 0:
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        
        # Average penalty across all neurons
        avg_penalty = total_penalty / total_neurons
        
        # Scale the loss to be in a reasonable range
        # Since we're only targeting specific indices, the penalty might be smaller, so we use sqrt scaling
        return torch.sqrt(avg_penalty + 1e-8)

    def get_channel_regularization_loss(self):
        """Get regularization loss from ChannelLogic layers."""
        if not self.use_channel_logic:
            return torch.tensor(0.0, device=self.device)
        
        total_reg_loss = torch.tensor(0.0, device=self.device)
        
        for layer_group in [self.n_layers, self.k_layers, self.m_layers]:
            for layer in layer_group:
                if hasattr(layer, 'compute_regularization_loss'):
                    total_reg_loss += layer.compute_regularization_loss()
        
        return total_reg_loss

    def get_total_parameters(self):
        """Get total parameter count."""
        total_params = sum(p.numel() for p in self.parameters())
        layer_type = "ChannelLogic" if self.use_channel_logic else "DiffLogic"
        print(f"Total {layer_type} model parameters: {total_params:,}")
        return total_params

    def _init_logic_layer(self, layer, layer_type=None):
        """Initialize logic layer weights based on the layer type. Only for DiffLogic layers."""
        if self.use_channel_logic:
            # ChannelLogic layers handle their own initialization
            return
            
        with torch.no_grad():
            if hasattr(layer, 'weights'):
                if self.difflogic_init_type == 'residual':
                    layer.weights.data.zero_()
                    if layer.weights.size(1) > 3:
                        layer.weights.data[:, 3].fill_(5.0)
                elif self.difflogic_init_type == 'noisy_residual':
                    layer.weights.data.normal_(mean=0.0, std=self.noise_factor)
                    if layer.weights.size(1) > 3:
                        layer.weights.data[:, 3].add_(5.0)  
                elif self.difflogic_init_type == 'gaussian':
                    layer.weights.data.normal_(mean=0.0, std=self.noise_factor)
                elif self.difflogic_init_type == 'custom_gaussian':
                    # Custom initialization mode
                    # Initialize weights 0 and 15 with 0
                    layer.weights.data[:, 0] = 0.0  # FALSE function
                    layer.weights.data[:, 15] = 0.0  # TRUE function
                    
                    # Initialize all other weights with gaussian mean 1
                    for i in range(1, 15):
                        layer.weights.data[:, i].normal_(mean=1.0, std=self.noise_factor)
                    
                    # Special handling for weights 3 and 5 based on layer type
                    if layer_type in ['k']:
                        # For K layers: weights 3 and 5 have mean 2
                        layer.weights.data[:, 3].normal_(mean=2.0, std=self.noise_factor)
                        layer.weights.data[:, 5].normal_(mean=2.0, std=self.noise_factor)
                    else:
                        # For all other layers (N, M): weights 3 and 5 have mean 0
                        layer.weights.data[:, 3].normal_(mean=0.0, std=self.noise_factor)
                        layer.weights.data[:, 5].normal_(mean=0.0, std=self.noise_factor)

    def _init_hidden(self, batch_size):
        return self.initial_hidden.unsqueeze(0).expand(batch_size, -1)

    def forward(self, x):
        """Float forward pass"""
        batch_size, seq_len = x.size(0), x.size(1)
        embedded = (self.embedding(x) > 0).float() if self.emb_mode else torch.sigmoid(self.embedding(x))
        
        # Apply dropout to embedding output
        embedded = self.emb_dropout(embedded)
        
        hidden = self._init_hidden(batch_size)
        all_outputs = []

        for t in range(seq_len):
            x_step = embedded[:, t, :]
            
            # N layers (float) with dropout after each layer
            for i, layer in enumerate(self.n_layers):
                x_step = layer(x_step)
                x_step = self.n_dropouts[i](x_step)  # Apply dropout
                
            combined = torch.cat([x_step, hidden], dim=1)

            # K layers with dropout after each layer
            k_out = combined
            for i, layer in enumerate(self.k_layers):
                k_out = layer(k_out)
                k_out = self.k_dropouts[i](k_out)  # Apply dropout
                
            hidden = k_out

            # M layers with dropout after each layer
            m_out = combined
            for i, layer in enumerate(self.m_layers):
                m_out = layer(m_out)
                m_out = self.m_dropouts[i](m_out)  # Apply dropout
            
            # Apply GroupSum followed by softmax with tau
            group_out = self.final_sum(m_out)
            all_outputs.append(group_out)

        binary_reg_loss = torch.mean(embedded * (1 - embedded))

        # Add channel regularization if using ChannelLogic
        if self.use_channel_logic:
            channel_reg_loss = self.get_channel_regularization_loss()
            binary_reg_loss = binary_reg_loss + channel_reg_loss

        return torch.stack(all_outputs, dim=1), binary_reg_loss

    def set_mode(self, mode):
        assert mode in ['train', 'eval','eval_col_emb','eval_col_layer','eval_col_all'], "Mode must be 'train' or 'eval'"
        self.train() if mode in ['train', 'eval','eval_col_emb'] else self.eval()
        self.emb_mode= mode in ['eval_col_emb','eval_col_all']

    def analyze_logic_function_distribution(self):
        """
        Analyzes the distribution of logic functions used in each layer of the model.
        Calculates per-neuron entropy of softmax probabilities and averages across the layer.
        
        Returns:
            dict: A dictionary containing distribution information for all layers
        """
        if self.use_channel_logic:
            print("Logic function distribution analysis not applicable for ChannelLogic layers.")
            return {}
        
        import torch.nn.functional as F
        
        # Define names for the 16 binary logic functions
        function_names = [
            "FALSE", "AND", "A & ~B", "A", "~A & B", "B", "XOR", "OR",
            "NOR", "XNOR", "~B", "A | ~B", "~A", "~A | B", "NAND", "TRUE"
        ]
        
        # Dictionary to store the results
        distribution = {
            'n_layers': [],
            'k_layers': [],
            'm_layers': []
        }
        
        # Analyze each layer group
        for layer_group_name, layer_group in [
            ('n_layers', self.n_layers), 
            ('k_layers', self.k_layers), 
            ('m_layers', self.m_layers)
        ]:
            for i, layer in enumerate(layer_group):
                if hasattr(layer, 'weights'):
                    # Get the most probable function for each neuron using argmax
                    func_indices = layer.weights.argmax(dim=1)
                    counts = torch.bincount(func_indices, minlength=16).tolist()
                    
                    # Calculate softmax probabilities
                    softmax_probs = F.softmax(layer.weights, dim=1)  # [out_dim, 16]
                    # Mean probabilities across all neurons
                    avg_softmax = softmax_probs.mean(dim=0).tolist()
                    
                    # Calculate argmax distribution (percentage of neurons with each function as their argmax)
                    argmax_counts = torch.bincount(func_indices, minlength=16).float()
                    avg_argmax_probs = (argmax_counts / layer.out_dim).tolist()
                    
                    # Calculate entropy for each neuron individually, then average
                    neuron_entropies = []
                    for n in range(softmax_probs.size(0)):
                        neuron_probs = softmax_probs[n]
                        neuron_entropy = -torch.sum(neuron_probs * torch.log2(neuron_probs + 1e-10))
                        neuron_entropies.append(neuron_entropy.item())
                    
                    # Average entropy across all neurons in the layer
                    mean_neuron_entropy = sum(neuron_entropies) / len(neuron_entropies)
                    
                    # Store the statistics
                    layer_info = {
                        'layer_idx': i,
                        'layer_name': f"{layer_group_name}_{i}",
                        'out_dim': layer.out_dim,
                        'function_counts': counts,
                        'function_names': function_names,
                        'most_used_function': counts.index(max(counts)),
                        'most_used_count': max(counts),
                        'percentage_most_used': max(counts) / layer.out_dim * 100,
                        'avg_softmax_probs': avg_softmax,
                        'avg_argmax_probs': avg_argmax_probs,  # Add the argmax distribution
                        'entropy_softmax': mean_neuron_entropy
                    }
                    distribution[layer_group_name].append(layer_info)
        
        return distribution

    def print_trainable_params(self):
        total_params = 0
        trainable_params = 0
        logic_params = 0
        
        for name, param in self.named_parameters():
            params = param.numel()
            total_params += params
            
            # Count logic layer parameters separately
            param_attr_name = 'channel_weights_raw' if self.use_channel_logic else 'weights'
            if param_attr_name in name:
                if 'n_layers' in name or 'k_layers' in name or 'm_layers' in name:
                    logic_params += params
            
            if param.requires_grad:
                trainable_params += params
                print(f"{name}: {params:,} (trainable)")
            else:
                print(f"{name}: {params:,} (frozen)")
        
        layer_type = "ChannelLogic" if self.use_channel_logic else "DiffLogic"
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Frozen parameters: {total_params - trainable_params:,}")
        print(f"{layer_type} layer parameters: {logic_params:,}")
        
        return trainable_params

    def get_predictions(self, outputs):
        if isinstance(outputs, tuple):
            return outputs[0]
        return outputs

    def save_model(self, path):
        indices_list = []
        for layer in self.n_layers + self.k_layers + self.m_layers:
            if hasattr(layer, 'indices'):
                indices_list.append((
                    layer.indices[0].cpu().clone(),
                    layer.indices[1].cpu().clone()
                ))
            else:
                # For ChannelLogic layers, save gate connections instead
                if hasattr(layer, 'gate_connections'):
                    indices_list.append(layer.gate_connections.cpu().clone())
                else:
                    indices_list.append(None)
                    
        torch.save({
            'model_state': self.state_dict(),
            'indices': indices_list,
            'config': {
                'num_input_tokens': self.num_input_tokens,
                'vocab_size': self.embedding.num_embeddings,
                'num_classes': self.num_classes,
                'embedding_dim': self.embedding_dim,
                'seq_length': self.seq_length,
                'n_layers_sizes': [l.out_dim for l in self.n_layers],
                'k_layers_sizes': [l.out_dim for l in self.k_layers],
                'm_layers_sizes': [l.out_dim for l in self.m_layers[:-1]],  # Exclude the final layer
                'device': self.device,
                'grad_factor': self.grad_factor,
                'connections': self.connections,
                'group_sum_tau': self.group_sum_tau,  # Save the tau parameter for GroupSum
                'gumbel_tau': self.gumbel_tau,  # Save the gumbel_tau parameter
                'use_st_estimator': self.use_st_estimator,  # Save the use_st_estimator parameter
                'dropout_prob': self.dropout_prob,  # Save the dropout probability
                'padding_idx': self.padding_idx,
                'bos_token_id': self.bos_token_id,
                'eos_token_id': self.eos_token_id,
                'frozen_layers': self.frozen_layers,
                'noise_layers': self.noise_layers,
                'noise_std': self.noise_std,
                'use_channel_logic': self.use_channel_logic,
                'channel_logic_k': self.channel_logic_k,
                'channel_regularization_lambda': self.channel_regularization_lambda,
                'use_compiled_layers': self.use_compiled_layers,
                'gradient_rescaling': self.gradient_rescaling,
                'adaptive_gradient_clipping': self.adaptive_gradient_clipping
            }
        }, path)

    @classmethod
    def load_model(cls, path, device='cuda'):
        checkpoint = torch.load(path, map_location=device)
        cfg = checkpoint['config']
        
        # Extract all configuration parameters with defaults for backward compatibility
        group_sum_tau = cfg.get('group_sum_tau', 1.0)
        gumbel_tau = cfg.get('gumbel_tau', None)
        use_st_estimator = cfg.get('use_st_estimator', True)
        dropout_prob = cfg.get('dropout_prob', 0.2)
        padding_idx = cfg.get('padding_idx', 0)
        bos_token_id = cfg.get('bos_token_id', 2)
        eos_token_id = cfg.get('eos_token_id', 3)
        frozen_layers = cfg.get('frozen_layers', None)
        noise_layers = cfg.get('noise_layers', None)
        noise_std = cfg.get('noise_std', 0.01)
        use_channel_logic = cfg.get('use_channel_logic', False)
        channel_logic_k = cfg.get('channel_logic_k', 2)
        channel_regularization_lambda = cfg.get('channel_regularization_lambda', 0.0)
        connections = cfg.get('connections', 'random')
        use_compiled_layers = cfg.get('use_compiled_layers', True)
        gradient_rescaling = cfg.get('gradient_rescaling', None)
        adaptive_gradient_clipping = cfg.get('adaptive_gradient_clipping', False)
        
        # Use num_input_tokens if available, otherwise fall back to vocab_size
        num_input_tokens = cfg.get('num_input_tokens', cfg.get('vocab_size'))
        
        model = cls(
            num_input_tokens=num_input_tokens,
            embedding_dim=cfg['embedding_dim'],
            seq_length=cfg['seq_length'],
            n_layers_sizes=cfg['n_layers_sizes'],
            k_layers_sizes=cfg['k_layers_sizes'],
            m_layers_sizes=cfg['m_layers_sizes'],
            num_classes=cfg['num_classes'],
            device=device,
            grad_factor=cfg['grad_factor'],
            connections=connections,
            group_sum_tau=group_sum_tau,
            gumbel_tau=gumbel_tau,
            use_st_estimator=use_st_estimator,
            dropout_prob=dropout_prob,
            padding_idx=padding_idx,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            frozen_layers=frozen_layers,
            noise_layers=noise_layers,
            noise_std=noise_std,
            use_channel_logic=use_channel_logic,
            channel_logic_k=channel_logic_k,
            channel_regularization_lambda=channel_regularization_lambda,
            use_compiled_layers=use_compiled_layers,
            gradient_rescaling=gradient_rescaling,
            adaptive_gradient_clipping=adaptive_gradient_clipping
        )
        
        model.load_state_dict(checkpoint['model_state'])
        
        all_layers = model.n_layers + model.k_layers + model.m_layers
        for layer, saved_data in zip(all_layers, checkpoint['indices']):
            if saved_data is not None:
                if model.use_channel_logic:
                    # For ChannelLogic layers, restore gate connections
                    if hasattr(layer, 'gate_connections'):
                        layer.gate_connections.data.copy_(saved_data.to(device))
                else:
                    # For DiffLogic layers, restore indices
                    if hasattr(layer, 'indices'):
                        layer.indices = (
                            saved_data[0].to(device),
                            saved_data[1].to(device)
                        )
        
        return model

    def __del__(self):
        """Clean up gradient hooks when model is deleted"""
        for hook in self.gradient_hooks:
            try:
                hook.remove()
            except:
                pass
        for hook in self.rescaling_hooks:
            try:
                hook.remove()
            except:
                pass