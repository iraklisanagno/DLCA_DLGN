import torch
import torch.nn as nn
import torch.nn.functional as F
from difflogic import LogicLayer, GroupSum, PackBitsTensor
from models.base_model import BaseModel
from utils.gumbel_logic_wrapper import patch_logic_layers_with_gumbel


class SyncedFeedForwardDiffLogicModel(BaseModel):
    def __init__(
        self, num_input_tokens, embedding_dim, seq_length,
        logic_layer_sizes, num_classes,
        group_factor=1, device="cuda", grad_factor=1.0,
        connections='random', predefined_indices=None,
        difflogic_init_type='noisy_residual',  # 'gaussian', 'noisy_residual' or 'residual'
        noise_factor=1.0,         # New parameter for Gumbel noise scaling
        seed=None,  # Seed parameter
        group_sum_tau=1.0,  # Tau parameter for softmax in GroupSum
        gumbel_tau=None,  # Parameter for Gumbel-Softmax temperature
        use_st_estimator=True,  # Whether to use straight-through estimator with Gumbel-Softmax
        dropout_prob=0.2  # Dropout probability
    ):
        super().__init__()

        # Set random seed for reproducibility
        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        self.device = device
        self.seq_length = seq_length
        self.embedding_dim = embedding_dim
        self.logic_input_dim = embedding_dim  
        self.num_classes = num_classes
        self.group_factor = group_factor
        self.difflogic_init_type = difflogic_init_type
        self.noise_factor = noise_factor  # Store the noise factor
        self.emb_mode = False  # Default to continuous mode
        self.group_sum_tau = group_sum_tau  # Store the tau parameter for softmax
        self.gumbel_tau = gumbel_tau  # Store the gumbel tau parameter
        self.use_st_estimator = use_st_estimator  # Store the ST estimator flag
        self.grad_factor = grad_factor
        self.dropout_prob = dropout_prob  # Store dropout probability

        # Calculate final layer size
        final_layer_size = self.num_classes * group_factor
        self.logic_layer_sizes = logic_layer_sizes.copy()
        self.logic_layer_sizes[-1] = final_layer_size

        # Embedding layer
        self.embedding = nn.Embedding(num_input_tokens, embedding_dim, padding_idx=0)

        # Dropout layers
        self.emb_dropout = nn.Dropout(dropout_prob)  # Dropout after embedding
        self.logic_dropouts = nn.ModuleList([nn.Dropout(dropout_prob) for _ in self.logic_layer_sizes])  # Dropouts after logic layers

        # Configure connections list
        if isinstance(connections, list):
            assert len(connections) == len(self.logic_layer_sizes), \
                "Connections list must match logic_layer_sizes length"
        else:
            connections = [connections] * len(self.logic_layer_sizes)

        # Build and initialize logic layers
        self.logic_layers = nn.ModuleList()
        prev_dim = self.logic_input_dim
        for i, layer_size in enumerate(self.logic_layer_sizes):
            layer = LogicLayer(
                in_dim=prev_dim,
                out_dim=layer_size,
                device=device,
                grad_factor=grad_factor,
                implementation='cuda',
                connections=connections[i]
            )
            # Apply difflogic initialization
            self._init_logic_layer(layer)

            # Override indices if provided
            if predefined_indices and predefined_indices[i] is not None:
                a, b = predefined_indices[i]
                layer.indices = (
                    a.to(device) if isinstance(a, torch.Tensor) else torch.tensor(a, device=device, dtype=torch.int64),
                    b.to(device) if isinstance(b, torch.Tensor) else torch.tensor(b, device=device, dtype=torch.int64)
                )

            self.logic_layers.append(layer)
            prev_dim = layer_size

        # Final grouping layer
        self.group_sum = GroupSum(k=self.num_classes, tau=group_sum_tau, device=device)
        
        # Apply Gumbel-Softmax patching if gumbel_tau is provided
        if gumbel_tau is not None:
            patch_logic_layers_with_gumbel(
                self, tau=gumbel_tau, use_st_estimator=use_st_estimator, verbose=False
            )
            
        self.set_mode('train')

    def _init_logic_layer(self, layer):
        """Initialize logic layer weights based on initialization type"""
        with torch.no_grad():
            if self.difflogic_init_type == 'residual':
                layer.weights.data.zero_()
                if layer.weights.size(1) > 3:
                    layer.weights.data[:, 3].fill_(5.0)
            elif self.difflogic_init_type == 'noisy_residual':
                # Initialize all weights with Gaussian noise (mean=0, std=noise_factor)
                layer.weights.data.normal_(mean=0.0, std=self.noise_factor)
                # Set the mean of the third weight to 5.0
                if layer.weights.size(1) > 3:
                    layer.weights.data[:, 3].add_(5.0)  
            elif self.difflogic_init_type == 'gaussian':
                # Initialize all weights with Gaussian noise
                layer.weights.data.normal_(mean=0.0, std=self.noise_factor)
    
    def forward(self, x):
        # Input shape: (batch_size, seq_len)
        batch_size, seq_len = x.size()
        
        # Embed and normalize based on mode
        embedded = (self.embedding(x) > 0).float() if self.emb_mode else torch.sigmoid(self.embedding(x))
        
        # Apply dropout to embedding output
        embedded = self.emb_dropout(embedded)
        
        # Flatten tokens
        x_flat = embedded.view(-1, self.embedding_dim)  # (batch_size * seq_len, embedding_dim)
        
        # Process through logic layers with dropout
        for i, layer in enumerate(self.logic_layers):
            x_flat = layer(x_flat)
            x_flat = self.logic_dropouts[i](x_flat)  # Apply dropout after each logic layer
        
        # Apply group sum
        group_out = self.group_sum(x_flat)  # (batch_size * seq_len, num_classes)
        
        # Reshape back to batch dimensions
        out = group_out.view(batch_size, seq_len, -1)
        
        # Binary regularization
        binary_reg_loss = torch.mean(embedded * (1 - embedded))
        return out, binary_reg_loss

    def set_mode(self, mode):
        """
        Set the model mode for training or different evaluation scenarios
        
        Args:
            mode (str): One of 'train', 'eval', 'eval_col_emb', 'eval_col_layer', 'eval_col_all'
        """
        assert mode in ['train', 'eval', 'eval_col_emb', 'eval_col_layer', 'eval_col_all'], \
            "Mode must be one of 'train', 'eval', 'eval_col_emb', 'eval_col_layer', 'eval_col_all'"
        
        # Set training/evaluation mode
        self.train() if mode in ['train', 'eval', 'eval_col_emb'] else self.eval()
        
        # Set embedding mode (continuous or binarized)
        self.emb_mode = mode in ['eval_col_emb', 'eval_col_all']
        
    def analyze_logic_function_distribution(self):
        """
        Analyzes the distribution of logic functions used in each layer of the model.
        Calculates per-neuron entropy of softmax probabilities and averages across the layer.
        
        Returns:
            dict: A dictionary containing distribution information for all layers
        """
        import torch.nn.functional as F
        
        # Define names for the 16 binary logic functions
        function_names = [
            "FALSE", "AND", "A & ~B", "A", "~A & B", "B", "XOR", "OR",
            "NOR", "XNOR", "~B", "A | ~B", "~A", "~A | B", "NAND", "TRUE"
        ]
        
        # Dictionary to store the results
        distribution = {
            'logic_layers': []
        }
        
        # Analyze each layer
        for i, layer in enumerate(self.logic_layers):
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
                'layer_name': f"logic_layer_{i}",
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
            distribution['logic_layers'].append(layer_info)
        
        return distribution

    def print_trainable_params(self):
        """Prints the number of trainable parameters in the model"""
        total_params = 0
        for name, param in self.named_parameters():
            if param.requires_grad:
                params = param.numel()
                total_params += params
                print(f"{name}: {params:,}")

        print(f"Total trainable parameters: {total_params:,}")
        return total_params

    def save_model(self, path):
        """Save model with connection indices and configuration"""
        indices_list = []
        for layer in self.logic_layers:
            indices_list.append((
                layer.indices[0].cpu().clone(),
                layer.indices[1].cpu().clone()
            ))

        torch.save({
            'model_state': self.state_dict(),
            'indices': indices_list,
            'config': {
                'vocab_size': self.embedding.num_embeddings,
                'target_vocab_size': self.num_classes,
                'embedding_dim': self.embedding_dim,
                'seq_length': self.seq_length,
                'logic_sizes': [layer.out_dim for layer in self.logic_layers],
                'device': self.device,
                'grad_factor': self.grad_factor,
                'difflogic_init_type': self.difflogic_init_type,
                'noise_factor': self.noise_factor,
                'group_sum_tau': self.group_sum_tau,  # Save the tau parameter for GroupSum
                'gumbel_tau': self.gumbel_tau,  # Save the gumbel_tau parameter
                'use_st_estimator': self.use_st_estimator,  # Save the use_st_estimator parameter
                'dropout_prob': self.dropout_prob  # Save the dropout probability
            }
        }, path)

    @classmethod
    def load_model(cls, path, device='cuda'):
        """Load complete model with indices and configuration"""
        checkpoint = torch.load(path, map_location=device)
        cfg = checkpoint['config']
        cfg['device'] = device
        
        # Handle backward compatibility for new parameters
        noise_factor = cfg.get('noise_factor', 1.0)
        group_sum_tau = cfg.get('group_sum_tau', 1.0)
        gumbel_tau = cfg.get('gumbel_tau', None)
        use_st_estimator = cfg.get('use_st_estimator', True)
        dropout_prob = cfg.get('dropout_prob', 0.2)
        
        model = cls(
            num_input_tokens=cfg['vocab_size'],
            embedding_dim=cfg['embedding_dim'],
            seq_length=cfg['seq_length'],
            logic_layer_sizes=cfg['logic_sizes'],
            num_classes=cfg['target_vocab_size'],
            device=device,
            grad_factor=cfg['grad_factor'],
            difflogic_init_type=cfg.get('difflogic_init_type', 'gaussian'),
            noise_factor=noise_factor,
            group_sum_tau=group_sum_tau,
            gumbel_tau=gumbel_tau,
            use_st_estimator=use_st_estimator,
            dropout_prob=dropout_prob
        )
        
        model.load_state_dict(checkpoint['model_state'])
        
        # Restore indices
        for layer, (a_idx, b_idx) in zip(model.logic_layers, checkpoint['indices']):
            layer.indices = (
                a_idx.to(device),
                b_idx.to(device)
            )
        
        return model