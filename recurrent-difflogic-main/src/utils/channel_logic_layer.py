import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def get_cuda_capability():
    """Get CUDA compute capability of the current device."""
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        major, minor = torch.cuda.get_device_capability(device)
        return major + minor / 10
    return 0.0


class ChannelLogicLayer(torch.nn.Module):
    """
    Ultra-fast channel-based differentiable logic gate layer optimized for k=2.
    """
    
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        k: int = 2,
        device: str = 'cuda',
        grad_factor: float = 1.0,
        connections: str = 'random',
        regularization_lambda: float = 0.0,
        dtype: torch.dtype = torch.float32
    ):
        super().__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.k = k
        self.device = device
        self.grad_factor = grad_factor
        self.connections = connections
        self.regularization_lambda = regularization_lambda
        self.dtype = dtype
        
        # For k=2, we have exactly 4 combinations: [0,0], [1,0], [0,1], [1,1]
        self.num_combinations = 4
        
        # Initialize channel weights (4 weights per neuron instead of 16)
        init_scale = np.sqrt(2.0 / 4)
        self.channel_weights_raw = nn.Parameter(
            torch.randn(out_dim, 4, device=device, dtype=dtype) * init_scale
        )
        
        # Generate connections efficiently for k=2
        self.register_buffer('gate_connections', self._generate_connections_k2())
        
    def _generate_connections_k2(self) -> torch.Tensor:
        """Ultra-fast connection generation optimized for k=2."""
        if self.connections == 'random':
            if self.in_dim >= 2:
                # For k=2, each gate needs 2 inputs
                # Generate all connections at once using vectorized operations
                indices = torch.randint(0, self.in_dim, (self.out_dim, 2), device='cpu')
                return indices.to(self.device).long()
            else:
                # If in_dim < 2, sample with replacement
                indices = torch.randint(0, self.in_dim, (self.out_dim, 2), device='cpu')
                return indices.to(self.device).long()
        elif self.connections == 'unique':
            # Cyclic pattern for unique connections
            all_indices = torch.arange(self.out_dim * 2, device='cpu') % self.in_dim
            connections = all_indices.reshape(self.out_dim, 2)
            return connections.to(self.device).long()
        else:
            raise ValueError(f"Unknown connection type: {self.connections}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Ultra-fast forward pass specialized for k=2 binary gates.
        """
        batch_size = x.shape[0]
        
        # Apply gradient scaling if needed
        if self.grad_factor != 1.0:
            x = GradFactor.apply(x, self.grad_factor)
        
        # Ensure correct dtype
        x = x.to(self.dtype)
        
        # Get channel weights: w = sigma(xi) - computed once
        channel_weights = torch.sigmoid(self.channel_weights_raw)  # (out_dim, 4)
        
        # SPEED OPTIMIZATION: Direct indexing for k=2
        # gate_connections: (out_dim, 2)
        x0 = x[:, self.gate_connections[:, 0]]  # (batch_size, out_dim)
        x1 = x[:, self.gate_connections[:, 1]]  # (batch_size, out_dim)
        
        # ULTRA-FAST: Compute all 4 combination probabilities directly
        # For k=2: [0,0], [1,0], [0,1], [1,1]
        p00 = (1 - x0) * (1 - x1)  # P(x0=0, x1=0)
        p10 = x0 * (1 - x1)       # P(x0=1, x1=0)  
        p01 = (1 - x0) * x1       # P(x0=0, x1=1)
        p11 = x0 * x1             # P(x0=1, x1=1)
        
        # SPEED OPTIMIZATION: Use direct multiplication instead of matrix ops
        # channel_weights[:, 0] corresponds to [0,0]
        # channel_weights[:, 1] corresponds to [1,0]
        # channel_weights[:, 2] corresponds to [0,1]
        # channel_weights[:, 3] corresponds to [1,1]
        outputs = (p00 * channel_weights[:, 0].unsqueeze(0) +
                  p10 * channel_weights[:, 1].unsqueeze(0) +
                  p01 * channel_weights[:, 2].unsqueeze(0) +
                  p11 * channel_weights[:, 3].unsqueeze(0))
        
        return outputs  # (batch_size, out_dim)
    
    def compute_regularization_loss(self) -> torch.Tensor:
        """Fast regularization loss computation."""
        if self.regularization_lambda == 0.0:
            return torch.tensor(0.0, device=self.device, dtype=self.dtype)
            
        channel_weights = torch.sigmoid(self.channel_weights_raw)
        # Efficient squared difference computation
        diff_squared = (channel_weights - 0.5).pow(2)
        reg_loss = torch.sum(diff_squared)
        return self.regularization_lambda * reg_loss
    
    def extra_repr(self) -> str:
        return f"in_dim={self.in_dim}, out_dim={self.out_dim}, k={self.k}, " \
               f"connections={self.connections}, reg_lambda={self.regularization_lambda}"


class GradFactor(torch.autograd.Function):
    """Optimized gradient scaling function."""
    
    @staticmethod
    def forward(ctx, input, factor):
        ctx.factor = factor
        return input
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * ctx.factor, None