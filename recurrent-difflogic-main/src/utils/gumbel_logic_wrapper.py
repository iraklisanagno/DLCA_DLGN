from types import MethodType
import torch
import torch.nn.functional as F
from difflogic import LogicLayer
from difflogic.difflogic import LogicLayerCudaFunction
from difflogic.packbitstensor import PackBitsTensor

def patch_logic_layers_with_gumbel(model, tau=1.0, use_st_estimator=True, verbose=False):
    """
    Monkey-patch every LogicLayer in a model to use Gumbel-Softmax with memory-efficient implementation.
    
    Args:
        model: The model containing LogicLayers to patch
        tau: Temperature parameter for Gumbel-Softmax
        use_st_estimator: Whether to use Straight-Through estimator
        verbose: Whether to print messages when forward methods are first called
    """
    
    # Helper function to apply Gumbel-Softmax to weights
    def _weights_to_ops(self, training):
        if training and use_st_estimator:
            # Gumbel-Softmax with hard sampling (ST estimator)
            gumbel_noise = -torch.log(-torch.log(torch.rand_like(self.weights)))
            gumbel_logits = (self.weights + gumbel_noise) / self.gumbel_tau
            soft = F.softmax(gumbel_logits, dim=-1)
            
            # Straight-through trick: use hard in forward, soft in backward
            index = soft.argmax(dim=-1, keepdim=True)
            hard = torch.zeros_like(soft).scatter_(-1, index, 1.0)
            weights = (hard - soft).detach() + soft
        elif training:
            # Regular softmax if not using ST estimator
            weights = F.softmax(self.weights / self.gumbel_tau, dim=-1)
        else:
            # Use discrete operations during evaluation
            weights = F.one_hot(self.weights.argmax(dim=-1), 16).float()
        
        return weights
    
    # Efficient implementation that computes only needed gates
    def _compute_selected_gates(self, a, b, weights):
        """
        Memory-efficient gate computation that avoids materializing all gates at once
        """
        batch_size = a.size(0)
        out_dim = weights.size(0)
        device = a.device
        
        # Check if we have one-hot/discrete weights (from argmax)
        is_one_hot = ((weights.sum(dim=-1) - 1.0).abs() < 1e-6).all()
        
        if is_one_hot:
            # Discrete case - only compute the selected gate for each neuron
            selected_gates = weights.argmax(dim=-1)  # [out_dim]
            output = torch.zeros(batch_size, out_dim, device=device)
            
            # For each gate type, compute only for neurons that use it
            for gate_idx in range(16):
                mask = (selected_gates == gate_idx)
                if mask.any():
                    output[:, mask] = self._compute_gate(a[:, mask], b[:, mask], gate_idx)
                    
            return output
        else:
            # Soft case - compute weighted sum of gates without materializing all at once
            output = torch.zeros(batch_size, out_dim, device=device)
            
            for gate_idx in range(16):
                # Only compute gate if any neuron has non-negligible weight for it
                if (weights[:, gate_idx] > 1e-6).any():
                    gate_weight = weights[:, gate_idx].unsqueeze(0)  # [1, out_dim]
                    gate_output = self._compute_gate(a, b, gate_idx)  # [batch, out_dim]
                    output += gate_output * gate_weight
                    
            return output
    
    # Single gate computation function
    def _compute_gate(self, a, b, gate_idx):
        """Compute a single logic gate output (gate_idx 0-15)"""
        if gate_idx == 0:  # FALSE
            return torch.zeros_like(a)
        elif gate_idx == 1:  # AND
            return a * b
        elif gate_idx == 2:  # A & ~B
            return a * (1 - b)
        elif gate_idx == 3:  # A
            return a
        elif gate_idx == 4:  # ~A & B
            return (1 - a) * b
        elif gate_idx == 5:  # B
            return b
        elif gate_idx == 6:  # XOR
            return a + b - 2 * a * b
        elif gate_idx == 7:  # OR
            return a + b - a * b
        elif gate_idx == 8:  # NOR
            return 1 - (a + b - a * b)
        elif gate_idx == 9:  # XNOR
            return 1 - (a + b - 2 * a * b)
        elif gate_idx == 10:  # ~B
            return 1 - b
        elif gate_idx == 11:  # A | ~B
            return a + (1 - b) - a * (1 - b)
        elif gate_idx == 12:  # ~A
            return 1 - a
        elif gate_idx == 13:  # ~A | B
            return (1 - a) + b - (1 - a) * b
        elif gate_idx == 14:  # NAND
            return 1 - a * b
        elif gate_idx == 15:  # TRUE
            return torch.ones_like(a)
    
    # Python implementation with Gumbel
    def forward_python_gumbel(self, x):
        if verbose and not getattr(self, "_print_done", False):
            print(f"[GUMBEL] LogicLayer(id={id(self)}) using forward_python_gumbel")
            self._print_done = True
        
        assert x.shape[-1] == self.in_dim
        if self.indices[0].dtype != torch.long:
            self.indices = self.indices[0].long(), self.indices[1].long()
            
        # Get input pairs
        a = x[:, self.indices[0]]  # [batch, out_dim]
        b = x[:, self.indices[1]]  # [batch, out_dim]
        
        # Get gate weights with Gumbel-Softmax
        weights = _weights_to_ops(self, self.training)
        
        # Compute only needed gates to save memory
        return _compute_selected_gates(self, a, b, weights)
    
    # CUDA implementation with Gumbel
    def forward_cuda_gumbel(self, x):
        if verbose and not getattr(self, "_print_done", False):
            print(f"[GUMBEL] LogicLayer(id={id(self)}) using forward_cuda_gumbel")
            self._print_done = True
        
        assert x.ndim == 2
        assert x.device.type == "cuda", x.device
        x = x.transpose(0, 1).contiguous()
        
        a, b = self.indices
        w = _weights_to_ops(self, self.training).to(x.dtype)
        
        return LogicLayerCudaFunction.apply(
            x, a, b, w,
            self.given_x_indices_of_y_start,
            self.given_x_indices_of_y
        ).transpose(0, 1)
    
    # Master forward method
    def forward_gumbel(self, x):
        if verbose and not getattr(self, "_print_done", False):
            print(f"[GUMBEL] LogicLayer(id={id(self)}) master forward")
            
        if self.implementation == "cuda":
            if isinstance(x, PackBitsTensor):
                return self.forward_cuda_eval(x)
            return self.forward_cuda(x)
        elif self.implementation == "python":
            return self.forward_python(x)
        else:
            raise ValueError(f"Unknown implementation: {self.implementation}")
    
    # Apply the patching to all LogicLayers in the model
    for layer in model.modules():
        if isinstance(layer, LogicLayer):
            # Add gumbel_tau parameter
            layer.gumbel_tau = tau
            
            # Add helper methods
            layer._compute_gate = MethodType(_compute_gate, layer)
            layer._compute_selected_gates = MethodType(_compute_selected_gates, layer)
            
            # Replace forward methods
            layer.forward_python = MethodType(forward_python_gumbel, layer)
            layer.forward_cuda = MethodType(forward_cuda_gumbel, layer)
            layer.forward = MethodType(forward_gumbel, layer)
            
            if verbose:
                print(f"[GUMBEL] Patched LogicLayer(id={id(layer)})")