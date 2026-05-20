import math
import warnings
from typing import List, Dict, Any, Optional, Union, Callable
import torch


class ScheduleFunction:
    """Base class for learning rate schedule functions"""
    
    def __call__(self, step: int) -> float:
        """Calculate the schedule value for a given step"""
        raise NotImplementedError
    
    @classmethod
    def create(cls, config: Dict[str, Any]) -> 'ScheduleFunction':
        """Factory method to create a schedule function from config"""
        schedule_type = config.get('type', 'constant')
        
        if schedule_type == 'linear_ramp':
            return LinearRampFunction(
                config.get('start_value', 0.0),
                config.get('end_value', 1.0),
                config.get('start_step', 0),
                config.get('end_step', 1000)
            )
        elif schedule_type == 'cosine':
            return CosineFunction(
                config.get('start_value', 0.0),
                config.get('end_value', 1.0),
                config.get('start_step', 0),
                config.get('end_step', 1000)
            )
        elif schedule_type == 'step':
            return StepFunction(
                config.get('initial_value', 1.0),
                config.get('factor', 0.5),
                config.get('step_sizes', [500, 1000, 1500])
            )
        elif schedule_type == 'warmup_cooldown':
            return WarmupCooldownFunction(
                config.get('warmup_start_value', 0.0),
                config.get('peak_value', 1.0),
                config.get('end_value', 0.0),
                config.get('warmup_steps', 100),
                config.get('total_steps', 1000),
                config.get('warmup_method', 'linear')
            )
        elif schedule_type == 'transformer_warmup':
            return TransformerWarmupFunction(
                config.get('d_model', 512),
                config.get('warmup_steps', 4000),
                config.get('base_lr', 1e-4)
            )
        elif schedule_type == 'exponential':
            return ExponentialFunction(
                config.get('initial_value', 1.0),
                config.get('decay_rate', 0.95),
                config.get('decay_steps', 1000)
            )
        elif schedule_type == 'polynomial':
            return PolynomialFunction(
                config.get('initial_value', 1.0),
                config.get('end_value', 0.01),
                config.get('power', 1.0),
                config.get('total_steps', 10000)
            )
        elif schedule_type == 'adaptive':
            return AdaptiveFunction(
                config.get('initial_value', 1.0),
                config.get('update_frequency', 100),
                config.get('target_gradient_ratio', 0.1),
                config.get('adjustment_factor', 1.2),
                config.get('min_multiplier', 0.1),
                config.get('max_multiplier', 10.0),
                config.get('smoothing_factor', 0.9),
                config.get('warmup_steps', 1000),
                config.get('reference_layers', ['m_layers', 'p_layers'])
            )
        else:  # Default to constant
            return ConstantFunction(config.get('value', 1.0))


class LinearRampFunction(ScheduleFunction):
    """Linear ramp schedule from start_value to end_value over steps"""
    
    def __init__(self, start_value: float, end_value: float, 
                 start_step: int, end_step: int):
        self.start_value = start_value
        self.end_value = end_value
        self.start_step = start_step
        self.end_step = end_step
        self.total_steps = max(1, end_step - start_step)  # Avoid division by zero
    
    def __call__(self, step: int) -> float:
        """Compute linear ramp value based on current step"""
        if step < self.start_step:
            return self.start_value
        elif step >= self.end_step:
            return self.end_value
        else:
            # Linear interpolation
            progress = (step - self.start_step) / self.total_steps
            return self.start_value + progress * (self.end_value - self.start_value)


class CosineFunction(ScheduleFunction):
    """Cosine annealing schedule from start_value to end_value over steps"""
    
    def __init__(self, start_value: float, end_value: float, 
                 start_step: int, end_step: int):
        self.start_value = start_value
        self.end_value = end_value
        self.start_step = start_step
        self.end_step = end_step
        self.total_steps = max(1, end_step - start_step)  # Avoid division by zero
    
    def __call__(self, step: int) -> float:
        """Compute cosine annealing value based on current step"""
        if step < self.start_step:
            return self.start_value
        elif step >= self.end_step:
            return self.end_value
        else:
            # Cosine schedule
            progress = (step - self.start_step) / self.total_steps
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            return self.end_value + (self.start_value - self.end_value) * cosine_decay


class StepFunction(ScheduleFunction):
    """Step schedule that reduces by a factor at specified steps"""
    
    def __init__(self, initial_value: float, factor: float, 
                 step_sizes: List[int]):
        self.initial_value = initial_value
        self.factor = factor
        self.step_sizes = sorted(step_sizes)  # Ensure steps are in order
    
    def __call__(self, step: int) -> float:
        """Compute step schedule value based on current step"""
        value = self.initial_value
        for milestone in self.step_sizes:
            if step >= milestone:
                value *= self.factor
            else:
                break
        return value


class ExponentialFunction(ScheduleFunction):
    """Exponential decay schedule"""
    
    def __init__(self, initial_value: float, decay_rate: float, decay_steps: int):
        self.initial_value = initial_value
        self.decay_rate = decay_rate
        self.decay_steps = max(1, decay_steps)
    
    def __call__(self, step: int) -> float:
        """Compute exponential decay value based on current step"""
        decay_factor = self.decay_rate ** (step / self.decay_steps)
        return self.initial_value * decay_factor


class PolynomialFunction(ScheduleFunction):
    """Polynomial decay schedule"""
    
    def __init__(self, initial_value: float, end_value: float, power: float, total_steps: int):
        self.initial_value = initial_value
        self.end_value = end_value
        self.power = power
        self.total_steps = max(1, total_steps)
    
    def __call__(self, step: int) -> float:
        """Compute polynomial decay value based on current step"""
        if step >= self.total_steps:
            return self.end_value
        
        decay_factor = (1 - step / self.total_steps) ** self.power
        return self.end_value + (self.initial_value - self.end_value) * decay_factor


class ConstantFunction(ScheduleFunction):
    """Constant value schedule"""
    
    def __init__(self, value: float):
        self.value = value
    
    def __call__(self, step: int) -> float:
        """Return constant value regardless of step"""
        return self.value


class AdaptiveFunction(ScheduleFunction):
    """
    Adaptive schedule function that adjusts based on gradient statistics.
    This is a stateful function that needs to be tied to a specific layer group.
    FIXED VERSION: Reduced history requirement and enhanced debugging.
    """
    
    def __init__(self, initial_value: float, update_frequency: int = 100,
                 target_gradient_ratio: float = 0.1, adjustment_factor: float = 1.2,
                 min_multiplier: float = 0.1, max_multiplier: float = 10.0,
                 smoothing_factor: float = 0.9, warmup_steps: int = 1000,
                 reference_layers: List[str] = None):
        self.initial_value = initial_value
        self.current_value = initial_value
        self.update_frequency = update_frequency
        self.target_gradient_ratio = target_gradient_ratio
        self.adjustment_factor = adjustment_factor
        self.min_multiplier = min_multiplier
        self.max_multiplier = max_multiplier
        self.smoothing_factor = smoothing_factor
        self.warmup_steps = warmup_steps
        self.reference_layers = reference_layers or ['m_layers', 'p_layers']
        
        # State tracking
        self.last_update_step = 0
        self.gradient_history = {
            'mean_grad_norm': 0.0,
            'step_count': 0
        }
        
        # Will be set by the LayerLRScheduler
        self.layer_name = None
        self.scheduler_ref = None
    
    def __call__(self, step: int) -> float:
        """Return current adaptive value"""
        return self.current_value
    
    def update_adaptive_value(self, step: int, current_grad_norm: float, reference_grad_norm: float):
        """Update the adaptive value based on gradient statistics - FIXED VERSION"""
        # Check if we should update
        if (step - self.last_update_step < self.update_frequency or 
            step < self.warmup_steps):
            return
        
        # Update gradient history
        if self.gradient_history['step_count'] == 0:
            self.gradient_history['mean_grad_norm'] = current_grad_norm
        else:
            self.gradient_history['mean_grad_norm'] = (
                self.smoothing_factor * self.gradient_history['mean_grad_norm'] + 
                (1 - self.smoothing_factor) * current_grad_norm
            )
        
        self.gradient_history['step_count'] += 1
        self.last_update_step = step
        
        # 🔧 CRITICAL FIX: Reduced history requirement from 10 to 3
        # Calculate adjustment if we have enough history and valid reference
        if (self.gradient_history['step_count'] >= 3 and  # Changed from 10 to 3
            reference_grad_norm > 0 and current_grad_norm > 0):
            
            current_ratio = current_grad_norm / reference_grad_norm
            
            # 🔧 Enhanced debugging for first few updates
            if hasattr(self, 'layer_name') and self.layer_name and self.gradient_history['step_count'] <= 20:
                print(f"🔄 Step {step}: {self.layer_name} ratio={current_ratio:.4f}, target={self.target_gradient_ratio:.4f}, current_value={self.current_value:.3f}")
            
            # Determine if adjustment is needed
            if current_ratio < self.target_gradient_ratio * 0.5:  # Very small gradients
                new_value = min(self.current_value * self.adjustment_factor, self.max_multiplier)
                
                if abs(new_value - self.current_value) > 0.01:  # Meaningful change
                    old_value = self.current_value
                    self.current_value = new_value
                    if hasattr(self, 'layer_name') and self.layer_name:
                        print(f"🔼 Step {step}: Increasing LR for {self.layer_name}: "
                              f"{old_value:.2f} → {self.current_value:.2f} "
                              f"(grad ratio: {current_ratio:.4f} < target: {self.target_gradient_ratio:.4f})")
                        
            elif current_ratio > self.target_gradient_ratio * 2.0:  # Very large gradients
                new_value = max(self.current_value / self.adjustment_factor, self.min_multiplier)
                
                if abs(new_value - self.current_value) > 0.01:  # Meaningful change
                    old_value = self.current_value
                    self.current_value = new_value
                    if hasattr(self, 'layer_name') and self.layer_name:
                        print(f"🔽 Step {step}: Decreasing LR for {self.layer_name}: "
                              f"{old_value:.2f} → {self.current_value:.2f} "
                              f"(grad ratio: {current_ratio:.4f} > target: {self.target_gradient_ratio:.4f})")
            else:
                # 🔧 Debug message for stable ratios (only for first few updates)
                if hasattr(self, 'layer_name') and self.layer_name and self.gradient_history['step_count'] <= 10:
                    print(f"✅ Step {step}: {self.layer_name} stable (ratio: {current_ratio:.4f} in target range)")


class WarmupCooldownFunction(ScheduleFunction):
    """Warmup followed by exponential decay cooldown schedule"""
    
    def __init__(self, warmup_start_value: float, peak_value: float, end_value: float,
                 warmup_steps: int, total_steps: int, warmup_method: str = 'linear'):
        self.warmup_start_value = warmup_start_value
        self.peak_value = peak_value
        self.end_value = end_value
        self.warmup_steps = max(1, warmup_steps)  # Avoid division by zero
        self.total_steps = max(warmup_steps + 1, total_steps)  # Ensure valid total steps
        self.warmup_method = warmup_method
        
        self.cooldown_steps = self.total_steps - self.warmup_steps
        if self.cooldown_steps > 0 and self.peak_value > 0 and self.end_value > 0:
            self.decay_rate = math.log(self.peak_value / self.end_value) / self.cooldown_steps
        else:
            self.decay_rate = 0

    def __call__(self, step: int) -> float:
        """Compute warmup followed by exponential decay cooldown"""
        if step < self.warmup_steps:
            # Warmup phase
            alpha = step / self.warmup_steps
            if self.warmup_method == 'linear':
                return self.warmup_start_value + alpha * (self.peak_value - self.warmup_start_value)
            else:  # Cosine warmup
                return self.warmup_start_value + 0.5 * (self.peak_value - self.warmup_start_value) * \
                       (1 - math.cos(alpha * math.pi))
        else:
            # Cooldown phase (exponential decay)
            if self.cooldown_steps <= 0:
                return self.end_value
            t = step - self.warmup_steps
            return self.peak_value * math.exp(-self.decay_rate * t)


class TransformerWarmupFunction(ScheduleFunction):
    """
    Implements the learning rate schedule from "Attention is All You Need" paper.
    Returns a MULTIPLIER that should be applied to a base learning rate.
    """
    
    def __init__(self, d_model: int, warmup_steps: int = 4000, base_lr: float = 1e-4):
        self.d_model = d_model
        self.warmup_steps = max(1, warmup_steps)
        self.base_lr = base_lr
        self.scale = d_model ** -0.5
        
    def __call__(self, step: int) -> float:
        """Compute transformer warmup schedule multiplier based on current step"""
        step = max(1, step)
        
        arg1 = step ** -0.5
        arg2 = step * (self.warmup_steps ** -1.5)
        
        # Calculate the absolute LR as per the paper
        absolute_lr = self.scale * min(arg1, arg2)
        
        # Return as a multiplier relative to base_lr
        return absolute_lr / self.base_lr


class StepBasedScheduler:
    """Base class for step-based schedulers that can be used with PyTorch optimizers"""
    
    def __init__(self, optimizer=None, last_step=-1, verbose=False):
        self.last_step = last_step
        self.verbose = verbose
        
        if optimizer is not None:
            self.optimizer = optimizer
            self.base_lrs = [group['lr'] for group in optimizer.param_groups]
            # Initialize with first step
            self.step()
    
    def get_lr(self) -> List[float]:
        """Get learning rates for all parameter groups"""
        raise NotImplementedError
    
    def step(self, step: Optional[int] = None) -> List[float]:
        """Update optimizer learning rates based on step"""
        if not hasattr(self, 'optimizer'):
            return []
            
        if step is None:
            self.last_step += 1
        else:
            self.last_step = step
        
        # Get new learning rates
        lrs = self.get_lr()
        
        # Update optimizer param groups
        for param_group, lr in zip(self.optimizer.param_groups, lrs):
            param_group['lr'] = lr
        
        if self.verbose:
            print(f"Step {self.last_step}: Learning rate updated to {lrs}")
        
        return lrs
    
    def get_last_lr(self) -> List[float]:
        """Return last computed learning rate by current scheduler"""
        return self.get_lr()


class FunctionBasedScheduler(StepBasedScheduler):
    """
    Generic step-based scheduler that uses a ScheduleFunction to determine values.
    Can be used for learning rates, weight factors, etc.
    """
    
    def __init__(self, schedule_function: ScheduleFunction, optimizer=None, last_step=-1, verbose=False):
        self.schedule_function = schedule_function
        super().__init__(optimizer, last_step, verbose)
    
    def get_lr(self) -> List[float]:
        """Get learning rate based on schedule function"""
        current_step = max(1, self.last_step + 1)  # Convert to 1-based step number
        factor = self.schedule_function(current_step)
        
        if not hasattr(self, 'base_lrs'):
            return [factor]
            
        return [base_lr * factor for base_lr in self.base_lrs]
    
    def get_value(self) -> float:
        """Get the current scheduled value (for non-LR applications)"""
        current_step = max(1, self.last_step + 1)  # Convert to 1-based step number
        return self.schedule_function(current_step)


class WarmupCooldownScheduler(FunctionBasedScheduler):
    """
    Step-based learning rate scheduler with warmup and cooldown phases.
    
    Args:
        optimizer: PyTorch optimizer
        warmup_start_value: Initial learning rate multiplier at the start of warmup
        peak_value: Maximum learning rate multiplier after warmup
        end_value: Final learning rate multiplier after cooldown
        warmup_steps: Number of warmup steps
        total_steps: Total number of steps (warmup + cooldown)
        warmup_method: Warmup method ('linear' or 'cosine')
        last_step: The index of the last step. Default: -1
        verbose: If True, prints a message to stdout for each update. Default: False
    """
    
    def __init__(self, optimizer, warmup_start_value=0.0, peak_value=1.0, 
                 end_value=0.0, warmup_steps=1000, total_steps=10000, 
                 warmup_method="linear", last_step=-1, verbose=False):
        
        schedule_function = WarmupCooldownFunction(
            warmup_start_value, peak_value, end_value,
            warmup_steps, total_steps, warmup_method
        )
        super().__init__(schedule_function, optimizer, last_step, verbose)


class TransformerWarmupScheduler(FunctionBasedScheduler):
    """
    Implements the learning rate schedule from "Attention is All You Need".
    Returns a multiplier that gets applied to the base learning rate.
    All operations are step-based.
    """
    
    def __init__(self, optimizer, d_model, warmup_steps=4000, last_step=-1, verbose=False):
        # Get the base learning rate from the optimizer
        base_lr = optimizer.param_groups[0]['lr']
        
        # Create the transformer warmup function with the correct base_lr
        schedule_function = TransformerWarmupFunction(d_model, warmup_steps, base_lr)
        super().__init__(schedule_function, optimizer, last_step, verbose)

class StepBasedReduceLROnPlateau(StepBasedScheduler):
    """
    ReduceLROnPlateau that works based on steps rather than epochs.
    Reduces learning rate when a metric has stopped improving.
    FIXED VERSION: Tracks base LR independently and prevents premature LR reduction.
    """
    
    def __init__(self, optimizer, mode='min', factor=0.1, patience=10,
                 threshold=1e-4, threshold_mode='rel', cooldown=0,
                 min_lr=0, eps=1e-8, verbose=False):
        self.mode = mode
        self.factor = factor
        self.patience = patience
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.cooldown = cooldown
        self.cooldown_counter = 0
        self.mode_worse = float('inf') if mode == 'min' else -float('inf')
        
        # ✅ FIX: Initialize best to None instead of mode_worse to prevent immediate reduction
        self.best = None  # This prevents the scheduler from triggering on first metric
        self.num_bad_steps = 0
        self.eps = eps
        
        # ✅ NEW: Track our own base learning rates independently
        self.current_base_lrs = [group['lr'] for group in optimizer.param_groups]
        
        # Initialize min_lr
        if isinstance(min_lr, list) or isinstance(min_lr, tuple):
            if len(min_lr) != len(optimizer.param_groups):
                raise ValueError(f"Expected {len(optimizer.param_groups)} min_lrs, got {len(min_lr)}")
            self.min_lrs = list(min_lr)
        else:
            self.min_lr = min_lr
            self.min_lrs = [min_lr] * len(optimizer.param_groups)
        
        # Initialize parent class manually to avoid calling step()
        self.last_step = -1
        self.verbose = verbose
        self.optimizer = optimizer
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        
        # Don't call super().__init__() to avoid automatic step() call
    
    def get_lr(self) -> List[float]:
        """Get current learning rates from optimizer (modified by layer scheduler)"""
        return [group['lr'] for group in self.optimizer.param_groups]
    
    def get_last_lr(self) -> List[float]:
        """Return our tracked base learning rates, not the modified ones"""
        return self.current_base_lrs.copy()
    
    def step(self, metrics: float = None, step: Optional[int] = None) -> None:
        """Update learning rates based on metrics"""
        # Handle case where step() is called without metrics (from parent init)
        if metrics is None:
            return
            
        if step is None:
            self.last_step += 1
        else:
            self.last_step = step
        
        current = metrics
        
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
        
        # ✅ FIX: Handle first metric properly
        if self.best is None:
            self.best = current
            self.num_bad_steps = 0
            if self.verbose:
                print(f'Step {self.last_step}: Initialized best metric to {current:.4f}')
            return  # Don't evaluate for reduction on first metric
        
        if self._is_better(current, self.best):
            self.best = current
            self.num_bad_steps = 0
            if self.verbose:
                print(f'Step {self.last_step}: New best metric {current:.4f}')
        else:
            self.num_bad_steps += 1
            if self.verbose:
                print(f'Step {self.last_step}: No improvement ({current:.4f} vs best {self.best:.4f}). '
                      f'Bad steps: {self.num_bad_steps}/{self.patience}')
        
        if self.num_bad_steps > self.patience:
            if self.cooldown_counter <= 0:
                self._reduce_lr()
                self.cooldown_counter = self.cooldown
                self.num_bad_steps = 0
    
    def _reduce_lr(self):
        """Reduce learning rate by factor"""
        # Extend min_lrs if needed
        if len(self.optimizer.param_groups) > len(self.min_lrs):
            additional_groups = len(self.optimizer.param_groups) - len(self.min_lrs)
            self.min_lrs.extend([self.min_lr] * additional_groups)
            if self.verbose:
                print(f"Added {additional_groups} new min_lr values for new parameter groups")
        
        # Extend current_base_lrs if needed
        if len(self.optimizer.param_groups) > len(self.current_base_lrs):
            additional_groups = len(self.optimizer.param_groups) - len(self.current_base_lrs)
            self.current_base_lrs.extend([self.current_base_lrs[0]] * additional_groups)
        
        for i, param_group in enumerate(self.optimizer.param_groups):
            # ✅ FIX: Use our tracked base LR, not current param group LR
            old_base_lr = self.current_base_lrs[i]
            min_lr_val = self.min_lrs[i] if i < len(self.min_lrs) else self.min_lr
            new_base_lr = max(old_base_lr * self.factor, min_lr_val)
            
            # ✅ Update our independent tracking
            self.current_base_lrs[i] = new_base_lr
            
            # Update optimizer (will be overridden by layer scheduler later)
            param_group['lr'] = new_base_lr
            
            if self.verbose:
                print(f'Step {self.last_step}: Reducing BASE learning rate of group {i} to {new_base_lr:.4e}.')
    
    def _is_better(self, current, best):
        """Check if current metric is better than best so far"""
        if self.mode == 'min':
            if self.threshold_mode == 'rel':
                return current < best * (1 - self.threshold)
            else:  # 'abs'
                return current < best - self.threshold
        else:  # mode == 'max'
            if self.threshold_mode == 'rel':
                return current > best * (1 + self.threshold)
            else:  # 'abs'
                return current > best + self.threshold

class AuxScheduler:
    """Manages auxiliary loss weight schedules using unified schedule functions - step-based"""
    
    def __init__(self, schedules_config: List[Dict[str, Any]], verbose: bool = False):
        self.schedulers = []
        self.verbose = verbose
        
        for i, config in enumerate(schedules_config):
            # Create schedule function from config
            schedule_function = ScheduleFunction.create(config)
            
            # Create scheduler using the schedule function
            scheduler = FunctionBasedScheduler(schedule_function, optimizer=None, verbose=verbose)
            self.schedulers.append(scheduler)
            
            if self.verbose:
                print(f"Created auxiliary scheduler {i}: {config}")
    
    def step(self, step: int) -> None:
        """Update all schedulers to current step"""
        for scheduler in self.schedulers:
            scheduler.last_step = step - 1  # Convert to 0-based indexing
    
    def get_values(self, step: int) -> List[float]:
        """Get schedule values for current step"""
        # Update all schedulers to current step
        self.step(step)
        
        # Return values from all schedulers
        return [scheduler.get_value() for scheduler in self.schedulers]


class LayerLRScheduler:
    """
    Manages layer-specific learning rate schedules using unified schedule functions - step-based.
    Now includes adaptive functionality that can monitor gradients and adjust LR accordingly.
    FIXED VERSION: Enhanced parameter detection and proper gradient collection.
    """
    
    def __init__(self, optimizer: torch.optim.Optimizer, model: torch.nn.Module,
                 schedules_config: List[Dict[str, Any]], verbose: bool = False):
        """
        Initialize the layer-specific LR scheduler with adaptive capabilities.
        
        Args:
            optimizer: PyTorch optimizer
            model: PyTorch model with named parameters
            schedules_config: List of schedule configs for each layer group
            verbose: If True, prints debugging information
        """
        self.optimizer = optimizer
        self.model = model
        self.verbose = verbose
        self.schedulers = []
        self.layer_to_scheduler_idx = {}  # Maps layer_name -> scheduler_idx
        self.base_lrs = {}
        
        # Track adaptive schedulers
        self.adaptive_functions = {}  # Maps layer_name -> AdaptiveFunction
        self.has_adaptive = False
        
        # Debug counters
        self._debug_step_count = 0
        
        # Map parameters to their base learning rates
        for group_idx, group in enumerate(optimizer.param_groups):
            for param in group['params']:
                self.base_lrs[param] = group['lr']
        
        # Create schedules for each layer group
        for idx, config in enumerate(schedules_config):
            layer_patterns = config.get('layers', [])
            
            # Create schedule function from config
            schedule_function = ScheduleFunction.create(config)
            
            # If this is an adaptive function, set up the reference
            if isinstance(schedule_function, AdaptiveFunction):
                self.has_adaptive = True
                layer_name = layer_patterns[0] if layer_patterns else f"layer_{idx}"
                schedule_function.layer_name = layer_name
                schedule_function.scheduler_ref = self
                self.adaptive_functions[layer_name] = schedule_function
            
            # Create scheduler using the schedule function
            scheduler = FunctionBasedScheduler(schedule_function, optimizer=None, verbose=verbose)
            self.schedulers.append(scheduler)
            
            # Map layer names to this scheduler
            for layer_pattern in layer_patterns:
                self.layer_to_scheduler_idx[layer_pattern] = idx
                
            if self.verbose:
                schedule_type = config.get('type', 'unknown')
                print(f"Created layer scheduler {idx} ({schedule_type}) for patterns: {layer_patterns}")
        
        if self.has_adaptive and self.verbose:
            print(f"🔄 Adaptive LR enabled for {len(self.adaptive_functions)} layer groups")
    
    def _collect_gradient_stats(self) -> Dict[str, float]:
        """
        Collect gradient statistics for adaptive adjustment - ENHANCED VERSION.
        Uses robust parameter matching with multiple pattern strategies.
        """
        if not self.has_adaptive:
            return {}
        
        layer_stats = {}
        self._debug_step_count += 1
        
        for layer_name, adaptive_func in self.adaptive_functions.items():
            total_grad_norm_sq = 0.0
            param_count = 0
            matched_params = []
            
            # Enhanced pattern matching strategies
            patterns_to_check = [layer_name]
            
            # Add specific patterns based on layer name
            if 'embedding' in layer_name:
                patterns_to_check.extend(['embedding.weight', 'embed', '.embedding.'])
            elif 'm_layers' in layer_name:
                patterns_to_check.extend(['m_layers.', 'm_layer', '.m_layers.'])
            elif 'p_layers' in layer_name:
                patterns_to_check.extend(['p_layers.', 'p_layer', '.p_layers.'])
            elif 'n_layers' in layer_name:
                patterns_to_check.extend(['n_layers.', 'n_layer', '.n_layers.'])
            elif 'k_layers' in layer_name:
                patterns_to_check.extend(['k_layers.', 'k_layer', '.k_layers.'])
            elif 'l_layers' in layer_name:
                patterns_to_check.extend(['l_layers.', 'l_layer', '.l_layers.'])
            
            # Find parameters for this layer pattern
            for name, param in self.model.named_parameters():
                if param.grad is not None and param.requires_grad:
                    # Check if parameter name matches any of our patterns
                    param_matches = any(pattern in name for pattern in patterns_to_check)
                    
                    if param_matches:
                        grad_norm = param.grad.norm().item()
                        total_grad_norm_sq += grad_norm ** 2
                        param_count += 1
                        matched_params.append(name)
            
            if param_count > 0:
                mean_grad_norm = (total_grad_norm_sq / param_count) ** 0.5
                layer_stats[layer_name] = mean_grad_norm
                
                # Enhanced debugging for first few steps
                if self._debug_step_count <= 5:
                    print(f"📊 Layer {layer_name}: {param_count} params, grad_norm={mean_grad_norm:.2e}")
                    print(f"    Patterns: {patterns_to_check}")
                    print(f"    Matched: {matched_params[:3]}{'...' if len(matched_params) > 3 else ''}")
            else:
                if self._debug_step_count <= 5:
                    print(f"⚠️  No parameters found for layer {layer_name}")
                    print(f"    Patterns searched: {patterns_to_check}")
                    print(f"    Available params: {[name for name, _ in self.model.named_parameters() if _.requires_grad][:5]}...")
        
        return layer_stats
    
    def _get_reference_gradient_norm(self, layer_stats: Dict[str, float]) -> float:
        """Get reference gradient norm from reference layers - ENHANCED WITH DEBUGGING"""
        reference_norm = 0.0
        
        # Get reference layers from any adaptive function
        reference_layers = ['m_layers', 'p_layers']  # Default
        if self.adaptive_functions:
            first_adaptive = next(iter(self.adaptive_functions.values()))
            reference_layers = first_adaptive.reference_layers
        
        # Enhanced debugging for first few steps
        if self._debug_step_count <= 5:
            print(f"🔍 Reference calculation: looking for {reference_layers}")
            print(f"    Available layer stats: {list(layer_stats.keys())}")
            print(f"    Layer stat values: {layer_stats}")
        
        # Use average of available reference norms
        reference_norms = []
        for ref_layer in reference_layers:
            if ref_layer in layer_stats and layer_stats[ref_layer] > 0:
                reference_norms.append(layer_stats[ref_layer])
                if self._debug_step_count <= 5:
                    print(f"    ✅ Found {ref_layer}: {layer_stats[ref_layer]:.2e}")
        
        if reference_norms:
            reference_norm = sum(reference_norms) / len(reference_norms)
            if self._debug_step_count <= 5:
                print(f"    📊 Computed reference norm: {reference_norm:.2e}")
        else:
            # Fallback: use average of all available layers
            all_norms = [norm for norm in layer_stats.values() if norm > 0]
            if all_norms:
                reference_norm = sum(all_norms) / len(all_norms)
                print(f"⚠️ Using fallback reference norm: {reference_norm:.2e}")
            else:
                reference_norm = 1e-6  # Minimum fallback
                print(f"⚠️ Using minimum fallback reference norm: {reference_norm:.2e}")
        
        return reference_norm
    
    def step(self, step: int) -> None:
        """Update learning rates with adaptive adjustment if enabled."""
        # Update all schedulers to current step
        for scheduler in self.schedulers:
            scheduler.last_step = step - 1  # Convert to 0-based indexing
        
        # Handle adaptive updates if any
        if self.has_adaptive:
            layer_stats = self._collect_gradient_stats()
            reference_norm = self._get_reference_gradient_norm(layer_stats)
            
            # Update adaptive functions
            for layer_name, adaptive_func in self.adaptive_functions.items():
                if layer_name in layer_stats:
                    current_norm = layer_stats[layer_name]
                    adaptive_func.update_adaptive_value(step, current_norm, reference_norm)
        
        # Apply learning rates
        self._apply_learning_rates()
    
    def _apply_learning_rates(self):
        """
        Apply current learning rates to optimizer parameter groups by layer.
        BULLETPROOF VERSION: Uses parameter index mapping to avoid tensor comparison.
        """
        try:
            # Create a comprehensive mapping of parameters to their intended learning rates
            param_to_new_lr = {}
            
            # Get multipliers for each layer
            layer_multipliers = {}
            for layer_pattern, scheduler_idx in self.layer_to_scheduler_idx.items():
                scheduler = self.schedulers[scheduler_idx]
                layer_multipliers[layer_pattern] = scheduler.get_value()
            
            # Assign learning rates to parameters based on their names
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    # Find the appropriate multiplier for this parameter
                    multiplier = 1.0  # Default
                    matched_layer = None
                    
                    for layer_pattern, mult in layer_multipliers.items():
                        if layer_pattern in name:
                            multiplier = mult
                            matched_layer = layer_pattern
                            break
                    
                    # Calculate new learning rate
                    base_lr = self.base_lrs.get(param, 0.001)
                    new_lr = base_lr * multiplier
                    param_to_new_lr[param] = (new_lr, matched_layer or 'unassigned')
            
            # Update optimizer parameter groups
            for group_idx, group in enumerate(self.optimizer.param_groups):
                if group['params']:
                    # Use the first parameter in the group to determine the group's LR
                    first_param = group['params'][0]
                    if first_param in param_to_new_lr:
                        new_lr, layer_name = param_to_new_lr[first_param]
                        group['lr'] = new_lr
                        group['group_name'] = layer_name
                        
                        if self._debug_step_count <= 3 and self.verbose:
                            print(f"  Updated group {group_idx} ({layer_name}): {len(group['params'])} params, LR = {new_lr:.6f}")
            
        except Exception as e:
            print(f"⚠️  Error in _apply_learning_rates: {str(e)}")
            if self._debug_step_count <= 3:
                import traceback
                traceback.print_exc()
    
    def get_lr_multipliers(self, step: int) -> Dict[str, float]:
        """Get the current LR multipliers for logging purposes."""
        # Update all schedulers to current step
        for scheduler in self.schedulers:
            scheduler.last_step = step - 1
            
        multipliers = {}
        
        # Get multipliers for each scheduler
        for layer_pattern, scheduler_idx in self.layer_to_scheduler_idx.items():
            scheduler = self.schedulers[scheduler_idx]
            multiplier = scheduler.get_value()
            
            # Use simplified name for logging
            short_name = self._get_short_name(layer_pattern)
            if short_name:
                multipliers[short_name] = multiplier
            else:
                multipliers[layer_pattern] = multiplier
        
        return multipliers
    
    def _get_short_name(self, layer_pattern: str) -> str:
        """Convert layer pattern to a shorter name for logging"""
        if layer_pattern == 'embedding':
            return "emb"
        elif layer_pattern == 'n_layers':
            return "n_layers"
        elif layer_pattern == 'k_layers':
            return "k_layers"
        elif layer_pattern == 'l_layers':
            return "l_layers"
        elif layer_pattern == 'p_layers':
            return "p_layers"
        elif layer_pattern == 'm_layers':
            return "m_layers"
        
        return layer_pattern
    
    def debug_parameter_detection(self):
        """Debug method to check parameter detection"""
        print("🔍 PARAMETER DETECTION DEBUG:")
        print("Model parameters:")
        param_names = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                grad_status = "HAS_GRAD" if param.grad is not None else "NO_GRAD"
                param_names.append(name)
                if len(param_names) <= 10:  # Show first 10 params
                    print(f"  {name}: {param.shape} [{grad_status}]")
        
        if len(param_names) > 10:
            print(f"  ... and {len(param_names) - 10} more parameters")
        
        print(f"\nLayer patterns: {list(self.layer_to_scheduler_idx.keys())}")
        
        for layer_pattern in self.layer_to_scheduler_idx.keys():
            matching = []
            for name in param_names:
                if layer_pattern in name:
                    matching.append(name)
            
            print(f"  {layer_pattern}: {len(matching)} matches")
            if matching:
                print(f"    Examples: {matching[:3]}{'...' if len(matching) > 3 else ''}")
            else:
                print(f"    ⚠️  NO MATCHES FOUND")
    
    def get_adaptive_status(self) -> Dict[str, Any]:
        """Get status of adaptive learning rates for debugging."""
        if not self.has_adaptive:
            return {"adaptive_enabled": False}
        
        status = {
            "adaptive_enabled": True,
            "adaptive_layers": {},
        }
        
        for layer_name, adaptive_func in self.adaptive_functions.items():
            status["adaptive_layers"][layer_name] = {
                "current_value": adaptive_func.current_value,
                "gradient_history": adaptive_func.gradient_history.copy(),
                "last_update_step": adaptive_func.last_update_step,
            }
        
        return status


# Create a factory function for instantiating schedulers
def create_scheduler(name: str, optimizer: torch.optim.Optimizer, config: Dict[str, Any]) -> Any:
    """Create a step-based scheduler based on name and config"""
    verbose = config.pop('verbose', False)  # Remove verbose from config if present
    
    if name == 'ReduceLROnPlateau':
        return StepBasedReduceLROnPlateau(optimizer, verbose=verbose, **config)
    elif name == 'TransformerWarmupScheduler':
        return TransformerWarmupScheduler(optimizer, verbose=verbose, **config)
    elif name == 'WarmupCooldownScheduler':
        return WarmupCooldownScheduler(optimizer, verbose=verbose, **config)
    elif name == 'StepLR':
        # Convert PyTorch StepLR to step-based
        step_size = config.get('step_size', 1000)
        gamma = config.get('gamma', 0.1)
        return FunctionBasedScheduler(
            StepFunction(1.0, gamma, [step_size * i for i in range(1, 100)]),
            optimizer, verbose=verbose
        )
    elif name == 'ExponentialLR':
        # Convert PyTorch ExponentialLR to step-based
        gamma = config.get('gamma', 0.95)
        decay_steps = config.get('decay_steps', 1000)
        return FunctionBasedScheduler(
            ExponentialFunction(1.0, gamma, decay_steps),
            optimizer, verbose=verbose
        )
    elif name == 'PolynomialLR':
        # Polynomial decay scheduler
        total_steps = config.get('total_steps', 10000)
        power = config.get('power', 1.0)
        return FunctionBasedScheduler(
            PolynomialFunction(1.0, 0.0, power, total_steps),
            optimizer, verbose=verbose
        )
    elif name == 'CosineAnnealingLR':
        # Convert PyTorch CosineAnnealingLR to step-based
        T_max = config.get('T_max', 1000)
        eta_min = config.get('eta_min', 0.0)
        return FunctionBasedScheduler(
            CosineFunction(1.0, eta_min, 0, T_max),
            optimizer, verbose=verbose
        )
    else:
        try:
            # Try to create a custom scheduler based on schedule function
            schedule_function = ScheduleFunction.create({'type': name, **config})
            return FunctionBasedScheduler(schedule_function, optimizer, verbose=verbose)
        except Exception as e:
            raise ValueError(f"Unknown scheduler: {name}. Error: {e}")


# Compatibility aliases for backward compatibility
StepCounterReduceLROnPlateau = StepBasedReduceLROnPlateau
WarmupCooldownLR = WarmupCooldownScheduler