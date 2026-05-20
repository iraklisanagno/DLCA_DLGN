from typing import Dict, Callable, List
from torch.utils.data import DataLoader
from datetime import datetime
import os
import json
import torch._dynamo
torch._dynamo.config.suppress_errors = True

import wandb
import inspect
import time
import torch
import copy

def parse_value(value):
    """Convert string values to proper types"""
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        metrics: Dict[str, Callable],
        config: dict,
        tokenized_data: dict = None,
        visualizations: Dict[str, Callable] = None,
        is_synced: bool = True
    ):
        self.model = model.to(config.get('device', torch.device("cuda" if torch.cuda.is_available() else "cpu")))
        self.train_loader = tokenized_data["train"]
        self.val_loader = tokenized_data["validation"]
        self.test_loader = tokenized_data["test"]
        self.tokenizer_interface = tokenized_data["model_interface"]
        self.metrics = metrics
        self.config = config
        self.visualization = visualizations
        self.is_synced = is_synced

        self.gradient_log_frequency = self.config['training'].get('gradient_log_frequency', None)
        self.log_gradients = self.gradient_log_frequency is not None
        
        # Enhanced logging options
        self.log_parameter_updates = self.config['training'].get('log_parameter_updates', True)
        self.log_gradient_norms = self.config['training'].get('log_gradient_norms', True)
        
        # Collapse mode configuration
        self.collapse_mode = self.config['training'].get('collapse_mode', False)
        self.collapse_modes = ['train', 'eval', 'eval_col_emb', 'eval_col_layer', 'eval_col_all']
        
        setattr(self.model, 'is_synced', is_synced)

        self._init_components()
        self._compute_and_store_total_params()
        self._init_wandb()

        if self.visualization:
            self.fixed_val_batch = next(iter(self.val_loader))
            
        self.global_step = 0
        self.current_weights = []
        
        # Storage for parameter tracking
        self.prev_params = None
        if self.log_parameter_updates:
            self._store_parameters()

        self._run_initial_evaluation()

    def _store_parameters(self):
        """Store current parameters for update calculation"""
        self.prev_params = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.prev_params[name] = param.data.clone().detach()

    def _run_initial_evaluation(self):
        """Run initial validation and visualization at step 0"""
        self.current_weights = self.aux_schedules.get_values(0)
        val_metrics = self._run_validation()
        
        initial_log_data = {}
        initial_log_data.update(val_metrics)

        
        for i, weight in enumerate(self.current_weights):
            initial_log_data[f'hyperparameter/aux_{i}_weight'] = weight
        
        self._log_learning_rates(initial_log_data)
        initial_log_data["epoch"] = 0
        initial_log_data["step"] = 0
        initial_log_data["phase"] = "initial_evaluation"
        
        try:
            wandb.log(initial_log_data, step=0)
        except Exception as e:
            print(f"⚠️  Error logging initial metrics to wandb: {str(e)}")
        
        if self.visualization:
            try:
                self._log_visualizations(step=0)
            except Exception as e:
                print(f"⚠️  Error during initial visualization: {str(e)}")

    def _compute_and_store_total_params(self):
        """Computes total trainable parameters and stores in config."""
        if hasattr(self.model, 'print_trainable_params') and callable(getattr(self.model, 'print_trainable_params')):
            total_params = self.model.print_trainable_params()
            self.config['model']['total_trainable_params'] = total_params

    def _init_components(self):
        self.losses = {
            self.config['training']['main_loss']: self._get_loss_function()
        }

        optim_cfg = self.config['training']['optimizer']
        self.opt = getattr(torch.optim, optim_cfg['name'])(
            self.model.parameters(),
            **optim_cfg['params']
        )

        sched_cfg = self.config['training'].get('scheduler')
        if sched_cfg:
            from utils.scheduler import create_scheduler
            self.sched = create_scheduler(
                sched_cfg['name'],
                self.opt,
                sched_cfg['params']
            )
        else:
            self.sched = None

        from utils.scheduler import AuxScheduler
        self.aux_schedules = AuxScheduler(self.config['training'].get('aux_scheduler', []))

        from utils.scheduler import LayerLRScheduler
        if self.config['training'].get('layer_lr_scheduler'):
            self.layer_lr_scheduler = LayerLRScheduler(
                self.opt, 
                self.model, 
                self.config['training']['layer_lr_scheduler'],
                verbose=False
            )
        else:
            self.layer_lr_scheduler = None

        self.device = next(self.model.parameters()).device

    def _get_loss_function(self):
        loss_name = self.config['training']['main_loss']
        label_smoothing = self.config['training'].get('label_smoothing', 0.1)
        
        if loss_name == 'cross_entropy':
            return torch.nn.CrossEntropyLoss(
                ignore_index=self.tokenizer_interface['padding_idx'],
                label_smoothing=label_smoothing
            )
        elif loss_name == 'nll':
            return torch.nn.NLLLoss(ignore_index=self.tokenizer_interface['padding_idx'])
        elif loss_name == 'mse':
            return torch.nn.MSELoss()
        elif loss_name == 'l1':
            return torch.nn.L1Loss()
        elif loss_name == 'smooth_l1':
            return torch.nn.SmoothL1Loss()
        elif loss_name == 'kl_div':
            return torch.nn.KLDivLoss(reduction='batchmean')
        elif loss_name == 'bce':
            return torch.nn.BCEWithLogitsLoss()
        else:
            raise ValueError(f"Unsupported loss function: {loss_name}")

    def _init_wandb(self):
        """Initialize wandb or use existing run if already initialized."""
        wandb_cfg = self.config.get('wandb', {})
        wandb_dir = wandb_cfg.get('dir', '/mnt/tmp/wandb')
        os.makedirs(wandb_dir, exist_ok=True)
        
        self.config['device'] = str(self.device)
        self.config['model_is_synced'] = self.is_synced
        self.config['gradient_logging_enabled'] = self.log_gradients
        self.config['parameter_update_logging_enabled'] = self.log_parameter_updates
        self.config['gradient_norm_logging_enabled'] = self.log_gradient_norms
        self.config['collapse_mode_enabled'] = self.collapse_mode
        
        if self.layer_lr_scheduler and hasattr(self.layer_lr_scheduler, 'has_adaptive'):
            self.config['adaptive_lr_enabled'] = self.layer_lr_scheduler.has_adaptive
            if self.layer_lr_scheduler.has_adaptive:
                self.config['adaptive_lr_layers'] = list(self.layer_lr_scheduler.adaptive_functions.keys())
        
        if wandb.run is not None:
            wandb.config.update(self.config, allow_val_change=True)
            wandb.watch(self.model, log="all", log_freq=50)
            return
        
        wandb.init(
            project=wandb_cfg.get('project', 'Architecture'),
            name=self._build_run_name(),
            config=self.config,
            dir=wandb_dir,
            settings=wandb.Settings(start_method="fork")
        )
        
        wandb.watch(self.model, log="all", log_freq=50)

    def _build_run_name(self):
        timestamp = datetime.now().strftime("%m%d_%H%M")
        model_name = self.config['model']['name']
        opt_name = self.config['training']['optimizer']['name']
        lr = self.config['training']['optimizer']['params']['lr']
        sync_prefix = "synced" if self.is_synced else "unsynced"
        grad_suffix = "_grad" if self.log_gradients else ""
        update_suffix = "_upd" if self.log_parameter_updates else ""
        collapse_suffix = "_collapse" if self.collapse_mode else ""
        
        adaptive_suffix = ""
        if self.layer_lr_scheduler and hasattr(self.layer_lr_scheduler, 'has_adaptive') and self.layer_lr_scheduler.has_adaptive:
            adaptive_suffix = "_adaptive"
        
        return f"{sync_prefix}_{model_name}_{opt_name}_lr{lr:.0e}_{timestamp}{grad_suffix}{update_suffix}{adaptive_suffix}{collapse_suffix}"

    def _is_valid_loss(self, loss_tensor):
        """Check if loss is valid (not NaN or inf)"""
        if loss_tensor is None:
            return False
        return torch.isfinite(loss_tensor).all()

    def _safe_loss_calculation(self, predictions_flat, targets_flat):
        """Calculate loss with error handling and NaN/inf protection"""
        try:
            loss = self.losses[self.config['training']['main_loss']](predictions_flat, targets_flat)
            
            if not self._is_valid_loss(loss):
                return torch.tensor(0.0, device=predictions_flat.device, requires_grad=True)
            
            return loss
            
        except Exception as e:
            return torch.tensor(0.0, device=predictions_flat.device, requires_grad=True)

    def _should_log_gradients(self):
        """Determine if gradients should be logged at this step"""
        return self.log_gradients and (self.global_step % self.gradient_log_frequency == 0)

    def _log_gradients_and_updates(self):
        """Enhanced logging for gradients, parameter updates, and norms"""
        if not self.log_gradients:
            return {}
            
        logging_stats = {}
        
        try:
            key_layer_groups = {
                'embedding': ['embedding', 'embed'],
                'n_layers': ['n_layers', 'n_layer'],
                'k_layers': ['k_layers', 'k_layer'], 
                'l_layers': ['l_layers', 'l_layer'],
                'p_layers': ['p_layers', 'p_layer'],
                'm_layers': ['m_layers', 'm_layer']
            }
            
            group_grad_stats = {group_name: [] for group_name in key_layer_groups.keys()}
            group_update_stats = {group_name: [] for group_name in key_layer_groups.keys()}
            
            total_grad_norm = 0.0
            total_update_norm = 0.0
            param_count = 0
            
            for name, param in self.model.named_parameters():
                if param.grad is not None and param.requires_grad:
                    # Gradient statistics
                    grad_abs_mean = torch.abs(param.grad).mean().item()
                    grad_norm = torch.norm(param.grad).item()
                    total_grad_norm += grad_norm ** 2
                    
                    # LOG SCALAR VALUES alongside histograms
                    logging_stats[f'gradients/{name}.abs_mean'] = grad_abs_mean
                    logging_stats[f'gradients/{name}.norm'] = grad_norm
                    
                    # Parameter update statistics
                    if self.log_parameter_updates and self.prev_params and name in self.prev_params:
                        param_update = param.data - self.prev_params[name]
                        update_abs_mean = torch.abs(param_update).mean().item()
                        update_norm = torch.norm(param_update).item()
                        total_update_norm += update_norm ** 2
                        
                        # LOG SCALAR VALUES for updates
                        logging_stats[f'updates/{name}.abs_mean'] = update_abs_mean
                        logging_stats[f'updates/{name}.norm'] = update_norm
                        
                        # Assign to layer groups
                        for group_name, group_patterns in key_layer_groups.items():
                            if any(pattern in name for pattern in group_patterns):
                                group_grad_stats[group_name].append(grad_abs_mean)
                                group_update_stats[group_name].append(update_abs_mean)
                                break
                    else:
                        # Assign gradients to layer groups
                        for group_name, group_patterns in key_layer_groups.items():
                            if any(pattern in name for pattern in group_patterns):
                                group_grad_stats[group_name].append(grad_abs_mean)
                                break
                    
                    param_count += 1
            
            # Log group-wise gradient statistics (SCALAR VALUES)
            for group_name, grad_means in group_grad_stats.items():
                if grad_means:
                    avg_grad_mean = sum(grad_means) / len(grad_means)
                    logging_stats[f'gradients/{group_name}/abs_mean'] = avg_grad_mean
            
            # Log group-wise update statistics (SCALAR VALUES)
            if self.log_parameter_updates:
                for group_name, update_means in group_update_stats.items():
                    if update_means:
                        avg_update_mean = sum(update_means) / len(update_means)
                        logging_stats[f'updates/{group_name}/abs_mean'] = avg_update_mean
            
            # Log global norms
            if self.log_gradient_norms and param_count > 0:
                global_grad_norm = (total_grad_norm ** 0.5)
                logging_stats['gradients/global_norm'] = global_grad_norm
                
                if self.log_parameter_updates:
                    global_update_norm = (total_update_norm ** 0.5)
                    logging_stats['updates/global_norm'] = global_update_norm
                    
                    if global_grad_norm > 1e-8:
                        logging_stats['updates/grad_ratio'] = global_update_norm / global_grad_norm
                        
        except Exception as e:
            return {}
            
        return logging_stats

    def _get_layer_lr_multipliers(self):
        """Get current LR multipliers for each layer group"""
        if not self.layer_lr_scheduler:
            return {}
        
        try:
            raw_multipliers = self.layer_lr_scheduler.get_lr_multipliers(self.global_step)
            return raw_multipliers
            
        except Exception as e:
            return {}

    def _log_learning_rates(self, log_data):
        """Log base scheduler LR and effective layer LRs with proper separation"""
        try:
            # Get the base learning rate that the main scheduler would set
            base_scheduler_lr = self.config['training']['optimizer']['params']['lr']  # Default
            
            if self.sched and hasattr(self.sched, 'get_last_lr'):
                try:
                    scheduler_lrs = self.sched.get_last_lr()
                    if scheduler_lrs:
                        base_scheduler_lr = scheduler_lrs[0]
                        log_data["hyperparameter/lr_base_scheduler"] = base_scheduler_lr
                except Exception as e:
                    pass
            
            if self.layer_lr_scheduler:
                try:
                    layer_lr_multipliers = self._get_layer_lr_multipliers()
                    
                    # Log effective LRs for each layer (base_scheduler_lr * multiplier)
                    for layer_group, multiplier in layer_lr_multipliers.items():
                        effective_lr = base_scheduler_lr * multiplier
                        log_data[f'hyperparameter/lr_effective_{layer_group}'] = effective_lr
                        # Also log the multiplier itself
                        log_data[f'hyperparameter/lr_mult_{layer_group}'] = multiplier
                        
                except Exception as e:
                    pass
        
        except Exception as e:
            pass

    def _step(self, batch, is_train=True, mode_prefix=None):
        if batch is None:
            prefix = f"{mode_prefix}/" if mode_prefix else "train/" if is_train else "val/"
            return {
                f'{prefix}loss/main': 0.0,
                f'{prefix}metric/accuracy': 0.0,
                f'{prefix}metric/bleu': 0.0
            }

        src, tgt = [t.to(self.device) for t in batch]
        
        pad_idx = self.tokenizer_interface['padding_idx']
        if (src == pad_idx).all() or (tgt == pad_idx).all():
            prefix = f"{mode_prefix}/" if mode_prefix else "train/" if is_train else "val/"
            return {
                f'{prefix}loss/main': 0.0,
                f'{prefix}metric/accuracy': 0.0,
                f'{prefix}metric/bleu': 0.0
            }

        try:
            if self.is_synced:
                outputs = self.model(src)
                if isinstance(outputs, tuple):
                    logits, *aux_losses = outputs
                else:
                    logits = outputs
                    aux_losses = []
                aligned_logits = logits
                aligned_target = tgt
            else:
                outputs = self.model(src, tgt)
                
                if isinstance(outputs, tuple):
                    logits, *aux_losses = outputs
                else:
                    logits = outputs
                    aux_losses = []
                
                if logits.size(1) == tgt.size(1):
                    aligned_logits = logits[:, :-1, :].contiguous()
                    aligned_target = tgt[:, 1:].contiguous()
                elif logits.size(1) == tgt.size(1) - 1:
                    aligned_logits = logits
                    aligned_target = tgt[:, 1:].contiguous()
                else:
                    min_len = min(logits.size(1), tgt.size(1) - 1)
                    aligned_logits = logits[:, :min_len, :].contiguous()
                    aligned_target = tgt[:, 1:1+min_len].contiguous()

        except Exception as e:
            prefix = f"{mode_prefix}/" if mode_prefix else "train/" if is_train else "val/"
            return {
                f'{prefix}loss/main': 0.0,
                f'{prefix}metric/accuracy': 0.0,
                f'{prefix}metric/bleu': 0.0
            }

        predictions_flat = aligned_logits.reshape(-1, aligned_logits.size(-1))
        targets_flat = aligned_target.reshape(-1)
        main_loss = self._safe_loss_calculation(predictions_flat, targets_flat)

        prefix = f"{mode_prefix}/" if mode_prefix else "train/" if is_train else "val/"
        log_dict = {f'{prefix}loss/main': main_loss.item()}
        
        if is_train:
            total_loss = main_loss.clone()
            
            for i, aux_loss in enumerate(aux_losses):
                try:
                    if self._is_valid_loss(aux_loss):
                        weight = self.current_weights[i] if i < len(self.current_weights) else 1.0
                        if aux_loss.requires_grad:
                            weighted_loss = aux_loss * weight
                            total_loss = total_loss + weighted_loss
                        log_dict[f'{prefix}loss/aux_{i}'] = aux_loss.item()
                    else:
                        log_dict[f'{prefix}loss/aux_{i}'] = 0.0
                except Exception as e:
                    log_dict[f'{prefix}loss/aux_{i}'] = 0.0

            try:
                self.opt.zero_grad()
                
                if self._is_valid_loss(total_loss):
                    total_loss.backward()
                    
                    # Enhanced gradient and update logging
                    if self._should_log_gradients():
                        gradient_update_stats = self._log_gradients_and_updates()
                        log_dict.update(gradient_update_stats)
                
                if self.config['training'].get('gradient_clipping', False):
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        max_norm=self.config['training'].get('gradient_clip_val', 1.0)
                    )
                    log_dict[f'{prefix}gradients/global_norm_clipped'] = grad_norm.item()
                
                self.opt.step()
                
                # Store parameters after update for next iteration
                if self.log_parameter_updates:
                    self._store_parameters()
                
                # Call main scheduler first (sets base LR)
                if self.sched:
                    try:
                        from utils.scheduler import StepBasedReduceLROnPlateau
                        if isinstance(self.sched, StepBasedReduceLROnPlateau):
                            self.sched.step(main_loss.item())
                        else:
                            self.sched.step()
                    except Exception as e:
                        pass
                
                # Then call layer scheduler (applies multipliers to base LR)
                if self.layer_lr_scheduler:
                    try:
                        self.layer_lr_scheduler.step(self.global_step)
                    except Exception as e:
                        pass
                
            except Exception as e:
                pass
        else:
            for i, aux_loss in enumerate(aux_losses):
                try:
                    if self._is_valid_loss(aux_loss):
                        log_dict[f'{prefix}loss/aux_{i}'] = aux_loss.item()
                    else:
                        log_dict[f'{prefix}loss/aux_{i}'] = 0.0
                except Exception as e:
                    log_dict[f'{prefix}loss/aux_{i}'] = 0.0

        with torch.no_grad():
            try:
                if hasattr(self.model, 'get_predictions'):
                    metric_logits = self.model.get_predictions(outputs)
                    if not self.is_synced and metric_logits.size(1) != aligned_target.size(1):
                        if metric_logits.size(1) == aligned_target.size(1) + 1:
                            metric_logits = metric_logits[:, :-1, :].contiguous()
                        elif metric_logits.size(1) + 1 == aligned_target.size(1):
                            metric_logits = metric_logits[:, 1:, :].contiguous()
                        else:
                            metric_logits = aligned_logits
                else:
                    metric_logits = aligned_logits
                
                for name, metric_fn in self.metrics.items():
                    try:
                        sig = inspect.signature(metric_fn)
                        kwargs = {}
                        if 'vocab' in sig.parameters:
                            kwargs['vocab'] = self.tokenizer_interface['tgt_vocab']
                        if 'pad_index' in sig.parameters:
                            kwargs['pad_index'] = self.tokenizer_interface['padding_idx']
                            
                        metric_result = metric_fn(metric_logits, aligned_target, **kwargs)
                        
                        if isinstance(metric_result, torch.Tensor):
                            if self._is_valid_loss(metric_result):
                                log_dict[f'{prefix}metric/{name}'] = metric_result.item()
                            else:
                                log_dict[f'{prefix}metric/{name}'] = 0.0
                        else:
                            log_dict[f'{prefix}metric/{name}'] = metric_result
                            
                    except Exception as e:
                        log_dict[f'{prefix}metric/{name}'] = 0.0
                        
            except Exception as e:
                for name in self.metrics.keys():
                    log_dict[f'{prefix}metric/{name}'] = 0.0

        return log_dict

    def _agg(self, logs):
        """Aggregate log dictionaries with error handling"""
        if not logs:
            return {}
            
        valid_logs = [log for log in logs if log is not None and isinstance(log, dict)]
        
        if not valid_logs:
            return {}
            
        keys = valid_logs[0].keys()
        aggregated = {}
        
        for k in keys:
            try:
                values = [d.get(k, 0) for d in valid_logs]
                numeric_values = [v for v in values if isinstance(v, (int, float)) and not (isinstance(v, float) and (v != v or abs(v) == float('inf')))]
                
                if numeric_values:
                    aggregated[k] = sum(numeric_values) / len(numeric_values)
                else:
                    aggregated[k] = 0.0
            except Exception as e:
                aggregated[k] = 0.0
                
        return aggregated

    def _run_validation(self):
        """Run validation pass and return aggregated metrics"""
        val_logs = []
        
        with torch.no_grad():
            self.model.eval()
            for batch in self.val_loader:
                try:
                    result = self._step(batch, is_train=False, mode_prefix='val')
                    if result is not None:
                        val_logs.append(result)
                except Exception as e:
                    pass
                                
            val_metrics = self._agg(val_logs)
            self.model.train()
            return val_metrics

    def _run_test_evaluation(self, mode_prefix='test'):
        """Run test evaluation and return aggregated metrics"""
        test_logs = []
        
        with torch.no_grad():
            self.model.eval()
            for batch in self.test_loader:
                try:
                    result = self._step(batch, is_train=False, mode_prefix=mode_prefix)
                    if result is not None:
                        test_logs.append(result)
                except Exception as e:
                    pass
                                
            test_metrics = self._agg(test_logs)
            self.model.train()
            return test_metrics

    def _run_collapse_mode_evaluation(self):
        """Run test evaluation across all collapse modes if enabled"""
        if not self.collapse_mode:
            return {}
        
        print("🔍 Running collapse mode evaluation across all model modes...")
        
        collapse_results = {}
        original_mode = None
        
        # Store original model mode if the method exists
        if hasattr(self.model, 'emb_mode'):
            original_mode = getattr(self.model, 'emb_mode', None)
        
        try:
            for mode in self.collapse_modes:
                print(f"  Evaluating in mode: {mode}")
                
                # Set model mode
                if hasattr(self.model, 'set_mode'):
                    self.model.set_mode(mode)
                else:
                    print(f"    Warning: Model doesn't have set_mode method, skipping {mode}")
                    continue
                
                # Run test evaluation with mode-specific prefix
                mode_metrics = self._run_test_evaluation(mode_prefix=f'collapse_{mode}')
                
                # Store results
                for metric_name, metric_value in mode_metrics.items():
                    collapse_results[metric_name] = metric_value
                
                print(f"    Mode {mode} completed - Loss: {mode_metrics.get(f'collapse_{mode}/loss/main', 'N/A'):.4f}")
        
        except Exception as e:
            print(f"  Error during collapse mode evaluation: {str(e)}")
        
        finally:
            # Restore original model mode
            if hasattr(self.model, 'set_mode'):
                if original_mode is not None:
                    # Restore original mode if we tracked it
                    if hasattr(self.model, 'emb_mode'):
                        self.model.emb_mode = original_mode
                else:
                    # Default to train mode
                    self.model.set_mode('train')
        
        print(f"🔍 Collapse mode evaluation completed - {len(collapse_results)} metrics logged")
        return collapse_results

    def _log_visualizations(self, epoch=None, step=None):
        """Generate and log visualizations"""
        if not self.visualization:
            return
            
        vis_log_dict = {}
        
        for vis_name, val in self.visualization.items():
            vis_fn, visualization_config = val
            try:
                vis_images = vis_fn(
                    model=self.model,
                    batch=self.fixed_val_batch,
                    tokenizer_interface=self.tokenizer_interface,
                    device=self.device,
                    visualization_config=visualization_config
                )
                
                if isinstance(vis_images, list):
                    for i, img in enumerate(vis_images):
                        vis_log_dict[f"visualization/{vis_name}_{i}"] = img
                else:
                    vis_log_dict[f"visualization/{vis_name}"] = vis_images
                    
            except Exception as e:
                pass

        if vis_log_dict:
            try:
                wandb.log(vis_log_dict, step=step if step is not None else epoch)
            except Exception as e:
                pass

    def train(self, epochs: int):
        self.current_epoch = 0
        log_every_n_steps = self.config['training'].get('log_every_n_steps', 100)
        visualize_every_n_steps = self.config['training'].get('visualize_every_n_steps', 0)
        
        step_based_visualization = visualize_every_n_steps > 0
        
        try:
            for epoch in range(1, epochs + 1):
                self.current_epoch = epoch
                epoch_train_logs = []
                
                self.model.train()
                
                for batch_idx, batch in enumerate(self.train_loader):
                    self.global_step += 1
                    self.current_weights = self.aux_schedules.get_values(self.global_step)
                    
                    try:
                        batch_logs = self._step(batch, is_train=True)
                        if batch_logs is not None:
                            epoch_train_logs.append(batch_logs)
                    except Exception as e:
                        pass

                    should_log = (self.global_step % log_every_n_steps == 0) or (self.global_step == 1)
                    should_visualize = step_based_visualization and (
                        (self.global_step % visualize_every_n_steps == 0) or (self.global_step == 1)
                    )

                    if should_log:
                        self._handle_logging(log_every_n_steps, epoch_train_logs, batch_idx)
                    
                    if should_visualize:
                        self._log_visualizations(step=self.global_step)

            # Final test evaluation
            try:
                # Standard test evaluation
                test_metrics = self._run_test_evaluation()
                
                # Collapse mode evaluation if enabled
                collapse_metrics = self._run_collapse_mode_evaluation()
                
                # Combine all metrics
                final_log_data = {}
                final_log_data.update(test_metrics)
                final_log_data.update(collapse_metrics)
                final_log_data["phase"] = "final_test_evaluation"
                final_log_data["step"] = self.global_step
                final_log_data["epoch"] = self.current_epoch
                
                if self.collapse_mode:
                    final_log_data["collapse_mode_enabled"] = True
                    final_log_data["collapse_modes_tested"] = len(self.collapse_modes)
                
                try:
                    wandb.log(final_log_data, step=self.global_step)
                except Exception as e:
                    pass
                    
            except Exception as e:
                pass
                        
        except Exception as e:
            pass

    def _handle_logging(self, log_every_n_steps, epoch_train_logs, batch_idx):
        """Handle logging during training steps"""
        try:
            val_metrics = self._run_validation()
            
            recent_train_logs = epoch_train_logs[-min(log_every_n_steps, len(epoch_train_logs)):]
            train_metrics = self._agg(recent_train_logs)
            
            log_data = self._prepare_log_data(train_metrics, val_metrics, batch_idx)
            
            try:
                wandb.log(log_data, step=self.global_step)
            except Exception as e:
                pass
            
            # Enhanced console output showing base and actual learning rates
            train_loss = train_metrics.get('train/loss/main', 0.0)
            val_loss = val_metrics.get('val/loss/main', 0.0)
            
            # Get base scheduler LR
            base_scheduler_lr = "N/A"
            if self.sched and hasattr(self.sched, 'get_last_lr'):
                try:
                    scheduler_lrs = self.sched.get_last_lr()
                    if scheduler_lrs:
                        base_scheduler_lr = f"{scheduler_lrs[0]:.2e}"
                except:
                    pass
            
            # Get a few actual LRs from parameter groups
            actual_lrs = []
            for i, group in enumerate(self.opt.param_groups[:3]):  # Show first 3
                group_name = group.get('group_name', f'g{i}')
                lr = group['lr']
                actual_lrs.append(f"{group_name}:{lr:.2e}")
            
            lr_display = " | ".join(actual_lrs)
            
            # Add gradient/update info if available
            grad_info = ""
            if self._should_log_gradients():
                grad_norm = train_metrics.get('gradients/global_norm', 0.0)
                update_norm = train_metrics.get('updates/global_norm', 0.0)
                if grad_norm > 0 and update_norm > 0:
                    grad_info = f" | Grad: {grad_norm:.2e} | Update: {update_norm:.2e}"
                elif grad_norm > 0:
                    grad_info = f" | Grad: {grad_norm:.2e}"
            
            # Add collapse mode indicator
            collapse_info = " | Collapse: ON" if self.collapse_mode else ""
            
            print(f"Step {self.global_step} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | Base LR: {base_scheduler_lr} | Actual: {lr_display}{grad_info}{collapse_info}")
        
        except Exception as e:
            pass

    def _prepare_log_data(self, train_metrics, val_metrics, batch_idx):
        """Prepare logging data dictionary"""
        log_data = {}
        log_data.update(train_metrics)
        log_data.update(val_metrics)
        
        for i, weight in enumerate(self.current_weights):
            if weight > 0:
                log_data[f'hyperparameter/aux_{i}_weight'] = weight
        
        self._log_learning_rates(log_data)
        
        log_data["epoch"] = self.current_epoch
        log_data["step"] = self.global_step
        log_data["batch"] = batch_idx
        
        if self.collapse_mode:
            log_data["collapse_mode_active"] = True
        
        return log_data

    def save(self, path: str):
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            self.model.save_model(path) 
        except Exception as e:
            pass