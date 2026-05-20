#!/usr/bin/env python

"""
sweep_runner.py

Simplified sweep runner for Weights & Biases hyperparameter search.

This script reads a sweep definition JSON (which includes:
  - path to the base model config JSON
  - W&B sweep settings (method, metric, parameters, etc.)
), and either creates a new sweep or joins an existing one.

The sweep definition JSON should include::
  {
    "base_config": "/path/to/configs/model_config.json",
    "count_per_agent": 10,
    "runs_per_config": 3,
    "wandb": {
      "project": "my-project",
      "method": "bayes",
      "metric": {"name": "validation_loss", "goal": "minimize"},
      "parameters": {
        "model.params.embedding_dim": {"distribution": "categorical", "values": [128,256,512]},
        "training.optimizer.params.lr": {"distribution": "log_uniform_values", "min": 1e-5, "max": 1e-3}
      }
    }
  }

Usage:
  # Create a new sweep
  python sweep_runner.py --sweep_def path/to/sweep_definition.json

  # Join an existing sweep
  python sweep_runner.py --sweep_def path/to/sweep_definition.json --sweep_id your_sweep_id
"""

import os
import sys
import json
import argparse
import tempfile
import copy
import wandb
from datetime import datetime

# Adjust the Python path to find modules in the Singularity environment
sys.path.insert(0, '/mnt/src')  # Add the mounted directory to Python path

# Import main function after adjusting path
print("Current directory:", os.getcwd())
print("Python path:", sys.path)

# Import main function after adjusting path
from main import main as run_main

# Redirect W&B file storage to writable mounts
os.environ['WANDB_DIR'] = '/mnt/wandb_logs'
os.environ['WANDB_CONFIG_DIR'] = '/mnt/wandb_config'


def load_sweep_def(path):
    with open(path, 'r') as f:
        return json.load(f)


def load_config(path):
    with open(path, 'r') as f:
        return json.load(f)


def update_nested_dict(d, key_path, value):
    """
    Update a nested dictionary using a key path like 'model.params.embedding_dim'.
    Supports array indexing with 'array.0.property' syntax.
    """
    keys = key_path.split('.')
    current = d
    
    # Navigate to the parent of the leaf node
    for i, key in enumerate(keys[:-1]):
        # Handle array indices
        if isinstance(current, list):
            try:
                idx = int(key)
                # Extend the list if needed
                while len(current) <= idx:
                    current.append({})
                current = current[idx]
                continue
            except ValueError:
                raise TypeError(f"Cannot use string key '{key}' on a list in path '{key_path}'")
        
        # Handle dict keys
        if isinstance(current, dict):
            if key not in current:
                current[key] = {}
            current = current[key]
        else:
            # We're trying to use a string key on something that's not a dict
            raise TypeError(f"Cannot use key '{key}' on a {type(current)} in path '{key_path}'")
    
    # Set the final value
    last_key = keys[-1]
    if isinstance(current, list):
        try:
            idx = int(last_key)
            # Extend the list if needed
            while len(current) <= idx:
                current.append(None)
            current[idx] = value
        except ValueError:
            raise TypeError(f"Cannot use string key '{last_key}' on a list in path '{key_path}'")
    elif isinstance(current, dict):
        current[last_key] = value
    else:
        raise TypeError(f"Cannot set key '{last_key}' on a {type(current)} in path '{key_path}'")
    
    return d


def get_nested_value(d, key_path, default=None):
    """
    Get a value from a nested dictionary using a key path like 'model.params.embedding_dim'.
    Supports array indexing with 'array.0.property' syntax.
    Returns default if the path doesn't exist.
    """
    keys = key_path.split('.')
    current = d
    
    for i, key in enumerate(keys):
        if current is None:
            return default
            
        # Handle array indexing
        if isinstance(current, list):
            try:
                # Try to convert key to integer index
                idx = int(key)
                if 0 <= idx < len(current):
                    current = current[idx]
                    continue
                else:
                    return default  # Index out of bounds
            except ValueError:
                # Not an integer, try to find a matching property across all items
                matching_found = False
                for item in current:
                    if isinstance(item, dict) and key in item:
                        current = item[key]
                        matching_found = True
                        break
                if not matching_found:
                    return default
        elif isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return default
            
    return current


def get_unique_sweep_id_filepath(sweep_def_path):
    """Create a unique sweep ID file path based on SLURM job ID and sweep definition"""
    # Extract the base filename without extension
    sweep_name = os.path.basename(sweep_def_path).split('.')[0]
    
    # Get SLURM job ID - this is shared by all tasks in the same array
    slurm_job_id = os.environ.get('SLURM_ARRAY_JOB_ID') or os.environ.get('SLURM_JOB_ID', 'unknown')
    
    # Create unique filename
    unique_filename = f"sweep_id_{slurm_job_id}_{sweep_name}.txt"
    
    return os.path.join('/mnt/sweep/tmp', unique_filename)


def create_temp_config(base_config_path, overrides):
    """
    Creates a temporary config file with the overridden parameters.
    
    Args:
        base_config_path: Path to the base config JSON file
        overrides: Dictionary of parameter overrides (key paths and values)
        
    Returns:
        Path to the created temporary config file
    """
    # Load the base config
    base_config = load_config(base_config_path)
    
    # Create a deep copy to avoid modifying the original
    config = copy.deepcopy(base_config)
    
    # Apply overrides
    for key, value in overrides.items():
        if not key.startswith('_'):  # Skip internal keys
            config = update_nested_dict(config, key, value)
    
    # Create temp file and save the modified config
    fd, temp_path = tempfile.mkstemp(suffix='.json', prefix='temp_config_')
    with os.fdopen(fd, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Created temp config at {temp_path} with overrides: {overrides}")
    return temp_path


def build_run_name(base_config_data, config_dict, repeat_index=0):
    """
    Build a descriptive run name for W&B.
    
    Args:
        base_config_data: Base configuration dictionary
        config_dict: Dictionary of hyperparameters from the sweep
        repeat_index: Index for this repeat run
        
    Returns:
        A descriptive run name
    """
    timestamp = datetime.now().strftime("%m%d_%H%M")
    
    # Extract model name from base config
    model_name = get_nested_value(base_config_data, 'model.name', 'Model')
    
    # Check if model name is in the sweep overrides
    if 'model.name' in config_dict:
        model_name = config_dict['model.name']
    
    # Get embedding dimension
    embedding_dim = get_nested_value(base_config_data, 'model.params.embedding_dim', '?')
    if 'model.params.embedding_dim' in config_dict:
        embedding_dim = config_dict['model.params.embedding_dim']
    
    # Get learning rate
    lr = get_nested_value(base_config_data, 'training.optimizer.params.lr', '?')
    if 'training.optimizer.params.lr' in config_dict:
        lr = config_dict['training.optimizer.params.lr']
    
    # Format the learning rate in scientific notation
    try:
        lr_str = f"{float(lr):.0e}"
    except (ValueError, TypeError):
        lr_str = str(lr)
    
    # Get batch size if it exists
    batch_size = get_nested_value(base_config_data, 'tokenizer.params.batch_size', None)
    if 'tokenizer.params.batch_size' in config_dict:
        batch_size = config_dict['tokenizer.params.batch_size']
    batch_str = f"_b{batch_size}" if batch_size else ""
    
    # Include repeat index if there are multiple repeats
    repeat_str = f"_r{repeat_index}" if repeat_index > 0 else ""
    
    # Build the final name
    run_name = f"{model_name}_e{embedding_dim}{batch_str}_lr{lr_str}{repeat_str}_{timestamp}"
    
    return run_name


class Args:
    """Simple class to mimic argparse.Namespace for main.py."""
    def __init__(self, config_path, set_args=None):
        self.config = config_path
        self.set = set_args if set_args is not None else []


def run_training(base_config, config_dict, repeat_index=0, project=None):
    """
    Execute training directly with the given configuration.
    
    Does NOT catch exceptions from run_main() - allows errors to propagate 
    and terminate the process for proper SLURM task handling
    """
    # Load the base config to check for seed and validate parameters
    base_config_data = load_config(base_config)
    
    # Add the repeat index to the config
    overrides = config_dict.copy()
    overrides['_repeat'] = repeat_index
    
    # Determine seed to use for this repeat
    base_seed = 42  # Default seed if not specified
    
    # Check if seed is specified in the base config
    if 'model' in base_config_data and 'params' in base_config_data['model'] and 'seed' in base_config_data['model']['params']:
        base_seed = base_config_data['model']['params']['seed']
    
    # Check if seed was overridden in the sweep
    if 'model.params.seed' in overrides:
        base_seed = overrides['model.params.seed']
    
    # Set a new seed for this repeat (base_seed + repeat_index)
    overrides['model.params.seed'] = base_seed + repeat_index
    print(f"Using seed {overrides['model.params.seed']} for repeat {repeat_index}")
    
    # Create a temporary config file with the overrides
    temp_config_path = create_temp_config(base_config, overrides)
    
    try:
        # Create args object for main function
        args = Args(temp_config_path)
        
        # Store original sys.argv
        original_argv = sys.argv
        
        # Set sys.argv to what main.py would expect
        sys.argv = [
            'main.py',
            '--config', temp_config_path
        ]
        
        # Build a descriptive run name
        run_name = build_run_name(base_config_data, overrides, repeat_index)
        
        print(f"Running main.py with config: {temp_config_path}")
        
        # First, make sure any existing wandb run is finished
        if wandb.run is not None:
            wandb.finish()
            
        # Clear any existing wandb environment variables that might affect run creation
        for env_var in list(os.environ.keys()):
            if env_var.startswith('WANDB_') and env_var != 'WANDB_DIR' and env_var != 'WANDB_CONFIG_DIR':
                del os.environ[env_var]
                
        # Set environment variables to guide the new run creation in Trainer
        os.environ['WANDB_PROJECT'] = project
        os.environ['WANDB_NAME'] = run_name
        
        # Run the training code - DO NOT catch exceptions
        # Let errors propagate up to terminate the process
        run_main()
        
        # If we get here, the run completed successfully
        if wandb.run is not None:
            wandb.finish()
            
        # Restore original sys.argv
        sys.argv = original_argv
    finally:
        # Clean up the temporary config file
        if os.path.exists(temp_config_path):
            os.remove(temp_config_path)
            print(f"Removed temp config file: {temp_config_path}")


def agent_function(base_config, runs_per_config):
    """
    Function called by wandb.agent for each configuration.
    
    Does NOT catch exceptions from run_training, allowing failed runs
    to terminate the process for proper SLURM task handling.
    """
    # Initialize wandb to get the config for this sweep run
    with wandb.init() as run:
        # Extract the config parameters and project
        config_dict = {k: v for k, v in run.config.items()}
        project = run.project
        print(f"Agent received config: {config_dict}")
        
        # Finish this run since we're only using it to get the config
        run.finish()
        
        # Run the training code - DO NOT catch exceptions
        for i in range(runs_per_config):
            print(f"Running repeat {i+1}/{runs_per_config} of current config")
            run_training(base_config, config_dict, repeat_index=i, project=project)
            # If run_training raises an exception, control will never return here
            # and the process will terminate with a non-zero exit code


def prepare_parameters_for_sweep(sweep_def, base_config_path):
    """
    Prepare and validate parameters for the sweep.
    
    Args:
        sweep_def: Sweep definition dictionary
        base_config_path: Path to the base config file
        
    Returns:
        Updated sweep definition with validated parameters
    """
    base_config_data = load_config(base_config_path)
    
    # Validate that parameters exist in the base config
    if 'wandb' in sweep_def and 'parameters' in sweep_def['wandb']:
        valid_parameters = {}
        for param_name, param_config in sweep_def['wandb']['parameters'].items():
            value = get_nested_value(base_config_data, param_name)
            if value is not None:
                valid_parameters[param_name] = param_config
                print(f"✓ Parameter {param_name} found in base config")
            else:
                # Instead of filtering out, keep the parameter and just warn
                print(f"⚠ Parameter {param_name} not found in base config - keeping it anyway")
                valid_parameters[param_name] = param_config
        
        # Ensure we always have at least one parameter to avoid W&B error
        if not valid_parameters:
            print("No valid parameters found. Adding a dummy parameter to avoid W&B error.")
            # Add a safe parameter that definitely exists
            valid_parameters['model.params.seed'] = {
                "values": [45, 46]  # Just vary the seed slightly
            }
        
        sweep_def['wandb']['parameters'] = valid_parameters
    
    return sweep_def


def main():
    parser = argparse.ArgumentParser(description="W&B sweep runner")
    parser.add_argument('--sweep_def', type=str, required=True,
                        help='Path to sweep definition JSON')
    parser.add_argument('--sweep_id', type=str, default=None,
                        help='Existing sweep ID to join (if not provided, creates a new sweep)')
    parser.add_argument('--agent_count', type=int, default=None,
                        help='Number of runs for this agent to process (default: use count_per_agent from sweep_def)')
    args = parser.parse_args()

    sweep_def = load_sweep_def(args.sweep_def)
    base_config = sweep_def['base_config']
    runs_per_config = int(sweep_def.get('runs_per_config', 1))
    
    # Get the agent count - use command line arg or default to count in sweep def
    agent_count = args.agent_count
    if agent_count is None or agent_count <= 0:
        agent_count = int(sweep_def.get('count_per_agent', 5))
    
    # Prepare and validate parameters for the sweep
    sweep_def = prepare_parameters_for_sweep(sweep_def, base_config)

    # Set up the sweep
    project = sweep_def.get('wandb', {}).get('project', 'recurrent-difflogic')
    
    # Either join an existing sweep or create a new one
    if args.sweep_id:
        sweep_id = args.sweep_id
        print(f"Joining existing sweep: {sweep_id}")
    else:
        # Set up the sweep configuration
        wandb_cfg = sweep_def.get('wandb', {})
        sweep_config = {
            'method': wandb_cfg.get('method', 'bayes'),
            'metric': wandb_cfg.get('metric'),
            'parameters': wandb_cfg.get('parameters')
        }
        
        # Add early termination if specified
        if 'early_terminate' in wandb_cfg:
            sweep_config['early_terminate'] = wandb_cfg['early_terminate']
        
        # Create the sweep
        sweep_id = wandb.sweep(sweep_config, project=project)
        print(f"Created new sweep: {sweep_id}")

    # Display sweep information
    print(f"Sweep URL: https://wandb.ai/{os.environ.get('WANDB_ENTITY', 'sbuehrer-eth-z-rich')}/{project}/sweeps/{sweep_id}")
    print(f"Agent will process {agent_count} configurations with {runs_per_config} repeats each (different seeds)")
    print(f"SLURM Array Task ID: {os.environ.get('SLURM_ARRAY_TASK_ID', 'not in SLURM')}")
    print(f"Hostname: {os.uname().nodename}")
    
    # Use the same unique sweep ID file path generation as in sweep.py
    if not args.sweep_id:
        sweep_id_file = get_unique_sweep_id_filepath(args.sweep_def)
        
        # Ensure the directory exists
        os.makedirs(os.path.dirname(sweep_id_file), exist_ok=True)
        
        # Write sweep ID to the file
        with open(sweep_id_file, 'w') as f:
            f.write(sweep_id)
        print(f"Sweep ID saved to {sweep_id_file}")
    
    try:
        # Start the sweep agent
        wandb.agent(
            sweep_id,
            function=lambda: agent_function(base_config, runs_per_config),
            count=agent_count,  # Number of runs for this agent to process
            project=project
        )
    except Exception as e:
        # In case of "sweep is not running" or similar errors, provide informative message
        if "is not running" in str(e):
            print(f"Error: The sweep {sweep_id} appears to be completed or not running.")
            print("This can happen if multiple agents try to join the same sweep after it's completed.")
            print("Check the sweep status in the W&B UI and consider creating a new sweep if needed.")
        else:
            print(f"Error running sweep agent: {str(e)}")
        
        # Non-zero exit code indicates error
        sys.exit(1)


if __name__ == "__main__":
    main()