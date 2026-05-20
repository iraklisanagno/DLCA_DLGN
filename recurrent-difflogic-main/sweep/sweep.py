#!/usr/bin/env python

"""
sweep.py

Coordinator script for running W&B sweeps across multiple SLURM array tasks.
The first task (ARRAY_TASK_ID=0) creates the sweep, and other tasks join it.

Usage:
  Called by SLURM job array script
"""

import os
import sys
import time
import subprocess
import argparse
import json
import os.path

def load_json(file_path):
    """Load a JSON file"""
    with open(file_path, 'r') as f:
        return json.load(f)

def get_unique_sweep_id_filepath(sweep_def_path):
    """Create a unique sweep ID file path based on SLURM job ID and sweep definition"""
    # Extract the base filename without extension
    sweep_name = os.path.basename(sweep_def_path).split('.')[0]
    
    # Get SLURM job ID - this is shared by all tasks in the same array
    slurm_job_id = os.environ.get('SLURM_ARRAY_JOB_ID') or os.environ.get('SLURM_JOB_ID', 'unknown')
    
    # Create unique filename
    unique_filename = f"sweep_id_{slurm_job_id}_{sweep_name}.txt"
    
    return os.path.join('/mnt/sweep/tmp', unique_filename)

def determine_sweep_runner(sweep_def_path):
    """Determine which sweep runner to use based on the sweep definition"""
    try:
        sweep_def = load_json(sweep_def_path)
        
        # Check if this is an architecture optimization sweep
        if 'wandb' in sweep_def and 'parameters' in sweep_def['wandb']:
            parameters = sweep_def['wandb']['parameters']
            
            # Look for ANY architecture parameters (depth, width_factor, or scale)
            arch_params = [
                'model.params.k_layers_depth',
                'model.params.l_layers_depth', 
                'model.params.m_layers_depth',
                'model.params.n_layers_depth',
                'model.params.p_layers_depth',
                'model.params.k_layers_width_factor',
                'model.params.l_layers_width_factor',
                'model.params.m_layers_width_factor',
                'model.params.n_layers_width_factor',
                'model.params.p_layers_width_factor',
                'model.params.architecture_scale'
            ]
            
            # If any architecture parameters are present, use architecture runner
            for param in arch_params:
                if param in parameters:
                    print(f"Detected architecture optimization sweep (found parameter: {param})")
                    return '/mnt/sweep/sweep_runner_architecture.py'
        
        # Default to regular sweep runner
        print("Using regular sweep runner")
        return '/mnt/sweep/sweep_runner.py'
        
    except Exception as e:
        print(f"Error determining sweep runner: {e}")
        print("Defaulting to regular sweep runner")
        return '/mnt/sweep/sweep_runner.py'

def main():
    parser = argparse.ArgumentParser(description="W&B sweep coordinator for SLURM arrays")
    parser.add_argument('--sweep_def', type=str, 
                        default='/mnt/sweep/recurrent_difflogic/bayesian_tuning.json',
                        help='Path to sweep definition JSON')
    parser.add_argument('--agent_count', type=int, default=None,
                        help='Override number of runs for each agent (default: use count_per_agent from sweep_def)')
    parser.add_argument('--runner', type=str, default=None,
                        help='Override sweep runner script path (auto-detected by default)')
    args = parser.parse_args()

    print(f"🔧 Starting sweep coordinator")
    print(f"   Config: {args.sweep_def}")
    print(f"   Agent count: {args.agent_count}")

    # Load the sweep definition
    try:
        sweep_def = load_json(args.sweep_def)
        print(f"✅ Loaded sweep definition")
    except Exception as e:
        print(f"❌ Error loading sweep definition: {e}")
        sys.exit(1)
    
    # Determine agent count (priority: command line > sweep_def > default)
    if args.agent_count is not None:
        agent_count = args.agent_count
    else:
        agent_count = int(sweep_def.get('count_per_agent', 5))
    
    print(f"🎯 Using agent_count: {agent_count}")

    # Determine which sweep runner to use
    if args.runner:
        sweep_runner = args.runner
        print(f"🔧 Using explicitly specified sweep runner: {sweep_runner}")
    else:
        sweep_runner = determine_sweep_runner(args.sweep_def)
    
    # Verify the sweep runner exists
    if not os.path.exists(sweep_runner):
        print(f"❌ Error: Sweep runner not found at {sweep_runner}")
        
        # Try to find available runners
        print("🔍 Looking for available runners...")
        for runner_name in ['sweep_runner.py', 'sweep_runner_architecture.py']:
            runner_path = f'/mnt/sweep/{runner_name}'
            if os.path.exists(runner_path):
                print(f"   Found: {runner_path}")
        
        sys.exit(1)

    # Get SLURM array task ID
    array_task_id = int(os.environ.get('SLURM_ARRAY_TASK_ID', '0'))
    print(f"🚀 Starting sweep agent for SLURM array task {array_task_id}")
    print(f"   Using sweep runner: {sweep_runner}")
    
    # Create a unique sweep ID file path for this job array
    sweep_id_file = get_unique_sweep_id_filepath(args.sweep_def)
    print(f"📁 Using sweep ID file: {sweep_id_file}")
    
    # Ensure the directory exists
    os.makedirs(os.path.dirname(sweep_id_file), exist_ok=True)
    
    if array_task_id == 0:
        # First task creates a new sweep
        cmd = [
            'python', sweep_runner,
            '--sweep_def', args.sweep_def,
            '--agent_count', str(agent_count)
        ]
        print(f"🆕 Creating new sweep with command: {' '.join(cmd)}")
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"❌ Error creating sweep: {e}")
            sys.exit(1)
    else:
        # Other tasks wait for the sweep ID file to be created
        max_wait = 300  # Maximum wait time in seconds
        wait_time = 0
        while not os.path.exists(sweep_id_file) and wait_time < max_wait:
            print(f"⏳ Waiting for sweep ID file at {sweep_id_file} (waited {wait_time}s)...")
            time.sleep(5)
            wait_time += 5
        
        if not os.path.exists(sweep_id_file):
            print(f"❌ Error: Sweep ID file not found at {sweep_id_file} after waiting {max_wait} seconds")
            sys.exit(1)
        
        # Read the sweep ID
        with open(sweep_id_file, 'r') as f:
            sweep_id = f.read().strip()
        
        if not sweep_id:
            print("❌ Error: Empty sweep ID")
            sys.exit(1)
        
        print(f"🔗 Joining existing sweep: {sweep_id}")
        
        # Join the existing sweep
        cmd = [
            'python', sweep_runner,
            '--sweep_def', args.sweep_def,
            '--sweep_id', sweep_id,
            '--agent_count', str(agent_count)
        ]
        print(f"🔗 Joining sweep with command: {' '.join(cmd)}")
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError:
            if array_task_id != 0 and '--sweep_id' in ' '.join(cmd):
                print("⚠️  Sweep appears completed. Creating a new sweep instead...")
                # Create a new sweep instead of joining
                new_cmd = [
                    'python', sweep_runner,
                    '--sweep_def', args.sweep_def,
                    '--agent_count', str(agent_count)
                ]
                print(f"🆕 Creating new sweep with command: {' '.join(new_cmd)}")
                subprocess.run(new_cmd, check=True)

if __name__ == "__main__":
    main()