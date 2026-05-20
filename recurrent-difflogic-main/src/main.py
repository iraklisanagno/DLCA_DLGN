import argparse
import json
import os
import torch
import inspect

# Registry imports
from custom_datasets import DATASET_REGISTRY
from custom_tokenizers import TOKENIZER_REGISTRY
from metrics import METRIC_REGISTRY
from models import MODEL_REGISTRY, MODEL_SYNC_INFO
from visualization import VISUALISATION_REGISTRY
from trainer import Trainer


def parse_args():
    """Handle configuration management with CLI overrides"""
    parser = argparse.ArgumentParser(description="Training Pipeline")
    parser.add_argument("--config", type=str, help="Base config file path")
    parser.add_argument("--set", nargs='*', 
                      help="Config overrides (key=value)")
    return parser.parse_args()

def parse_value(value):
    """Convert string values to proper types"""
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value

def update_config(config, args):
    """Merge CLI overrides into config"""
    if args.set:
        for override in args.set:
            key_path, value = override.split('=', 1)
            keys = key_path.split('.')
            current = config
            for key in keys[:-1]:
                current = current.setdefault(key, {})
            current[keys[-1]] = parse_value(value)
    return config

def main():
    # Load and process configuration
    args = parse_args()
    config = {}
    if args.config:
        with open(args.config) as f:
            config = json.load(f)
    config = update_config(config, args)
    

    # Validate required parameters
    required_sections = ['dataset', 'model', 'training', "tokenizer"]
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required config section: {section}")

    # Load dataset
    try:
        dataset_loader = DATASET_REGISTRY[config['dataset']['name']]
    except KeyError:
        raise ValueError(f"Dataset {config['dataset']['name']} not registered")

    dataset = dataset_loader(**config['dataset'].get('params', {}))
    print(f"Loaded dataset: {config['dataset']['name']}")

    # Initialize tokenizer
    try:
        tokenizer = TOKENIZER_REGISTRY[config['tokenizer']['name']]
    except KeyError:
        raise ValueError(f"Tokenizer {config['tokenizer']['name']} not registered")

    tokenized_data = tokenizer(
        dataset,
        **config['tokenizer'].get('params', {})
    )
    print(f"Tokenized data with {config['tokenizer']['name']}")

    # Initialize model
    try:
        model_class = MODEL_REGISTRY[config['model']['name']]
        model_name = config['model']['name']
    except KeyError:
        raise ValueError(f"Model {config['model']['name']} not registered")

    # Get synchronization info for the model
    is_synced = MODEL_SYNC_INFO.get(model_name, True)  # Default to synced if not specified
    print(f"Model {model_name} is {'synced' if is_synced else 'unsynced'}")

    model_params = inspect.signature(model_class.__init__).parameters
    allowed_keys = set(model_params.keys()) - {'self'}

    # Combine config and tokenized interface, but filter tokenized keys
    combined_params = config['model'].get('params', {}).copy()
    combined_params.update({
        k: v for k, v in tokenized_data['model_interface'].items() if k in allowed_keys
    })

    model = model_class(**combined_params)
    print(f"Initialized model: {config['model']['name']}")

    # Configure metrics
    metrics_config = config.get('metrics', ['accuracy', 'perplexity', "precision_recall_f1"])
    metric_functions = {
        name: METRIC_REGISTRY[name] for name in metrics_config
        if name in METRIC_REGISTRY
    }
    print(f"Configured metrics: {', '.join(metric_functions.keys())}")

    # Initialize visualizations
    media_dict = config.get('visualization', [])
    visualizations = {}

    for vis in media_dict:
        try:
            visualizations[vis["type"]] = (VISUALISATION_REGISTRY[vis["type"]], vis)
        except KeyError:
            raise ValueError(
                f"Visualization media '{vis}' not registered in VISUALISATION_REGISTRY"
        )
    print(f"Initialized visualizations: {', '.join(visualizations.keys())}")
        
    # Initialize trainer with sync info
    trainer = Trainer(
        model=model,
        tokenized_data=tokenized_data,
        metrics=metric_functions,
        config=config,
        visualizations=visualizations,
        is_synced=is_synced  # Pass the sync information
    )

    # Run training
    trainer.train(
        epochs=config['training']['epochs'],
    )

    # Save model
    if 'save_path' in config:
        trainer.save(config['save_path'])

if __name__ == "__main__":
    main()