import torch
import torch.nn as nn
import torch.nn.functional as F
from models.base_model import BaseModel  

class SyncedGRU(BaseModel):
    def __init__(self, num_input_tokens, num_classes, embedding_dim, hidden_dim, seed=None):
        super(SyncedGRU, self).__init__()
        # Set random seed for reproducibility
        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        # Store configuration parameters
        self.num_input_tokens = num_input_tokens
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        
        # Network layers
        self.embedding = nn.Embedding(num_input_tokens, embedding_dim, padding_idx=0)
        self.gru = nn.GRU(input_size=embedding_dim,
                         hidden_size=hidden_dim,
                         batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        """
        x: [batch_size, seq_len, vocab_size] (one-hot input)
        Returns: tuple (probabilities, logits)
        """
        embedded = self.embedding(x)
        output, _ = self.gru(embedded)
        logits = self.output_layer(output)
        return F.softmax(logits, dim=-1)

    def save_model(self, path):
        """Saves model to specified path"""
        torch.save({
            'model_state': self.state_dict(),
            'config': {
                'num_input_tokens': self.num_input_tokens,
                'num_classes': self.num_classes,
                'embedding_dim': self.embedding_dim,
                'hidden_dim': self.hidden_dim
            }
        }, path)

    @classmethod
    def load_model(cls, path, device='cpu'):
        """Loads model from specified path"""
        checkpoint = torch.load(path, map_location=device)
        config = checkpoint['config']
        
        model = cls(
            num_input_tokens=config['num_input_tokens'],
            num_classes=config['num_classes'],
            embedding_dim=config['embedding_dim'],
            hidden_dim=config['hidden_dim']
        ).to(device)
        
        model.load_state_dict(checkpoint['model_state'])
        return model

    def print_trainable_params(self):
        """Prints number of trainable parameters"""
        total = 0
        for name, param in self.named_parameters():
            if param.requires_grad:
                num_params = param.numel()
                print(f"{name:20} {num_params:>10,}")
                total += num_params
        print(f"\nTotal trainable parameters: {total:,}")
        return total