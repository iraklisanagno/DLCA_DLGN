import torch
import torch.nn as nn
import torch.nn.functional as F
from models.base_model import BaseModel  

class UnsyncedRNN(BaseModel):
    def __init__(self, num_input_tokens, num_classes, embedding_dim, hidden_dim, 
                 num_layers=1, dropout=0.0, seed=None):
        super(UnsyncedRNN, self).__init__()
        
        # Set seed for reproducibility if provided
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        
        # Store configuration parameters
        self.num_input_tokens = num_input_tokens
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.seed = seed
        
        # Shared embedding layer
        self.embedding = nn.Embedding(num_input_tokens, embedding_dim, padding_idx=0)
        
        # Encoder
        self.encoder_rnn = nn.RNN(
            input_size=embedding_dim, 
            hidden_size=hidden_dim, 
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Decoder
        self.decoder_rnn = nn.RNN(
            input_size=embedding_dim, 
            hidden_size=hidden_dim, 
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Output projection layer
        self.output_layer = nn.Linear(hidden_dim, num_classes)
        
        # Dropout layer
        self.dropout_layer = nn.Dropout(dropout)

    def encode(self, src):
        """
        Encode source sequence
        src: [batch_size, src_seq_len] - token indices
        Returns: encoder_outputs, final_hidden_state
        """
        # Embed source tokens
        src_embedded = self.embedding(src)  # [batch_size, src_seq_len, embedding_dim]
        
        # Apply dropout
        src_embedded = self.dropout_layer(src_embedded)
        
        # Pass through encoder RNN
        encoder_outputs, encoder_hidden = self.encoder_rnn(src_embedded)
        # encoder_outputs: [batch_size, src_seq_len, hidden_dim]
        # encoder_hidden: [num_layers, batch_size, hidden_dim]
        
        return encoder_outputs, encoder_hidden

    def decode(self, tgt, encoder_hidden):
        """
        Decode target sequence
        tgt: [batch_size, tgt_seq_len] - token indices
        encoder_hidden: [num_layers, batch_size, hidden_dim] - initial hidden state from encoder
        Returns: decoder_outputs
        """
        # Embed target tokens
        tgt_embedded = self.embedding(tgt)  # [batch_size, tgt_seq_len, embedding_dim]
        
        # Apply dropout
        tgt_embedded = self.dropout_layer(tgt_embedded)
        
        # Pass through decoder RNN with encoder's final hidden state as initial state
        decoder_outputs, _ = self.decoder_rnn(tgt_embedded, encoder_hidden)
        # decoder_outputs: [batch_size, tgt_seq_len, hidden_dim]
        
        return decoder_outputs

    def forward(self, src, tgt=None):
        """
        Forward pass for encoder-decoder RNN
        src: [batch_size, src_seq_len] - source token indices
        tgt: [batch_size, tgt_seq_len] - target token indices (for training)
        Returns: logits [batch_size, tgt_seq_len, num_classes]
        """
        # Encode source sequence
        encoder_outputs, encoder_hidden = self.encode(src)
        
        if tgt is not None:
            # Training mode: use teacher forcing
            # During training, we use the actual target sequence (shifted)
            # We want to predict the next token, so we use tgt[:-1] as input
            # and compare predictions against tgt[1:] during loss calculation
            decoder_outputs = self.decode(tgt, encoder_hidden)
        else:
            # Inference mode: generate autoregressively
            # This would be implemented for actual inference
            # For now, we'll assume tgt is always provided during training
            raise NotImplementedError("Inference mode not implemented yet")
        
        # Project to vocabulary space
        logits = self.output_layer(decoder_outputs)  # [batch_size, tgt_seq_len, num_classes]
        
        return logits

    def generate(self, src, max_length=50, start_token=1, end_token=2):
        """
        Generate sequence autoregressively (for inference)
        src: [batch_size, src_seq_len] - source token indices
        max_length: maximum length of generated sequence
        start_token: token to start generation (usually <BOS>)
        end_token: token to end generation (usually <EOS>)
        Returns: generated sequences [batch_size, generated_seq_len]
        """
        self.eval()
        with torch.no_grad():
            batch_size = src.size(0)
            device = src.device
            
            # Encode source
            encoder_outputs, encoder_hidden = self.encode(src)
            
            # Initialize decoder input with start token
            decoder_input = torch.full((batch_size, 1), start_token, 
                                     dtype=torch.long, device=device)
            decoder_hidden = encoder_hidden
            
            # Store generated tokens
            generated = [decoder_input]
            
            for _ in range(max_length - 1):
                # Embed current input
                embedded = self.embedding(decoder_input)
                embedded = self.dropout_layer(embedded)
                
                # Get next hidden state and output
                decoder_output, decoder_hidden = self.decoder_rnn(embedded, decoder_hidden)
                
                # Project to vocabulary
                logits = self.output_layer(decoder_output)  # [batch_size, 1, num_classes]
                
                # Get next token (greedy decoding)
                next_token = torch.argmax(logits, dim=-1)  # [batch_size, 1]
                generated.append(next_token)
                
                # Use this token as next input
                decoder_input = next_token
                
                # Check if all sequences have generated end token
                if (next_token == end_token).all():
                    break
            
            # Concatenate all generated tokens
            generated_sequence = torch.cat(generated, dim=1)  # [batch_size, seq_len]
            
        return generated_sequence

    def save_model(self, path):
        """Saves model to specified path"""
        torch.save({
            'model_state': self.state_dict(),
            'config': {
                'num_input_tokens': self.num_input_tokens,
                'num_classes': self.num_classes,
                'embedding_dim': self.embedding_dim,
                'hidden_dim': self.hidden_dim,
                'num_layers': self.num_layers,
                'dropout': self.dropout,
                'seed': self.seed
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
            hidden_dim=config['hidden_dim'],
            num_layers=config.get('num_layers', 1),
            dropout=config.get('dropout', 0.0),
            seed=config.get('seed', None)
        ).to(device)
        
        model.load_state_dict(checkpoint['model_state'])
        return model

    def print_trainable_params(self):
        """Prints number of trainable parameters"""
        total = 0
        for name, param in self.named_parameters():
            if param.requires_grad:
                num_params = param.numel()
                print(f"{name:30} {num_params:>10,}")
                total += num_params
        print(f"\nTotal trainable parameters: {total:,}")
        return total

    def get_predictions(self, outputs):
        """
        Extract prediction logits from model outputs for metrics calculation
        This method is called by the trainer for metrics computation
        """
        if isinstance(outputs, tuple):
            return outputs[0]  # Return logits if outputs is a tuple
        return outputs  # Return logits directly