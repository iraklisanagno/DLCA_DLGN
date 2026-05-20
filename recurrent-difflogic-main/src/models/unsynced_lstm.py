import torch
import torch.nn as nn
import torch.nn.functional as F
from models.base_model import BaseModel  

class UnsyncedLSTM(BaseModel):
    def __init__(self, num_input_tokens, num_classes, embedding_dim, hidden_dim, 
                 num_layers=1, dropout=0.0, seed=None, padding_idx=0, 
                 bos_token_id=2, eos_token_id=3):
        super(UnsyncedLSTM, self).__init__()
        
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
        self.padding_idx = padding_idx
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        
        # Shared embedding layer
        self.embedding = nn.Embedding(num_input_tokens, embedding_dim, padding_idx=padding_idx)
        
        # Encoder LSTM
        self.encoder_lstm = nn.LSTM(
            input_size=embedding_dim, 
            hidden_size=hidden_dim, 
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Decoder LSTM
        self.decoder_lstm = nn.LSTM(
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
        Returns: encoder_outputs, (final_hidden_state, final_cell_state)
        """
        # Embed source tokens
        src_embedded = self.embedding(src)  # [batch_size, src_seq_len, embedding_dim]
        
        # Apply dropout
        src_embedded = self.dropout_layer(src_embedded)
        
        # Pass through encoder LSTM
        encoder_outputs, (encoder_hidden, encoder_cell) = self.encoder_lstm(src_embedded)
        # encoder_outputs: [batch_size, src_seq_len, hidden_dim]
        # encoder_hidden: [num_layers, batch_size, hidden_dim]
        # encoder_cell: [num_layers, batch_size, hidden_dim]
        
        return encoder_outputs, (encoder_hidden, encoder_cell)

    def decode(self, tgt, encoder_state):
        """
        Decode target sequence
        tgt: [batch_size, tgt_seq_len] - token indices
        encoder_state: tuple (hidden, cell) - initial state from encoder
        Returns: decoder_outputs
        """
        # Embed target tokens
        tgt_embedded = self.embedding(tgt)  # [batch_size, tgt_seq_len, embedding_dim]
        
        # Apply dropout
        tgt_embedded = self.dropout_layer(tgt_embedded)
        
        # Pass through decoder LSTM with encoder's final state as initial state
        decoder_outputs, _ = self.decoder_lstm(tgt_embedded, encoder_state)
        # decoder_outputs: [batch_size, tgt_seq_len, hidden_dim]
        
        return decoder_outputs

    def forward(self, src, tgt=None):
        """
        Forward pass for encoder-decoder LSTM
        src: [batch_size, src_seq_len] - source token indices
        tgt: [batch_size, tgt_seq_len] - target token indices (for training)
        Returns: logits [batch_size, tgt_seq_len-1, num_classes] for teacher forcing
        """
        # Encode source sequence
        encoder_outputs, encoder_state = self.encode(src)
        
        if tgt is not None:
            # Training mode: use teacher forcing
            # CRITICAL: We use tgt[:,:-1] as input and predict tgt[:,1:]
            # This implements the correct shifting for teacher forcing
            
            # Create decoder input by shifting target right (add BOS, remove last token)
            batch_size, tgt_len = tgt.shape
            device = tgt.device
            
            # For teacher forcing, we need to create the decoder input sequence
            # Assuming target has structure: [BOS, word1, word2, ..., wordN, EOS]
            # We use [BOS, word1, word2, ..., wordN] as input
            # And predict [word1, word2, ..., wordN, EOS]
            decoder_input = tgt[:, :-1]  # Remove last token (EOS) for input
            
            # Pass through decoder
            decoder_outputs = self.decode(decoder_input, encoder_state)
            
            # Project to vocabulary space
            logits = self.output_layer(decoder_outputs)
            # logits shape: [batch_size, tgt_seq_len-1, num_classes]
            
            return logits
        else:
            # Inference mode: generate autoregressively
            return self.generate_step_by_step(src, encoder_state)

    def generate_step_by_step(self, src, encoder_state, max_length=50):
        """
        Generate sequence step by step (used when tgt=None)
        """
        batch_size = src.size(0)
        device = src.device
        
        # Start with BOS token
        decoder_input = torch.full((batch_size, 1), self.bos_token_id, 
                                 dtype=torch.long, device=device)
        
        outputs = []
        hidden, cell = encoder_state
        
        for _ in range(max_length):
            # Get output for current step
            embedded = self.embedding(decoder_input)
            embedded = self.dropout_layer(embedded)
            decoder_out, (hidden, cell) = self.decoder_lstm(embedded, (hidden, cell))
            logits = self.output_layer(decoder_out)
            
            outputs.append(logits)
            
            # Get next token
            next_token = torch.argmax(logits, dim=-1)
            decoder_input = next_token
            
            # Check for EOS
            if (next_token == self.eos_token_id).all():
                break
        
        # Concatenate all outputs
        if outputs:
            return torch.cat(outputs, dim=1)
        else:
            # Return empty tensor if no outputs
            return torch.zeros(batch_size, 0, self.num_classes, device=device)

    def generate(self, src, max_length=50, start_token=None, end_token=None):
        """
        Generate sequence autoregressively (for inference)
        """
        if start_token is None:
            start_token = self.bos_token_id
        if end_token is None:
            end_token = self.eos_token_id
            
        self.eval()
        with torch.no_grad():
            batch_size = src.size(0)
            device = src.device
            
            # Encode source
            encoder_outputs, encoder_state = self.encode(src)
            
            # Initialize decoder input with start token
            decoder_input = torch.full((batch_size, 1), start_token, 
                                     dtype=torch.long, device=device)
            decoder_state = encoder_state
            
            # Store generated tokens
            generated = [decoder_input]
            
            for _ in range(max_length - 1):
                # Embed current input
                embedded = self.embedding(decoder_input)
                embedded = self.dropout_layer(embedded)
                
                # Get next hidden state and output
                decoder_output, decoder_state = self.decoder_lstm(embedded, decoder_state)
                
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
                'seed': self.seed,
                'padding_idx': self.padding_idx,
                'bos_token_id': self.bos_token_id,
                'eos_token_id': self.eos_token_id
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
            seed=config.get('seed', None),
            padding_idx=config.get('padding_idx', 0),
            bos_token_id=config.get('bos_token_id', 2),
            eos_token_id=config.get('eos_token_id', 3)
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
        """
        if isinstance(outputs, tuple):
            return outputs[0]
        return outputs