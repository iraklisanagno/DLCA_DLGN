import torch
import torch.nn as nn
import math
from models.base_model import BaseModel


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        # x: [batch, seq_len, d_model]
        return x + self.pe[:, :x.size(1)]


class Transformer(BaseModel):
    def __init__(
        self,
        num_input_tokens,
        num_classes,
        embedding_dim,
        hidden_dim,
        num_heads,
        num_layers,
        dropout=0.1,
        max_len=5000,
        seed=None,
    ):
        super().__init__()

        # reproducibility
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        self.num_input_tokens = num_input_tokens
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = dropout

        # embeddings + positional encodings
        self.src_embedding = nn.Embedding(num_input_tokens, embedding_dim, padding_idx=0)
        self.tgt_embedding = nn.Embedding(num_classes, embedding_dim, padding_idx=0)
        self.pos_encoder = PositionalEncoding(embedding_dim, max_len)
        self.pos_decoder = PositionalEncoding(embedding_dim, max_len)
        
        # Dropout applied to the sum of embeddings and positional encodings
        # This is explicitly mentioned in the paper but not in the default nn.Transformer
        self.emb_dropout = nn.Dropout(dropout)

        # Transformer
        # PyTorch's nn.Transformer implementation already handles dropout
        # in the correct places within each sublayer as specified in the paper:
        # 1. After self-attention before add & norm
        # 2. After feed-forward before add & norm
        self.transformer = nn.Transformer(
            d_model=embedding_dim,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=hidden_dim,
            dropout=dropout,  # This applies dropout to all appropriate sublayers
            batch_first=True,
        )

        # final output
        self.output_layer = nn.Linear(embedding_dim, num_classes)

    def _generate_square_subsequent_mask(self, sz, device):
        # Upper triangular mask for causal decoding
        return torch.triu(torch.full((sz, sz), float('-inf'), device=device), diagonal=1)

    def forward(self, src, tgt=None, src_key_padding_mask=None, tgt_key_padding_mask=None):
        """
        Args:
            src: LongTensor [batch, src_len]
            tgt: LongTensor [batch, tgt_len] or None. If None, uses src as decoder input.
            src_key_padding_mask: BoolTensor [batch, src_len]
            tgt_key_padding_mask: BoolTensor [batch, tgt_len]
        Returns:
            logits: FloatTensor [batch, tgt_len, num_classes]
        """
        # Use device from src if tgt is None
        device = src.device

        # Determine decoder input
        if tgt is None:
            tgt_input = src
        else:
            tgt_input = tgt

        # embed + positional
        src_emb = self.src_embedding(src) * math.sqrt(self.embedding_dim)
        src_emb = self.pos_encoder(src_emb)
        # Apply dropout to the sum as specified in the paper
        src_emb = self.emb_dropout(src_emb)
        
        tgt_emb = self.tgt_embedding(tgt_input) * math.sqrt(self.embedding_dim)
        tgt_emb = self.pos_decoder(tgt_emb)
        # Apply dropout to the sum as specified in the paper
        tgt_emb = self.emb_dropout(tgt_emb)

        # create causal mask
        tgt_seq_len = tgt_emb.size(1)
        tgt_mask = self._generate_square_subsequent_mask(tgt_seq_len, device)

        # encode
        memory = self.transformer.encoder(
            src_emb,
            src_key_padding_mask=src_key_padding_mask,
        )
        # decode
        output = self.transformer.decoder(
            tgt_emb,
            memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask,
        )

        # project to vocab
        logits = self.output_layer(output)
        return logits

    def save_model(self, path):
        torch.save({
            'model_state': self.state_dict(),
            'config': {
                'num_input_tokens': self.num_input_tokens,
                'num_classes': self.num_classes,
                'embedding_dim': self.embedding_dim,
                'hidden_dim': self.hidden_dim,
                'num_heads': self.num_heads,
                'num_layers': self.num_layers,
                'dropout': self.dropout,
            }
        }, path)

    @classmethod
    def load_model(cls, path, device='cpu'):
        checkpoint = torch.load(path, map_location=device)
        cfg = checkpoint['config']
        model = cls(
            num_input_tokens=cfg['num_input_tokens'],
            num_classes=cfg['num_classes'],
            embedding_dim=cfg['embedding_dim'],
            hidden_dim=cfg['hidden_dim'],
            num_heads=cfg['num_heads'],
            num_layers=cfg['num_layers'],
            dropout=cfg.get('dropout', 0.1),
        ).to(device)
        model.load_state_dict(checkpoint['model_state'])
        return model

    def print_trainable_params(self):
        total = 0
        for name, param in self.named_parameters():
            if param.requires_grad:
                cnt = param.numel()
                print(f"{name:50} {cnt:,}")
                total += cnt
        print(f"\nTotal trainable parameters: {total:,}")
        return total