import torch
import torch.nn as nn
import torch.nn.functional as F
from difflogic import LogicLayer, GroupSum, PackBitsTensor
from models.base_model import BaseModel
from utils.gumbel_logic_wrapper import patch_logic_layers_with_gumbel

class BidirectionalRecurrentDifflogic(BaseModel):
    def __init__(
        self, num_input_tokens, embedding_dim, seq_length,
        n_layers_sizes, j_layers_sizes, k_layers_sizes, g_layers_sizes, l_layers_sizes, p_layers_sizes, m_layers_sizes, num_classes,
        group_factor=1,
        device="cuda", grad_factor=1.0,
        connections='random', predefined_indices=None,
        difflogic_init_type='noisy_residual',
        noise_factor=1.0,
        hidden_state_init_type='zero',
        seed=None,
        group_sum_tau=1.0,
        gumbel_tau=None,
        use_st_estimator=True,
        dropout_prob=0.2,
        padding_idx=0, bos_token_id=2, eos_token_id=3
    ):
        super().__init__()
        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        self.device = device
        self.seq_length = seq_length
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.num_input_tokens = num_input_tokens
        self.group_factor = group_factor
        self.difflogic_init_type = difflogic_init_type
        self.noise_factor = noise_factor
        self.hidden_state_init_type = hidden_state_init_type
        self.group_sum_tau = group_sum_tau
        self.gumbel_tau = gumbel_tau
        self.use_st_estimator = use_st_estimator
        self.grad_factor = grad_factor
        self.dropout_prob = dropout_prob
        self.padding_idx = padding_idx
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id

        assert all(len(x) > 0 for x in [n_layers_sizes, j_layers_sizes, k_layers_sizes, g_layers_sizes, l_layers_sizes, p_layers_sizes, m_layers_sizes]), "All layer groups must have at least one layer"

        final_m_out = num_classes * group_factor
        m_layers_sizes.append(final_m_out)

        self.embedding = nn.Embedding(num_input_tokens, embedding_dim, padding_idx=padding_idx)
        self.emb_dropout = nn.Dropout(dropout_prob)
        self.n_dropouts = nn.ModuleList([nn.Dropout(dropout_prob) for _ in n_layers_sizes])
        self.j_dropouts = nn.ModuleList([nn.Dropout(dropout_prob) for _ in j_layers_sizes])
        self.k_dropouts = nn.ModuleList([nn.Dropout(dropout_prob) for _ in k_layers_sizes])
        self.g_dropouts = nn.ModuleList([nn.Dropout(dropout_prob) for _ in g_layers_sizes])
        self.l_dropouts = nn.ModuleList([nn.Dropout(dropout_prob) for _ in l_layers_sizes])
        self.p_dropouts = nn.ModuleList([nn.Dropout(dropout_prob) for _ in p_layers_sizes])
        self.m_dropouts = nn.ModuleList([nn.Dropout(dropout_prob) for _ in m_layers_sizes])

        # N layers
        self.n_layers = nn.ModuleList()
        prev_dim = embedding_dim
        for size in n_layers_sizes:
            layer = LogicLayer(
                in_dim=prev_dim, out_dim=size, device=device,
                grad_factor=grad_factor, implementation='cuda', connections=connections
            )
            self._init_logic_layer(layer)
            self.n_layers.append(layer)
            prev_dim = size
        self.n_out = prev_dim

        # J layers -- operate on top of last N and last K
        self.j_layers = nn.ModuleList()
        j_in_dim = self.n_out + k_layers_sizes[-1]
        for size in j_layers_sizes:
            layer = LogicLayer(
                in_dim=j_in_dim, out_dim=size, device=device,
                grad_factor=grad_factor, implementation='cuda', connections=connections
            )
            self._init_logic_layer(layer)
            self.j_layers.append(layer)
            j_in_dim = size
        self.j_out = j_in_dim

        # K layers (left-to-right recurrent)
        self.k_layers = nn.ModuleList()
        k_in_dim = self.n_out + k_layers_sizes[-1]
        for i, size in enumerate(k_layers_sizes):
            layer = LogicLayer(
                in_dim=k_in_dim, out_dim=size, device=device,
                grad_factor=grad_factor, implementation='cuda', connections=connections
            )
            self._init_logic_layer(layer)
            self.k_layers.append(layer)
            k_in_dim = size
        self.k_hidden_dim = k_layers_sizes[-1]

        # G layers (right-to-left recurrent)
        self.g_layers = nn.ModuleList()
        g_in_dim = self.j_out + g_layers_sizes[-1]
        for i, size in enumerate(g_layers_sizes):
            layer = LogicLayer(
                in_dim=g_in_dim, out_dim=size, device=device,
                grad_factor=grad_factor, implementation='cuda', connections=connections
            )
            self._init_logic_layer(layer)
            self.g_layers.append(layer)
            g_in_dim = size
        self.g_hidden_dim = g_layers_sizes[-1]

        # L layers (for decoder)
        self.l_layers = nn.ModuleList()
        prev_dim = embedding_dim
        for size in l_layers_sizes:
            layer = LogicLayer(
                in_dim=prev_dim, out_dim=size, device=device,
                grad_factor=grad_factor, implementation='cuda', connections=connections
            )
            self._init_logic_layer(layer)
            self.l_layers.append(layer)
            prev_dim = size
        self.l_out = prev_dim

        # P layers (decoder recurrent)
        self.p_layers = nn.ModuleList()
        p_in_dim = p_layers_sizes[-1] + g_layers_sizes[-1] + k_layers_sizes[-1] + l_layers_sizes[-1]
        for i, size in enumerate(p_layers_sizes):
            layer = LogicLayer(
                in_dim=p_in_dim, out_dim=size, device=device,
                grad_factor=grad_factor, implementation='cuda', connections=connections
            )
            self._init_logic_layer(layer)
            self.p_layers.append(layer)
            p_in_dim = size
        self.p_hidden_dim = p_layers_sizes[-1]

        # M layers (output)
        self.m_layers = nn.ModuleList()
        m_in_dim = self.p_hidden_dim + self.g_hidden_dim + self.k_hidden_dim + self.l_out
        for i, size in enumerate(m_layers_sizes):
            # Keep assert only for the final M layer
            if i == len(m_layers_sizes) - 1:
                assert size == final_m_out, f"Final M layer size must be {final_m_out}"
            layer = LogicLayer(
                in_dim=m_in_dim, out_dim=size, device=device,
                grad_factor=grad_factor, implementation='cuda', connections=connections
            )
            self._init_logic_layer(layer)
            self.m_layers.append(layer)
            m_in_dim = size

        # Initialize hidden states for K, G, and P layers
        self._init_hiddens()

        self.final_sum = GroupSum(k=num_classes, tau=group_sum_tau, device=device)

        if gumbel_tau is not None:
            patch_logic_layers_with_gumbel(
                self, tau=gumbel_tau, use_st_estimator=use_st_estimator, verbose=False
            )

        self.set_mode('train')

    def _init_logic_layer(self, layer):
        with torch.no_grad():
            if self.difflogic_init_type == 'residual':
                layer.weights.data.zero_()
                if layer.weights.size(1) > 3:
                    layer.weights.data[:, 3].fill_(5.0)
            elif self.difflogic_init_type == 'noisy_residual':
                layer.weights.data.normal_(mean=0.0, std=self.noise_factor)
                if layer.weights.size(1) > 3:
                    layer.weights.data[:, 3].add_(5.0)
            elif self.difflogic_init_type == 'gaussian':
                layer.weights.data.normal_(mean=0.0, std=self.noise_factor)

    def _init_hiddens(self):
        # K hidden state
        if self.hidden_state_init_type == 'learnable':
            self.initial_k_hidden = nn.Parameter(
                torch.sigmoid(torch.randn(self.k_hidden_dim, device=self.device))
            )
        else:
            buf = self._get_init_buffer(self.k_hidden_dim)
            self.register_buffer('initial_k_hidden', buf)
        
        # G hidden state
        if self.hidden_state_init_type == 'learnable':
            self.initial_g_hidden = nn.Parameter(
                torch.sigmoid(torch.randn(self.g_hidden_dim, device=self.device))
            )
        else:
            buf = self._get_init_buffer(self.g_hidden_dim)
            self.register_buffer('initial_g_hidden', buf)
        
        # P hidden state
        if self.hidden_state_init_type == 'learnable':
            self.initial_p_hidden = nn.Parameter(
                torch.sigmoid(torch.randn(self.p_hidden_dim, device=self.device))
            )
        else:
            buf = self._get_init_buffer(self.p_hidden_dim)
            self.register_buffer('initial_p_hidden', buf)

    def _get_init_buffer(self, dim):
        if self.hidden_state_init_type == 'zero':
            return torch.zeros(dim, device=self.device)
        elif self.hidden_state_init_type == 'one':
            return torch.ones(dim, device=self.device)
        elif self.hidden_state_init_type == 'gaussian':
            return torch.sigmoid(torch.randn(dim, device=self.device))
        elif self.hidden_state_init_type == 'uniform':
            return torch.rand(dim, device=self.device)
        else:
            raise ValueError(f"Unknown hidden_state_init_type: {self.hidden_state_init_type}")

    def _init_k_hidden(self, batch_size):
        return self.initial_k_hidden.unsqueeze(0).expand(batch_size, -1)

    def _init_p_hidden(self, batch_size):
        return self.initial_p_hidden.unsqueeze(0).expand(batch_size, -1)

    def _init_g_hidden(self, batch_size):
        return self.initial_g_hidden.unsqueeze(0).expand(batch_size, -1)

    def encode(self, src):
        """
        Encode source sequence using N, K, J, and G layers
        src: [batch_size, src_seq_len]
        Returns: encoder_outputs, final_g_out, final_k_out, encoder_binary_reg_loss
        """
        batch_size, seq_len = src.size(0), src.size(1)
        embedded = (self.embedding(src) > 0).float() if self.emb_mode else torch.sigmoid(self.embedding(src))
        embedded = self.emb_dropout(embedded)
        encoder_binary_reg_loss = torch.mean(embedded * (1 - embedded))

        k_hidden = self._init_k_hidden(batch_size)
        j_outputs = []
        last_n_output = None

        # Forward pass (left-to-right) for K
        for t in range(seq_len):
            x_step = embedded[:, t, :]
            # N layers
            for i, layer in enumerate(self.n_layers):
                x_step = layer(x_step)
                x_step = self.n_dropouts[i](x_step)
            last_n_output = x_step

            # K layers
            combined_k = torch.cat([x_step, k_hidden], dim=1)
            k_out = combined_k
            for i, layer in enumerate(self.k_layers):
                k_out = layer(k_out)
                k_out = self.k_dropouts[i](k_out)
            k_hidden = k_out

            # J layers
            j_input = torch.cat([last_n_output, k_hidden], dim=1)
            j_out = j_input
            for i, layer in enumerate(self.j_layers):
                j_out = layer(j_out)
                j_out = self.j_dropouts[i](j_out)
            j_outputs.append(j_out)

        g_hidden = self._init_g_hidden(batch_size)
        # G layers process the J outputs in reverse order through time
        for j_out in reversed(j_outputs):
            combined_g = torch.cat([j_out, g_hidden], dim=1)
            g_out = combined_g
            for i, layer in enumerate(self.g_layers):
                g_out = layer(g_out)
                g_out = self.g_dropouts[i](g_out)
            g_hidden = g_out

        return (
            g_hidden,
            k_hidden,
            encoder_binary_reg_loss
        )

    def decode(self, tgt, encoder_final_g, encoder_final_k):
        """
        Decode target sequence using L, P, M layers
        tgt: [batch_size, tgt_seq_len]
        encoder_final_g, encoder_final_k: [batch_size, ...]
        Returns: decoder_outputs (logits for each timestep), decoder_binary_reg_loss
        """
        batch_size, seq_len = tgt.size(0), tgt.size(1)
        embedded = (self.embedding(tgt) > 0).float() if self.emb_mode else torch.sigmoid(self.embedding(tgt))
        embedded = self.emb_dropout(embedded)
        decoder_binary_reg_loss = torch.mean(embedded * (1 - embedded))

        # P hidden state initialization
        p_hidden = self._init_p_hidden(batch_size)
        all_outputs = []
        
        for t in range(seq_len):
            x_step = embedded[:, t, :]
            for i, layer in enumerate(self.l_layers):
                x_step = layer(x_step)
                x_step = self.l_dropouts[i](x_step)
            l_out = x_step
        
            # P layers: concat [p_hidden, encoder_final_g, encoder_final_k, l_out]
            p_combined = torch.cat([p_hidden, encoder_final_g, encoder_final_k, l_out], dim=1)
            p_out = p_combined
            for i, layer in enumerate(self.p_layers):
                p_out = layer(p_out)
                p_out = self.p_dropouts[i](p_out)
            p_hidden = p_out

            # M layers: concat [p_hidden, encoder_final_g, encoder_final_k, l_out]
            m_combined = torch.cat([p_hidden, encoder_final_g, encoder_final_k, l_out], dim=1)
            m_out = m_combined
            for i, layer in enumerate(self.m_layers):
                m_out = layer(m_out)
                m_out = self.m_dropouts[i](m_out)
            group_out = self.final_sum(m_out)
            all_outputs.append(group_out)

        return torch.stack(all_outputs, dim=1), decoder_binary_reg_loss

    def forward(self, src, tgt=None):
        encoder_final_g, encoder_final_k, encoder_binary_reg_loss = self.encode(src)
        if tgt is not None:
            decoder_input = tgt[:, :-1]
            decoder_outputs, decoder_binary_reg_loss = self.decode(decoder_input, encoder_final_g, encoder_final_k)
            total_binary_reg_loss = encoder_binary_reg_loss + decoder_binary_reg_loss
            return decoder_outputs, total_binary_reg_loss
        else:
            return self.generate_step_by_step(src, encoder_final_g, encoder_final_k)

    def generate_step_by_step(self, src, encoder_final_g, encoder_final_k, max_length=50):
        batch_size = src.size(0)
        device = src.device
        decoder_input = torch.full((batch_size, 1), self.bos_token_id, dtype=torch.long, device=device)
        outputs = []
        p_hidden = self._init_p_hidden(batch_size)
        
        for t in range(max_length):
            embedded = (self.embedding(decoder_input) > 0).float() if self.emb_mode else torch.sigmoid(self.embedding(decoder_input))
            embedded = self.emb_dropout(embedded)
            x_step = embedded[:, 0, :]
            
            # L layers
            for i, layer in enumerate(self.l_layers):
                x_step = layer(x_step)
                x_step = self.l_dropouts[i](x_step)
            l_out = x_step
            
            # P layers: concat [p_hidden, encoder_final_g, encoder_final_k, l_out]
            p_combined = torch.cat([p_hidden, encoder_final_g, encoder_final_k, l_out], dim=1)
            p_out = p_combined
            for i, layer in enumerate(self.p_layers):
                p_out = layer(p_out)
                p_out = self.p_dropouts[i](p_out)
            p_hidden = p_out
            
            # M layers: concat [p_hidden, encoder_final_g, encoder_final_k, l_out]
            m_combined = torch.cat([p_hidden, encoder_final_g, encoder_final_k, l_out], dim=1)
            m_out = m_combined
            for i, layer in enumerate(self.m_layers):
                m_out = layer(m_out)
                m_out = self.m_dropouts[i](m_out)
            group_out = self.final_sum(m_out)
            outputs.append(group_out.unsqueeze(1))
            
            next_token = torch.argmax(group_out, dim=-1, keepdim=True)
            decoder_input = next_token.unsqueeze(1)
            if (next_token == self.eos_token_id).all():
                break
                
        if outputs:
            return torch.cat(outputs, dim=1)
        else:
            return torch.zeros(batch_size, 0, self.num_classes, device=device)

    def set_mode(self, mode):
        assert mode in ['train', 'eval', 'eval_col_emb', 'eval_col_layer', 'eval_col_all'], "Mode must be 'train' or 'eval'"
        self.train() if mode in ['train', 'eval', 'eval_col_emb'] else self.eval()
        self.emb_mode = mode in ['eval_col_emb', 'eval_col_all']

    def analyze_logic_function_distribution(self):
        function_names = [
            "FALSE", "AND", "A & ~B", "A", "~A & B", "B", "XOR", "OR",
            "NOR", "XNOR", "~B", "A | ~B", "~A", "~A | B", "NAND", "TRUE"
        ]
        distribution = {
            'n_layers': [],
            'j_layers': [],
            'k_layers': [],
            'g_layers': [],
            'l_layers': [],
            'p_layers': [],
            'm_layers': []
        }
        for layer_group_name, layer_group in [
            ('n_layers', self.n_layers), 
            ('j_layers', self.j_layers),
            ('k_layers', self.k_layers),
            ('g_layers', self.g_layers),
            ('l_layers', self.l_layers), 
            ('p_layers', self.p_layers), 
            ('m_layers', self.m_layers)
        ]:
            for i, layer in enumerate(layer_group):
                func_indices = layer.weights.argmax(dim=1)
                counts = torch.bincount(func_indices, minlength=16).tolist()
                softmax_probs = F.softmax(layer.weights, dim=1)
                avg_softmax = softmax_probs.mean(dim=0).tolist()
                argmax_counts = torch.bincount(func_indices, minlength=16).float()
                avg_argmax_probs = (argmax_counts / layer.out_dim).tolist()
                neuron_entropies = []
                for n in range(softmax_probs.size(0)):
                    neuron_probs = softmax_probs[n]
                    neuron_entropy = -torch.sum(neuron_probs * torch.log2(neuron_probs + 1e-10))
                    neuron_entropies.append(neuron_entropy.item())
                mean_neuron_entropy = sum(neuron_entropies) / len(neuron_entropies)
                layer_info = {
                    'layer_idx': i,
                    'layer_name': f"{layer_group_name}_{i}",
                    'out_dim': layer.out_dim,
                    'function_counts': counts,
                    'function_names': function_names,
                    'most_used_function': counts.index(max(counts)),
                    'most_used_count': max(counts),
                    'percentage_most_used': max(counts) / layer.out_dim * 100,
                    'avg_softmax_probs': avg_softmax,
                    'avg_argmax_probs': avg_argmax_probs,
                    'entropy_softmax': mean_neuron_entropy
                }
                distribution[layer_group_name].append(layer_info)
        return distribution

    def print_trainable_params(self):
        total_params = 0
        for name, param in self.named_parameters():
            if param.requires_grad:
                params = param.numel()
                total_params += params
                print(f"{name}: {params:,}")
        print(f"Total trainable parameters: {total_params:,}")
        return total_params

    def get_predictions(self, outputs):
        if isinstance(outputs, tuple):
            return outputs[0]
        return outputs

    def save_model(self, path):
        indices_list = []
        for layer in (
            self.n_layers +
            self.j_layers +
            self.k_layers +
            self.g_layers +
            self.l_layers +
            self.p_layers +
            self.m_layers
        ):
            indices_list.append((
                layer.indices[0].cpu().clone(),
                layer.indices[1].cpu().clone()
            ))
        torch.save({
            'model_state': self.state_dict(),
            'indices': indices_list,
            'config': {
                'num_input_tokens': self.num_input_tokens,
                'num_classes': self.num_classes,
                'embedding_dim': self.embedding_dim,
                'seq_length': self.seq_length,
                'n_layers_sizes': [l.out_dim for l in self.n_layers],
                'j_layers_sizes': [l.out_dim for l in self.j_layers],
                'k_layers_sizes': [l.out_dim for l in self.k_layers],
                'g_layers_sizes': [l.out_dim for l in self.g_layers],
                'l_layers_sizes': [l.out_dim for l in self.l_layers],
                'p_layers_sizes': [l.out_dim for l in self.p_layers],
                'm_layers_sizes': [l.out_dim for l in self.m_layers[:-1]],
                'device': self.device,
                'grad_factor': self.grad_factor,
                'group_sum_tau': self.group_sum_tau,
                'gumbel_tau': self.gumbel_tau,
                'use_st_estimator': self.use_st_estimator,
                'dropout_prob': self.dropout_prob,
                'padding_idx': self.padding_idx,
                'bos_token_id': self.bos_token_id,
                'eos_token_id': self.eos_token_id
            }
        }, path)

    @classmethod
    def load_model(cls, path, device='cuda'):
        checkpoint = torch.load(path, map_location=device)
        cfg = checkpoint['config']
        group_sum_tau = cfg.get('group_sum_tau', 1.0)
        gumbel_tau = cfg.get('gumbel_tau', None)
        use_st_estimator = cfg.get('use_st_estimator', True)
        dropout_prob = cfg.get('dropout_prob', 0.2)
        padding_idx = cfg.get('padding_idx', 0)
        bos_token_id = cfg.get('bos_token_id', 2)
        eos_token_id = cfg.get('eos_token_id', 3)
        model = cls(
            num_input_tokens=cfg['num_input_tokens'],
            embedding_dim=cfg['embedding_dim'],
            seq_length=cfg['seq_length'],
            n_layers_sizes=cfg['n_layers_sizes'],
            j_layers_sizes=cfg['j_layers_sizes'],
            k_layers_sizes=cfg['k_layers_sizes'],
            g_layers_sizes=cfg['g_layers_sizes'],
            l_layers_sizes=cfg['l_layers_sizes'],
            p_layers_sizes=cfg['p_layers_sizes'],
            m_layers_sizes=cfg['m_layers_sizes'],
            num_classes=cfg['num_classes'],
            device=device,
            grad_factor=cfg['grad_factor'],
            group_sum_tau=group_sum_tau,
            gumbel_tau=gumbel_tau,
            use_st_estimator=use_st_estimator,
            dropout_prob=dropout_prob,
            padding_idx=padding_idx,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id
        )
        all_layers = (
            model.n_layers +
            model.j_layers +
            model.k_layers +
            model.g_layers +
            model.l_layers +
            model.p_layers +
            model.m_layers
        )
        for layer, (a_idx, b_idx) in zip(all_layers, checkpoint['indices']):
            layer.indices = (
                a_idx.to(device),
                b_idx.to(device)
            )
        model.load_state_dict(checkpoint['model_state'])
        return model