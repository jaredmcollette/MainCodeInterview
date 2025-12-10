import utils
import math, random, time
from dataclasses import dataclass
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn import functional as F
from datasets import load_dataset
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
from tqdm import tqdm
import structlog

@dataclass
class Hyperparameters:
    block_size: int = 256
    batch_size: int = 64
    vocab_size: int = 12_000
    n_layer: int = 4
    n_head: int = 8
    d_model: int = 512
    dropout: float = 0.1
    lr: float = 1e-3
    pct_start: float = 0.2
    div_factor: float = 5.0
    final_div_factor: float = 200.0
    weight_decay: float = 0.01
    evals_per_epoch: int = 3
    expansion_factor: float = 6

    # SparseK specific parameters
    sparse_k: int = 32  # Number of tokens to attend to
    sparse_gate_temp: float = 0.3  # Temperature for gating
    sparse_initial_k: int = 64  # Starting K value for adaptive mechanism
    
    epochs: int = 7
    seed: int = 1337
    num_titles: int = 100_000
    val_frac: float = 0.10
    log_file: str = "./logs/mainrun.log"

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

def configure_logging(log_file: str):
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    
    file_handler = open(log_file, 'w')
    
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    class DualLogger:
        def __init__(self, file_handler):
            self.file_handler = file_handler
            self.logger = structlog.get_logger()
            
        def log(self, event, **kwargs):
            log_entry = json.dumps({"event": event, "timestamp": time.time(), **kwargs})
            self.file_handler.write(log_entry + "\n")
            self.file_handler.flush()
            
            if kwargs.get("prnt", True):
                if "step" in kwargs and "max_steps" in kwargs:
                    tqdm.write(f"[{kwargs.get('step'):>5}/{kwargs.get('max_steps')}] {event}: loss={kwargs.get('loss', 'N/A'):.6f} time={kwargs.get('elapsed_time', 0):.2f}s")
                else:
                    parts = [f"{k}={v}" for k, v in kwargs.items() if k not in ["prnt", "timestamp"]]
                    if parts:
                        tqdm.write(f"{event}: {', '.join(parts)}")
                    else:
                        tqdm.write(event)
    
    return DualLogger(file_handler)

logger = None

def get_titles(num_titles: int, seed: int, val_frac: float) -> str:
    ds = load_dataset("julien040/hacker-news-posts", split="train", cache_dir="./data").shuffle(seed=seed)
    titles = [row["title"].strip() for row in ds.take(num_titles)]
    n = int(num_titles * (1 - val_frac))
    return titles[:n], titles[n:]

def get_random_batch(split_ids: torch.Tensor, block_size: int, batch_size: int, device: torch.device):
    starts = torch.randint(0, len(split_ids) - block_size - 1, (batch_size,))
    x = torch.stack([split_ids[i:i+block_size] for i in starts])
    y = torch.stack([split_ids[i+1:i+block_size+1] for i in starts])
    return x.to(device), y.to(device)

def iter_full_split(split_ids: torch.Tensor, block_size: int, batch_size: int, device: torch.device):
    span = block_size * batch_size + 1
    for ptr in range(0, len(split_ids) - span + 1, span):
        batch = split_ids[ptr: ptr + span]
        x = batch[:-1].view(batch_size, block_size).to(device)
        y = batch[1:].view(batch_size, block_size).to(device)
        yield x, y

def train_tokenizer(titles: list[str], vocab_size: int, unk_token: str = "<unk>", pad_token: str = "<pad>", eos_token: str = "<eos>") -> Tokenizer:
    tokenizer = Tokenizer(models.BPE(unk_token=unk_token, fuse_unk=True, byte_fallback=True))
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.WhitespaceSplit(),
        pre_tokenizers.Punctuation()
    ])
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=[pad_token, eos_token, unk_token],
        min_frequency=2,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        continuing_subword_prefix="##"
    )
    tokenizer.train_from_iterator(titles, trainer)
    return tokenizer

class BPETokenizer:
    def __init__(self, tokenizer: Tokenizer):
        self.tk = tokenizer
        self.stoi = {tok: i for tok, i in tokenizer.get_vocab().items()}
        self.itos = {i: tok for tok, i in tokenizer.get_vocab().items()}

    def encode(self, s: str) -> list[int]:
        return self.tk.encode(s).ids

    def decode(self, ids: list[int]) -> str:
        return self.tk.decode(ids, skip_special_tokens=True)

    @property
    def vocab_size(self): return self.tk.get_vocab_size()

@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int
    n_layer: int
    n_head: int
    d_model: int
    dropout: float
    expansion_factor: float
    sparse_k: int
    sparse_gate_temp: float
    sparse_initial_k: int

# ALiBi implementation
def build_alibi_mask(n_head: int, max_len: int) -> torch.Tensor:
    # Generate slopes (m_i) for each head
    if n_head > 1:
        slopes = 2 ** -(torch.arange(0, n_head) / (n_head - 1) * 8)
    else:
        slopes = torch.tensor([1.0])  # Single head case    
    # Create distance matrix    
    positions = torch.arange(max_len)
    dist = positions[:, None] - positions[None, :]
    # Create bias matrix
    return -slopes.view(-1, 1, 1) * dist.abs().view(1, max_len, max_len)

class AdaptiveSparseKGate(nn.Module):
    """Learnable gating mechanism for adaptive K selection"""
    def __init__(self, d_model: int, initial_k: int, gate_temp: float):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, 1)
        self.temperature = gate_temp
        self.initial_k = initial_k
        
    def forward(self, x: torch.Tensor, training: bool = True) -> torch.Tensor:
        # Compute gate scores
        scores = self.gate_proj(x).squeeze(-1)  # [B, T]
        
        if training:
            # Gumbel softmax for differentiable top-K selection
            gumbel_noise = -torch.log(-torch.log(torch.rand_like(scores) + 1e-10))
            noisy_scores = scores + gumbel_noise
            gate = torch.sigmoid(noisy_scores / self.temperature)
        else:
            # Hard selection during inference
            gate = (scores > 0).float()
        
        return gate

"""
SparseK Attention with adaptive token selection
Combines the efficiency of sparse attention with full attention's expressiveness
"""
class SparseKSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_head == 0
        self.head_dim = cfg.d_model // cfg.n_head
        self.n_head = cfg.n_head
        self.n_kv_heads = max(1, cfg.n_head // 4)
        self.sparse_k = cfg.sparse_k
        self.temperature = cfg.sparse_gate_temp
        
        # Register ALiBi mask
        self.register_buffer("alibi_bias", build_alibi_mask(cfg.n_head, cfg.block_size))
        
        # Projections
        self.q_proj = nn.Linear(cfg.d_model, self.n_head * self.head_dim)
        self.kv_proj = nn.Linear(cfg.d_model, 2 * self.n_kv_heads * self.head_dim)
        self.o_proj = nn.Linear(self.n_head * self.head_dim, cfg.d_model)
        
        # QK-Norm for stability
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)
        
        # Adaptive gating mechanism
        self.adaptive_gate = AdaptiveSparseKGate(
            cfg.d_model, cfg.sparse_initial_k, cfg.sparse_gate_temp
        )
        
        # Sparse selection parameters
        self.selection_scale = nn.Parameter(torch.tensor(1.0))
        
        # Dropout
        self.attn_drop = cfg.dropout
        self.resid_drop = nn.Dropout(cfg.dropout)
        
        # Additional parameters for learned top-K
        self.topk_router = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.n_head),
            nn.Tanh()
        )
        
        # Causal mask buffer
        self.register_buffer("causal_mask", 
            torch.triu(torch.ones(cfg.block_size, cfg.block_size, dtype=torch.bool), diagonal=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()
        
        # Compute adaptive gates
        gate_scores = self.adaptive_gate(x, self.training)
        
        # Project Q (all heads)
        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).permute(0, 2, 1, 3)
        
        # Project K,V (fewer heads)
        kv = self.kv_proj(x).view(B, T, 2, self.n_kv_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)
        
        # Apply QK-Norm
        q = self.q_norm(q)
        k = self.k_norm(k)
        
        # Expand KV to match Q heads (GQA)
        k = k.repeat_interleave(self.n_head // self.n_kv_heads, dim=1)
        v = v.repeat_interleave(self.n_head // self.n_kv_heads, dim=1)
        
        # ---- SparseK Selection Phase ----
        # 1. Compute preliminary attention scores for token selection
        with torch.no_grad():
            # Use a simplified attention computation for selection
            q_sel = q.detach()
            k_sel = k.detach()
            
            # Preliminary dot product (cheaper computation)
            pre_att = torch.matmul(q_sel, k_sel.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
            
            # Add ALiBi bias for selection
            bias_sel = self.alibi_bias[:, :T, :T]
            pre_att = pre_att + bias_sel
            
            # Apply causal mask
            causal_mask_sel = self.causal_mask[:T, :T]
            pre_att = pre_att.masked_fill(causal_mask_sel, float('-inf'))
            
            # Compute importance scores
            importance = torch.softmax(pre_att / self.temperature, dim=-1)
            
            # For each query, select top-K tokens
            topk_values, topk_indices = torch.topk(pre_att, k=min(self.sparse_k, T), dim=-1)
        
        # ---- Full Attention Computation (only on selected tokens) ----
        # Create attention mask with only selected tokens
        sparse_mask = torch.zeros_like(pre_att, dtype=torch.bool)
        
        # Mark top-K positions
        for b in range(B):
            for h in range(self.n_head):
                sparse_mask[b, h].scatter_(1, topk_indices[b, h], True)
        
        # Compute full attention on selected tokens
        att = torch.matmul(q, k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        
        # Add ALiBi bias
        bias = self.alibi_bias[:, :T, :T]
        att = att + bias
        
        # Apply sparse mask + causal mask
        combined_mask = self.causal_mask[:T, :T] | (~sparse_mask)
        att = att.masked_fill(combined_mask, float('-inf'))
        
        # Apply softmax and dropout
        att = F.softmax(att, dim=-1)
        
        # Incorporate gate scores to adapt attention weights
        gate_factor = gate_scores.unsqueeze(1).unsqueeze(-1)  # [B, 1, T, 1]
        
        # Apply attention dropout
        att = F.dropout(att, p=self.attn_drop if self.training else 0)
        
        # Compute weighted values
        y = torch.matmul(att, v)
        
        # Reshape and project
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_drop(self.o_proj(y))
        
        return y

# Originally from Llama
def get_hidden_dim(d_model: int, multiplier: float, multiple_of: int = 64) -> int:
    """
    Calculates the hidden dimension size.
    It applies the multiplier and rounds up to the nearest multiple of 'multiple_of'
    for hardware efficiency.
    """
    hidden_dim = int(d_model * multiplier)
    hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
    return hidden_dim

# SwiGLU MLP Implementation
class MLP(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        # In SwiGLU, we project to hidden_dim twice (gate and value), then project back.
        hidden_dim = get_hidden_dim(cfg.d_model, cfg.expansion_factor, multiple_of=64)
        
        self.w1 = nn.Linear(cfg.d_model, hidden_dim, bias=False) # Gate
        self.w2 = nn.Linear(cfg.d_model, hidden_dim, bias=False) # Value
        self.c_proj = nn.Linear(hidden_dim, cfg.d_model, bias=False) # Output
        
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        # SwiGLU: (Swish(Gate) * Value) -> Output
        x = F.silu(self.w1(x)) * self.w2(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

# Block with SparseK Attention
class Block(nn.Module):
    def __init__(self, cfg: GPTConfig, depth: int, drop_rate: float = 0.1):
        super().__init__()
        self.norm = RMSNorm(cfg.d_model)
        self.attn = SparseKSelfAttention(cfg)
        self.mlp = MLP(cfg)
        self.drop_rate = drop_rate * (depth / cfg.n_layer)
        self.residual_scale = math.sqrt(2 * depth)
        
    def forward(self, x):
        # Stochastic depth
        if self.training and random.random() < self.drop_rate:
            return x
        
        residual = x
        x_norm = self.norm(x)
        
        # Compute attention and MLP in parallel
        attn_out = self.attn(x_norm)
        mlp_out = self.mlp(x_norm)
        
        parallel_out = (attn_out + mlp_out) / self.residual_scale
        return residual + parallel_out

class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.drop      = nn.Dropout(cfg.dropout)
        self.blocks    = nn.ModuleList([Block(cfg, i+1) for i in range(cfg.n_layer)])
        self.ln_f      = RMSNorm(cfg.d_model)
        self.head      = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        self.apply(self._init_weights)

        # Depth-scaled init for output layers
        for block in self.blocks:
            nn.init.normal_(block.attn.o_proj.weight, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))
            nn.init.normal_(block.mlp.c_proj.weight, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

        self.head.weight = self.token_emb.weight

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
    
    # --- Weight Decay logic ---
    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # Filter params that require grad
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        
        # 1. Decay: Linear weights and Embeddings
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        
        # 2. No Decay: Biases, LayerNorms, Positional Embeddings
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]

        # Using fused AdamW if available for speed
        fused_available = 'fused' in torch.optim.AdamW.__init__.__code__.co_varnames
        use_fused = fused_available and 'cuda' in device_type
        extra_args = dict(fused=True) if use_fused else dict()
        
        # Create AdamW optimizer
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)
        return optimizer

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        B, T = idx.size()
        tok = self.token_emb(idx)
        x = self.drop(tok)
        for block in self.blocks: x = block(x)
        x = self.ln_f(x)
        logits = self.head(x)
        if targets is None:
            loss = None
        else:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), reduction='mean')
        return logits, loss

def main():
    args = Hyperparameters()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    
    global logger
    logger = configure_logging(args.log_file)
    
    hyperparams_dict = vars(args)
    logger.log("hyperparameters_configured", **hyperparams_dict)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.log("device_info", device=device)

    train_titles, val_titles = get_titles(args.num_titles, args.seed, args.val_frac)
    
    eos_token = "<eos>"
    tok = BPETokenizer(train_tokenizer(train_titles, args.vocab_size, eos_token=eos_token))
    train_text = eos_token.join(train_titles) + eos_token
    val_text = eos_token.join(val_titles) + eos_token
    train_ids = torch.tensor(tok.encode(train_text), dtype=torch.long)
    val_ids = torch.tensor(tok.encode(val_text), dtype=torch.long)
    
    batches = len(train_ids) // (args.block_size * args.batch_size)
    max_steps = args.epochs * batches
    eval_interval = batches // args.evals_per_epoch
    logger.log("dataset_info",
               titles_count=len(train_titles),
               epochs=args.epochs,
               batches_per_epoch=batches,
               tokens_per_epoch=len(train_ids),
               vocab_size=tok.vocab_size)

    cfg = GPTConfig(
        vocab_size = tok.vocab_size,
        block_size = args.block_size,
        n_layer    = args.n_layer,
        n_head     = args.n_head,
        d_model    = args.d_model,
        dropout    = args.dropout,
        expansion_factor = args.expansion_factor,
        sparse_k   = args.sparse_k,
        sparse_gate_temp = args.sparse_gate_temp,
        sparse_initial_k = args.sparse_initial_k
    )
    model = GPT(cfg).to(device)
    model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.log("model_info", parameters_count=model_params)
    
    opt = model.configure_optimizers(args.weight_decay, args.lr, (0.9, 0.95), device)

    # OneCycleLR scheduler
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer=opt,
        max_lr=args.lr,
        total_steps=max_steps,
        pct_start=args.pct_start,              
        anneal_strategy='cos',
        div_factor=args.div_factor,             
        final_div_factor=args.final_div_factor 
    )

    def evaluate():
        model.eval()
        losses = 0.0
        with torch.no_grad():
            for xb, yb in iter_full_split(val_ids, args.block_size, args.batch_size, device):
                logits, _ = model(xb, yb)
                B, T, V = logits.size()
                loss = F.cross_entropy(logits.view(-1, V), yb.view(-1), reduction='sum')
                losses += loss.item()
        model.train()
        return losses / len(val_text)

    ptr = 0
    step = 0
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        for _ in tqdm(range(1, batches + 1), desc=f"Epoch {epoch}/{args.epochs}"):
            step += 1
            xb, yb = get_random_batch(train_ids, args.block_size, args.batch_size, device)
            _, loss = model(xb, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            scheduler.step()

            elapsed = time.time() - t0
            logger.log("training_step",
                      step=step,
                      max_steps=max_steps,
                      loss=loss.item(),
                      elapsed_time=elapsed,
                      prnt=False)

            if step == 1 or step % eval_interval == 0 or step == max_steps:
                val_loss = evaluate()
                logger.log("validation_step",
                          step=step,
                          max_steps=max_steps,
                          loss=val_loss,
                          elapsed_time=elapsed)

if __name__ == "__main__":
    try:
        main()
    finally:
        if logger and hasattr(logger, 'file_handler'):
            logger.file_handler.close()
