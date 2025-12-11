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
    block_size: int = 64
    batch_size: int = 256
    vocab_size: int = 12_000
    n_layer: int = 4
    n_head: int = 6
    d_model: int = 510
    dropout: float = 0.1
    lr: float = 1e-3
    warmup_frac: float = 0.1
    pct_start: float = 0.2
    div_factor: float = 5.0
    final_div_factor: float = 100.0
    weight_decay: float = 0.1
    evals_per_epoch: int = 3
    expansion_factor: float = 6

    # MoE Specifics
    num_experts: int = 4
    top_k: int = 2
    
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
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
        return x / rms * self.weight

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

# shuffle chunks for exact epoch coverage
class ChunkedDataLoader:
    def __init__(self, data_ids: torch.Tensor, block_size: int, batch_size: int, device: torch.device, seed: int):
        self.data = data_ids
        self.block_size = block_size
        self.batch_size = batch_size
        self.device = device
        # Calculate how many full blocks fit in the data
        n_blocks = (len(data_ids) - 1) // block_size
        self.indices = torch.arange(0, n_blocks * block_size, block_size)
        self.generator = torch.Generator().manual_seed(seed)
        
    def __iter__(self):
        # Shuffle indices at the start of each epoch
        perms = torch.randperm(len(self.indices), generator=self.generator)
        shuffled_indices = self.indices[perms]
        
        for i in range(0, len(shuffled_indices), self.batch_size):
            batch_indices = shuffled_indices[i : i + self.batch_size]
            if len(batch_indices) < self.batch_size:
                continue # Drop last incomplete batch
                
            x_list = []
            y_list = []
            for idx in batch_indices:
                x_list.append(self.data[idx : idx + self.block_size])
                y_list.append(self.data[idx + 1 : idx + self.block_size + 1])
                
            yield torch.stack(x_list).to(self.device), torch.stack(y_list).to(self.device)

    def __len__(self):
        return len(self.indices) // self.batch_size

def iter_full_split(split_ids: torch.Tensor, block_size: int, batch_size: int, device: torch.device):
    span = block_size * batch_size + 1
    for ptr in range(0, len(split_ids) - span + 1, span):
        batch = split_ids[ptr: ptr + span]
        x = batch[:-1].view(batch_size, block_size).to(device)
        y = batch[1:].view(batch_size, block_size).to(device)
        yield x, y

def train_tokenizer(titles: list[str], vocab_size: int, unk_token: str = "<unk>", pad_token: str = "<pad>", eos_token: str = "<eos>") -> Tokenizer:
    tokenizer = Tokenizer(models.BPE(unk_token=unk_token, fuse_unk=True, byte_fallback=True))
    # Pre-tokenizer: Split on whitespace but keep punctuation isolated
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.WhitespaceSplit(),
        pre_tokenizers.Punctuation()
    ])
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=[eos_token, unk_token],
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
    num_experts: int
    top_k: int

# Custom linear ALiBi
def build_alibi_mask(n_head: int, max_len: int) -> torch.Tensor:
    dtype = torch.float32
    device = None

    def get_slopes(nh: int):
        if nh == 0:
            return torch.empty((0,), dtype=dtype, device=device)
        slopes = torch.arange(1, nh + 1, dtype=torch.float32, device=device) / nh
        return slopes.to(dtype)

    slopes = get_slopes(n_head)
    arange = torch.arange(max_len, dtype=dtype, device=device)
    i_pos = arange[None, :, None]
    j_pos = arange[None, None, :]
    bias = slopes[:, None, None] * (j_pos - i_pos)
    return bias

def get_rotary_sin_cos(head_dim, max_seq_len, device):
    inv_freq = 1.0 / (10000 ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(max_seq_len, device=device, dtype=inv_freq.dtype)
    freqs = torch.outer(t, inv_freq)
    # Polar coordinates trick: concating freqs twice
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos(), emb.sin()

def apply_rope(q, k, cos, sin):
    # Rotate half helper
    def rotate_half(x):
        x1 = x[..., :x.shape[-1]//2]
        x2 = x[..., x.shape[-1]//2:]
        return torch.cat((-x2, x1), dim=-1)

    # Slice cos/sin to the sequence length of q
    T = q.shape[2]
    cos = cos[:T, :].unsqueeze(0).unsqueeze(0)
    sin = sin[:T, :].unsqueeze(0).unsqueeze(0)

    q_rot = (q * cos) + (rotate_half(q) * sin)
    k_rot = (k * cos) + (rotate_half(k) * sin)
    return q_rot, k_rot

class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_head == 0
        self.head_dim = cfg.d_model // cfg.n_head
        self.n_head   = cfg.n_head

        # Precompute RoPE cos/sin table
        # cos, sin = get_rotary_sin_cos(self.head_dim, cfg.block_size, device=None)
        # self.register_buffer("cos", cos, persistent=False)
        # self.register_buffer("sin", sin, persistent=False)

        # Register ALiBi mask
        self.register_buffer("alibi_bias", build_alibi_mask(cfg.n_head, cfg.block_size))

        self.q_proj = nn.Linear(cfg.d_model, self.n_head * self.head_dim)
        self.kv_proj = nn.Linear(cfg.d_model, 2 * self.n_head * self.head_dim)
        self.o_proj = nn.Linear(self.n_head * self.head_dim, cfg.d_model)

        self.attn_drop = cfg.dropout
        self.resid_drop= nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor):
        B, T, C = x.size()

        # Project Q (all heads)
        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).permute(0, 2, 1, 3)

        # Project K,V (fewer heads)
        kv = self.kv_proj(x).view(B, T, 2, self.n_head, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)

        # q, k = apply_rope(q, k, self.cos, self.sin)

        # Attention scores
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))

        # Add ALiBi bias
        bias = self.alibi_bias[:, :T, :T]  # Slice to current sequence length 
        att = att + bias

        # Apply causal mask
        mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        att = att.masked_fill(mask, float('-inf'))
        # Softmax and attention
        att = F.softmax(att, dim=-1)
        att = F.dropout(att, p=self.attn_drop if self.training else 0)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.o_proj(y))

# SwiGLU MLP Implementation
class MLP(nn.Module):
    def __init__(self, cfg: GPTConfig, depth: int):
        super().__init__()
        hidden_dim = int(cfg.d_model * cfg.expansion_factor)
        
        self.gate_proj = nn.Linear(cfg.d_model, hidden_dim, bias=False)
        self.up_proj = nn.Linear(cfg.d_model, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, cfg.d_model, bias=False)
        
        self.dropout = nn.Dropout(cfg.dropout)
        self.depth_scale = 1 / math.sqrt(depth)

    def forward(self, x):
        # SwiGLU variant
        x = F.silu(self.gate_proj(x)) * self.up_proj(x)
        x = self.down_proj(x) * self.depth_scale
        x = self.dropout(x)
        return x

# 2. MoE Router
class MoELayer(nn.Module):
    def __init__(self, cfg: GPTConfig, depth: int):
        super().__init__()
        self.num_experts = cfg.num_experts
        self.top_k = cfg.top_k
        
        # The router decides which experts get which tokens
        self.router = nn.Linear(cfg.d_model, self.num_experts, bias=False)
        
        # Create 'num_experts' instances of the SwiGLU layer
        self.experts = nn.ModuleList([MLP(cfg, depth) for _ in range(self.num_experts)])

        self.noise_std = 0.1  # Amount of noise to add

    def forward(self, x):
        B, T, C = x.shape
        flat_x = x.view(-1, C)
        
        # 1. Router logits
        router_logits = self.router(flat_x)

        if self.training:
            # Add random noise to logits to force exploration
            noise = torch.randn_like(router_logits) * self.noise_std
            # Multiply logits by a small factor to ensure noise has effect
            router_logits = router_logits + noise
        
        # 2. Select Top-K
        # routing_weights: (B*T, top_k)
        # selected_experts: (B*T, top_k) indices
        routing_weights, selected_experts = torch.topk(router_logits, self.top_k, dim=-1)
        routing_weights = F.softmax(routing_weights, dim=-1, dtype=torch.float).to(x.dtype)
        
        final_output = torch.zeros_like(flat_x)
        
        # 3. Process Experts
        # Loop through each rank (1st choice, 2nd choice...)
        for k in range(self.top_k):
            expert_idx = selected_experts[:, k]
            weight = routing_weights[:, k].unsqueeze(-1)
            
            # Iterate over all available experts
            for i, expert in enumerate(self.experts):
                # Mask: which tokens chose expert 'i' for rank 'k'?
                mask = (expert_idx == i)
                
                # if mask.any():
                inp = flat_x[mask]
                out = expert(inp)
                final_output[mask] += weight[mask] * out
                    
        return final_output.view(B, T, C)

class Block(nn.Module):
    def __init__(self, cfg: GPTConfig, depth: int, drop_rate: float = 0.0):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model)
        self.ffn_norm = RMSNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ffn = MoELayer(cfg, depth)

    def forward(self, x):
        x = x + self.attn(self.attn_norm(x))
        x = x + self.ffn(self.ffn_norm(x))
        return x

# Parallel Residual Block
# class Block(nn.Module):
#     def __init__(self, cfg: GPTConfig, depth: int, drop_rate: float = 0.0):
#         super().__init__()
#         # Shared layer norm for both branches 
#         self.attn_norm = RMSNorm(cfg.d_model)
#         self.ffn_norm = RMSNorm(cfg.d_model)
        
#         self.attn = CausalSelfAttention(cfg)
#         self.ffn  = MLP(cfg)
#         # self.drop_rate = drop_rate * (depth / cfg.n_layer)
#         self.residual_scale = math.sqrt(2 * depth)
        
#     def forward(self, x):
#         # Stochastic depth (disabled)
#         # if self.training and random.random() < self.drop_rate:
#         #     return x

#         residual = x
#         x_norm = self.norm(x)
#         # Single normalization shared by both branches
#         # Compute attention and MLP in parallel
#         attn_out = self.attn(x_norm)
#         mlp_out = self.ffn(x_norm)

#         parallel_out = (attn_out + mlp_out) / self.residual_scale
#         return residual + parallel_out

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
        self.gradient_checkpointing = True

        # Depth-scaled init for output layers
        for block in self.blocks:
            nn.init.normal_(block.attn.o_proj.weight, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))
            for expert in block.ffn.experts:
                nn.init.normal_(expert.down_proj.weight, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

        self.head.weight = self.token_emb.weight

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, RMSNorm):
            nn.init.constant_(module.weight, 1.0)
    
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
        fused = 'fused' in torch.optim.AdamW.__init__.__code__.co_varnames and 'cuda' in device_type

        # Create AdamW optimizer
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, fused=fused)
        return optimizer

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        B, T = idx.size()
        tok = self.token_emb(idx)
        x = self.drop(tok)
        if self.gradient_checkpointing:
            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs)
                return custom_forward

            for block in self.blocks:
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    x,
                    use_reentrant=False
                )
        else:
            for block in self.blocks: x = block(x)
        x = self.ln_f(x)
        logits = self.head(x)
        if targets is None:
            loss = None
        else:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), reduction='mean')
        return logits, loss

class CosineWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr=1e-6):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        super().__init__(optimizer)

    def get_lr(self):
        progress = self.last_epoch / self.total_steps
        if self.last_epoch < self.warmup_steps:
            lr = self.base_lrs[0] * self.last_epoch / self.warmup_steps
        else:
            cos_decay = 0.5 * (1 + math.cos(math.pi * (progress - self.warmup_steps / self.total_steps) / (1 - self.warmup_steps / self.total_steps)))
            lr = self.min_lr + (self.base_lrs[0] - self.min_lr) * cos_decay
        return [lr for _ in self.optimizer.param_groups]

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
    
    train_loader = ChunkedDataLoader(train_ids, args.block_size, args.batch_size, device, args.seed)
    batches_per_epoch = len(train_loader)
    max_steps = args.epochs * batches_per_epoch
    eval_interval = batches_per_epoch // args.evals_per_epoch
    logger.log("dataset_info",
               titles_count=len(train_titles),
               epochs=args.epochs,
               batches_per_epoch=batches_per_epoch,
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
        num_experts = args.num_experts,
        top_k = args.top_k

    )
    model = GPT(cfg).to(device)
    if hasattr(torch, 'compile'):
        try:
            model = torch.compile(model)
            logger.log("model_compiled", mode="default")
        except:
            logger.log("compilation_failed", error=str(e))
            pass

    model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.log("model_info", parameters_count=model_params)
    
    opt = model.configure_optimizers(args.weight_decay, args.lr, (0.9, 0.999), device)

    warmup_steps = int(args.warmup_frac * max_steps)
    scheduler = CosineWarmupScheduler(opt, warmup_steps, max_steps, 1e-6)

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
        for xb, yb in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}"):
            step += 1
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
