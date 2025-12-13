import utils
import math, random, time
from dataclasses import dataclass
import json
from pathlib import Path
from enum import Enum

import torch
import torch.nn as nn
from torch.nn import functional as F
from datasets import load_dataset
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
from tqdm import tqdm
import structlog

class PositionalEmbeddingType(str, Enum):
    ALIBI = "alibi"
    ROPE = "rope"

@dataclass
class Hyperparameters:
    # Dataset
    block_size: int = 512
    batch_size: int = 64
    vocab_size: int = 12_000

    # Architecture
    n_layer: int = 4
    n_head: int = 6
    d_model: int = 504
    dropout: float = 0.1
    expansion_factor: float = 6
    pos_emb_type: PositionalEmbeddingType = PositionalEmbeddingType.ALIBI

    # MoE Specifics
    num_experts: int = 4
    top_k: int = 2

    # Logging
    evals_per_epoch: int = 3

    # Optimizer/Scheduler
    lr: float = 8e-4
    min_lr: float = 1e-6
    weight_decay: float = 0.1
    betas: tuple[float, float] = (0.9, 0.999)
    warmup_frac: float = 0.1

    # Fixed Constraints
    epochs: int = 7
    seed: int = 1337
    num_titles: int = 100_000
    val_frac: float = 0.10
    log_file: str = "./logs/mainrun.log"

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

class ShuffledBlockDataLoader:
    """
    A custom data loader that treats the dataset as a continuous stream of tokens,
    chunks it into fixed-size blocks, and serves them in shuffled batches.
    
    This ensures that in one epoch, the model sees every possible full block 
    exactly once, but in a random order (Training without replacement).
    """
    def __init__(self, data_ids: torch.Tensor, block_size: int, batch_size: int, device: torch.device, seed: int):
        self.data = data_ids
        self.block_size = block_size
        self.batch_size = batch_size
        self.device = device

        # Calculate how many full context blocks fit into the dataset.
        # We subtract 1 because the targets are shifted by 1 position relative to inputs.
        n_blocks = (len(data_ids) - 1) // block_size

        # Pre-calculate the starting index for every block in the dataset.
        # instead of copying data, we just store the integer pointers (0, 512, 1024...)
        self.indices = torch.arange(0, n_blocks * block_size, block_size)

        # Create a specific generator to ensure shuffling is reproducible based on the seed
        self.generator = torch.Generator().manual_seed(seed)
        
    def __iter__(self):
        # 1. Randomization:
        # Generate a random permutation of indices [0, 1, ... num_blocks-1].
        # This effectively shuffles the order in which we visit the chunks 
        # without moving the actual data in memory.
        perms = torch.randperm(len(self.indices), generator=self.generator)
        shuffled_indices = self.indices[perms]

        # 2. Batching:
        # Iterate through the shuffled start indices in steps of 'batch_size
        for i in range(0, len(shuffled_indices), self.batch_size):
            batch_indices = shuffled_indices[i : i + self.batch_size]

            # Drop Last:
            # If we don't have enough chunks to fill the final batch (e.g., leftover 14 items 
            # for a batch of 64), we skip it to maintain constant tensor shapes.
            if len(batch_indices) < self.batch_size:
                continue
                
            x_list = []
            y_list = []

            # 3. Slicing:
            # Construct the actual tensors for this batch
            for idx in batch_indices:
                # Input: tokens [t, t+block_size]
                x_list.append(self.data[idx : idx + self.block_size])
                # Target: tokens [t+1, t+block_size+1] (Next token prediction)
                y_list.append(self.data[idx + 1 : idx + self.block_size + 1])
                
            # Stack into (B, T) tensors and move to GPU
            yield torch.stack(x_list).to(self.device), torch.stack(y_list).to(self.device)

    def __len__(self):
        # Returns the number of batches per epoch (excluding the dropped last partial batch).
        return len(self.indices) // self.batch_size

def iter_full_split(split_ids: torch.Tensor, block_size: int, batch_size: int, device: torch.device):
    span = block_size * batch_size + 1
    for ptr in range(0, len(split_ids) - span + 1, span):
        batch = split_ids[ptr: ptr + span]
        x = batch[:-1].view(batch_size, block_size).to(device)
        y = batch[1:].view(batch_size, block_size).to(device)
        yield x, y

def train_tokenizer(titles: list[str], vocab_size: int, unk_token: str = "<unk>", pad_token: str = "<pad>", eos_token: str = "<eos>") -> Tokenizer:

    # We enable `byte_fallback=True`. If a token isn't in the vocab, it decomposes 
    # into UTF-8 bytes. `fuse_unk` compacts multiple errors into one token to protect context.
    tokenizer = Tokenizer(models.BPE(unk_token=unk_token, fuse_unk=True, byte_fallback=True))

    # Before BPE merges happen, we enforce hard boundaries at whitespace and punctuation.
    # This prevents the model from "gluing" punctuation to words, reducing vocab redundancy.
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.WhitespaceSplit(),
        pre_tokenizers.Punctuation()
    ])

    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        # Optimization: We intentionally exclude `pad_token`. Our streaming data loader
        # never generates padding, so allocating a vocab slot for it would be a waste.
        special_tokens=[eos_token, unk_token],

        # Regularization: Prune tokens that appear only once to force the model 
        # to learn generalizable subwords rather than memorizing noise.
        min_frequency=2,

        # Reliability: Pre-seed the vocab with all 256 fundamental byte values. 
        # This guarantees that the `byte_fallback` mechanism always has a valid target.
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),

        # Morphology: Adds a visual marker (e.g., "##ing") to suffixes, helping
        # the model distinguish between root words and extensions.
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
    pos_emb_type: PositionalEmbeddingType

def get_slopes(nh: int, device: torch.device = None, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """Generates a unique slope (penalty factor) for each attention head."""
        if nh == 0:
            return torch.empty((0,), dtype=dtype, device=device)
        # Linear distribution of slopes: [1/nh, 2/nh, ..., 1.0]
        # Heads with steeper slopes focus more on recent context; 
        # Heads with shallower slopes have a broader effective receptive field.
        slopes = torch.arange(1, nh + 1, dtype=torch.float32, device=device) / nh
        return slopes.to(dtype)

def build_alibi_mask(n_head: int, max_len: int) -> torch.Tensor:
    """
    Constructs a static bias mask for Attention with Linear Biases (ALiBi).
    
    ALiBi adds a non-learnable bias to attention scores based on the distance 
    between the query and the key tokens. This allows the model to:
    1. Extrapolate to sequences longer than those seen during training.
    2. Focus on local context without needing rotary or absolute embeddings.
    
    Note: This specific implementation uses a LINEAR slope distribution 
    (1/n, 2/n... 1.0) rather than the GEOMETRIC distribution (1/2^1, 1/2^2...) 
    proposed in the original Press et al. paper.
    
    Returns:
        Tensor of shape [n_head, max_len, max_len] containing bias values.
    """
    dtype = torch.float32
    device = None # Created on CPU first to save GPU memory during initialization

    slopes = get_slopes(n_head, device, dtype)
    
    # Create the distance matrix via broadcasting
    arange = torch.arange(max_len, dtype=dtype, device=device)
    
    # Query positions (Row indices): Shape [1, max_len, 1]
    i_pos = arange[None, :, None]
    
    # Key positions (Column indices): Shape [1, 1, max_len]
    j_pos = arange[None, None, :]
    
    # Calculate relative distance: (j - i)
    # For causal attention, we only care where j <= i (past tokens).
    # This results in negative values (penalties).
    # Example: i=5 (current), j=3 (past) -> dist = -2
    dist = j_pos - i_pos
    
    # Apply slopes across heads
    # slopes: [n_head, 1, 1]
    # dist:   [1, max_len, max_len]
    # result: [n_head, max_len, max_len]
    bias = slopes[:, None, None] * dist
    return bias

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """
    Precomputes the complex exponentials (cis = cos + i*sin) for RoPE.
    
    Instead of computing cos/sin during every forward pass, we cache the values
    representing the rotation angles.
    
    Args:
        dim: The dimension of the attention head (must be even).
        end: The maximum sequence length (context window).
        theta: The base frequency scaling factor (typically 10000.0).
    """
    # 1. Calculate frequencies: 1 / theta^(2i/d)
    # We only need dim/2 frequencies because each frequency rotates a pair of numbers.
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    
    # 2. Create position indices: [0, 1, ..., end-1]
    t = torch.arange(end, device=freqs.device)
    
    # 3. Outer product: position * frequency
    # Shape: [end, dim/2] -> represents the angle 'm * theta' for each position.
    freqs = torch.outer(t, freqs).float()
    
    # 4. Convert to complex form: e^(i * angle) = cos(angle) + i*sin(angle)
    # torch.polar(abs, angle) creates complex numbers with magnitude 1.
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis

def apply_rotary_emb(xq, xk, freqs_cis):
    """
    Applies the Rotary Positional Embedding to Query and Key states.
    
    This effectively rotates the Q and K vectors by an amount corresponding 
    to their position in the sequence, allowing the attention mechanism to 
    understand relative distances.
    
    Args:
        xq: Query states [Batch, SeqLen, n_head, head_dim]
        xk: Key states   [Batch, SeqLen, n_head, head_dim]
        freqs_cis: Precomputed complex exponentials [MaxSeqLen, head_dim/2]
    """
    # 1. Reshape real inputs to look like complex pairs
    # Shape: [B, T, H, D] -> [B, T, H, D/2, 2] -> [B, T, H, D/2] (Complex)
    # We group adjacent elements to form real and imaginary parts.
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    
    # 2. Align frequencies with the current sequence length
    freqs_cis = freqs_cis[:xq.shape[1]].to(xq_.device)
    
    # 3. Reshape frequencies for broadcasting
    # freqs_cis: [T, D/2] -> [1, T, 1, D/2]
    # This allows the same rotation to apply across all batches and heads.
    freqs_cis = freqs_cis.view(1, xq.shape[1], 1, xq.shape[3]//2)
    
    # 4. Apply Rotation via Complex Multiplication
    # (a+ib) * (cos+isin) rotates the vector (a,b).
    # This is mathematically equivalent to the standard RoPE matrix multiplication 
    # but computationally faster.
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    
    return xq_out.type_as(xq), xk_out.type_as(xk)

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (RMSNorm).
    
    A simplified and computationally efficient alternative to LayerNorm used in modern 
    LLMs (e.g., LLaMA, Gopher). 
    
    unlike LayerNorm, RMSNorm does not re-center the mean to zero; it only re-scales 
    invariance. This reduces computational overhead while maintaining convergence stability.
    
    Formula:
        RMS(x) = sqrt(mean(x^2) + eps)
        output = (x / RMS(x)) * weight
    """
    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        # The learnable scaling parameter (gamma). 
        # Unlike LayerNorm, RMSNorm typically does not use an additive bias (beta).
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # 1. Calculate Root Mean Square
        # Square the input, take the mean across the last dimension, add epsilon for stability, then sqrt.
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
        
        # 2. Normalize and Scale
        # Project input onto the unit hypersphere and scale by the learned weights.
        return x / rms * self.weight

class CausalSelfAttention(nn.Module):
    """
    Multi-Head Causal Self-Attention mechanism.
    
    This layer implements the core 'scaled dot-product attention' with two 
    modern positional embedding variants:
    1. RoPE (Rotary Positional Embeddings): Applied to Q and K vectors.
    2. ALiBi (Attention with Linear Biases): Added to the attention scores.
    """
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_head == 0, "Model dimension must be divisible by number of heads"
        
        self.head_dim = cfg.d_model // cfg.n_head
        self.n_head   = cfg.n_head
        self.pos_emb_type = cfg.pos_emb_type

        # 1. Positional Bias Registration (ALiBi Only)
        if self.pos_emb_type == PositionalEmbeddingType.ALIBI:
            # We register the mask as a buffer so it is saved with the model state_dict
            # but is not updated by the optimizer (requires_grad=False).
            self.register_buffer("alibi_bias", build_alibi_mask(cfg.n_head, cfg.block_size).to(torch.float32))

        # 2. Projection Layers
        # Project input to Query (Q)
        self.q_proj = nn.Linear(cfg.d_model, self.n_head * self.head_dim)
        # Project input to Key (K) and Value (V) combined
        self.kv_proj = nn.Linear(cfg.d_model, 2 * self.n_head * self.head_dim)
        # Final output projection
        self.o_proj = nn.Linear(self.n_head * self.head_dim, cfg.d_model)

        # 3. Regularization
        self.attn_dropout_p = cfg.dropout # Droput applied to attention probabilities
        self.resid_dropout = nn.Dropout(cfg.dropout) # Dropout applied to final output

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor):
        B, T, C = x.size() # Batch, SeqLen, Dimensions

        # 1. Q Projection & Reshape
        # [B, T, C] -> [B, T, n_head, head_dim] -> [B, n_head, T, head_dim]
        # We permute to isolate heads for parallel computation.
        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).permute(0, 2, 1, 3)

        # 2. K, V Projection & Reshape
        # [B, T, C] -> [B, T, 2, n_head, head_dim] -> [2, B, n_head, T, head_dim]
        kv = self.kv_proj(x).view(B, T, 2, self.n_head, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0) # Split into K and V

        # 3. Rotary Positional Embeddings (RoPE)
        # If enabled, rotate Q and K vectors based on their sequence position.
        # This happens *before* the dot product.
        if self.pos_emb_type == PositionalEmbeddingType.ROPE:
            q, k = apply_rotary_emb(q, k, freqs_cis)

        # 4. Attention Scores (Scaled Dot-Product)
        # (B, H, T, D) @ (B, H, D, T) -> (B, H, T, T)
        # We divide by sqrt(head_dim) to stabilize gradients (standard Scaling).
        attn_scores = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))

        # 5. ALiBi Injection
        # If enabled, add static bias penalties based on relative token distance.
        if self.pos_emb_type == PositionalEmbeddingType.ALIBI:
            # Slice to current sequence length T to handle variable input sizes
            bias = self.alibi_bias[:, :T, :T].unsqueeze(0).to(attn_scores.device) 
            attn_scores = attn_scores + bias

        # 6. Causal Masking
        # Apply triangular mask to ensure tokens can only attend to past positions.
        # Upper triangle (future) is filled with -infinity (becomes 0 in softmax).
        mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        attn_scores = attn_scores.masked_fill(mask, float('-inf'))
        
        # 7. Softmax & Dropout
        # Convert scores to probabilities (Weights)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = F.dropout(attn_weights, p=self.attn_dropout_p if self.training else 0)
        
        # 8. Aggregation
        # Weighted sum of Values: (B, H, T, T) @ (B, H, T, D) -> (B, H, T, D)
        y = attn_weights @ v
        
        # Re-assemble heads: [B, H, T, D] -> [B, T, H, D] -> [B, T, C]
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        # 9. Output Projection & Residual Dropout
        return self.resid_dropout(self.o_proj(y))

class SwiGLU(nn.Module):
    """
    SwiGLU: A Gated Linear Unit variant with Swish (SiLU) activation.
    
    This replaces the standard MLP (Linear -> GELU -> Linear) found in older 
    Transformers (like GPT-2/BERT). 
    
    Mechanism:
    It projects the input into two parallel paths:
    1. A "Gate" path (activated by SiLU).
    2. A "Value" path (linear).
    The two paths are multiplied element-wise before being projected down.
    """
    def __init__(self, cfg: GPTConfig, layer_depth: int):
        super().__init__()

        # Architectural Choice:
        # While standard SwiGLU implementations (like LLaMA) often reduce this factor 
        # to ~2.67 to match the parameter count of a standard GELU MLP, we strictly 
        # use an expansion factor of 6.
        #
        # Empirically, this "Wider" configuration maximizes representational capacity,
        # yielding the lowest validation loss for this specific dataset size, 
        # prioritizing performance over parameter efficiency.
        hidden_dim = int(cfg.d_model * cfg.expansion_factor)
        
        # LLaMA-style naming convention:
        # 1. gate_proj: The "Switch" (determines which features pass through).
        self.gate_proj = nn.Linear(cfg.d_model, hidden_dim, bias=False)
        
        # 2. up_proj: The "Content" (projects input up to the hidden representation).
        self.up_proj = nn.Linear(cfg.d_model, hidden_dim, bias=False)
        
        # 3. down_proj: The "Output" (projects combined features back to model dim).
        self.down_proj = nn.Linear(hidden_dim, cfg.d_model, bias=False)
        
        self.dropout = nn.Dropout(cfg.dropout)
        
        # Depth Scaling:
        # This technique (from DeepNet/CogView) stabilizes gradients in deeper networks.
        self.output_scale = 1 / math.sqrt(layer_depth)

    def forward(self, x):
        # 1. Calculate the Gate (0 to 1-ish activation)
        gate = F.silu(self.gate_proj(x))
        
        # 2. Calculate the Raw Features
        features = self.up_proj(x)
        
        # 3. Apply Gating (Element-wise multiplication)
        # The gate selectively amplifies or suppresses specific features.
        x = gate * features
        
        # 4. Down-Project and Scale
        x = self.down_proj(x) * self.output_scale
        x = self.dropout(x)
        return x

class SparseMoE(nn.Module):
    """
    Sparse Mixture of Experts (MoE) Layer.
    
    Implements a Top-K Router that dynamically routes tokens to a subset of 
    expert networks (SwiGLU blocks).
    
    Benefits:
    - Scales model parameters significantly without increasing inference FLOPs.
    - Top-K routing ensures sparsity (only K experts are active per token).
    
    Mechanism:
    1. A 'Router' (Linear) predicts which experts are best for a given token.
    2. Gaussian noise is added during training (Jitter) to encourage load balancing.
    3. The top-k experts are selected, and their outputs are weighted-summed.
    """
    def __init__(self, cfg: GPTConfig, layer_depth: int):
        super().__init__()
        self.num_experts = cfg.num_experts
        self.top_k = cfg.top_k
        
        # The Gating Network (Router)
        # Maps token representations to 'num_experts' scores.
        self.router = nn.Linear(cfg.d_model, self.num_experts, bias=False)
        
        # The Experts
        # A collection of independent feed-forward networks (SwiGLU).
        self.experts = nn.ModuleList([SwiGLU(cfg, layer_depth) for _ in range(self.num_experts)])

        # Jitter Noise Standard Deviation
        # Standard technique to prevent "Router Collapse" (where the router 
        # always picks the same experts, leaving others untrained).
        self.jitter_std = 0.1 

    def forward(self, x):
        B, T, C = x.shape
        # Flatten batch and sequence to treat every token independently
        flat_x = x.view(-1, C)
        
        # 1. Calculate Router Logits
        router_logits = self.router(flat_x)

        if self.training:
            # Inject Jitter (Noise)
            # We add standard normal noise to the logits to ensure exploration.
            noise = torch.randn_like(router_logits) * self.jitter_std
            router_logits = router_logits + noise
        
        # 2. Select Top-K Experts
        # router_values: The raw logit scores of the chosen experts
        # expert_indices: The IDs (0 to num_experts-1) of the chosen experts
        router_values, expert_indices = torch.topk(router_logits, self.top_k, dim=-1)
        
        # 3. Calculate Gate Probabilities
        # We apply Softmax ONLY to the top-k values, not the full list.
        # This ensures the weights sum to 1.0 among the active experts.
        gate_probs = F.softmax(router_values, dim=-1, dtype=torch.float).to(x.dtype)
        
        # Initialize output tensor
        final_output = torch.zeros_like(flat_x)
        
        # 4. Expert Dispatch & Aggregation
        # We iterate through the k ranks (e.g., 1st best expert, 2nd best expert).
        # Note: In production (Megatron/Deepspeed), this is usually done via 
        # Permutation/Unpermutation scatters, but looping is clearer for logic.
        for k in range(self.top_k):
            # Which expert was selected for the k-th rank for each token?
            selected_expert_idxs = expert_indices[:, k]
            
            # The weighting factor for this rank
            gate_weight = gate_probs[:, k].unsqueeze(-1)
            
            # Iterate over all physical experts to find tokens assigned to them
            for i, expert in enumerate(self.experts):
                # Boolean Mask: Is token 't' assigned to expert 'i' at rank 'k'?
                expert_mask = (selected_expert_idxs == i)
                
                if expert_mask.any():
                    # Extract tokens assigned to this expert
                    tokens_for_expert = flat_x[expert_mask]
                    
                    # Process tokens
                    expert_out = expert(tokens_for_expert)
                    
                    # Weighted Accumulation
                    # output += (prob * expert_output)
                    final_output[expert_mask] += gate_weight[expert_mask] * expert_out
                    
        return final_output.view(B, T, C)

class Block(nn.Module):
    """
    A single Transformer Decoder Block using the Pre-Norm architecture.
    
    Structure:
        x = x + Attention(RMSNorm(x))
        x = x + MoE(RMSNorm(x))
        
    This 'Pre-Norm' design (normalizing *before* the layer) is preferred over 
    Post-Norm (original BERT/GPT) because it creates a clear gradient path 
    down the residual stream, significantly improving training stability 
    for deep networks.
    """
    def __init__(self, cfg: GPTConfig, layer_depth: int):
        super().__init__()
        
        # 1. Attention Block Components
        # 'input_norm' is applied before the Self-Attention layer.
        self.input_norm = RMSNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)

        # 2. Feed-Forward Block Components
        # 'post_attn_norm' is applied before the Mixture-of-Experts layer.
        self.post_attn_norm = RMSNorm(cfg.d_model)
        
        # We replace the standard dense MLP with a Sparse Mixture of Experts.
        # This increases total parameters (knowledge capacity) while keeping
        # active parameters (inference cost) low.
        self.moe = SparseMoE(cfg, layer_depth)

    def forward(self, x, freqs_cis: torch.Tensor = None):
        # 1. Self-Attention Sublayer
        # Note the Residual connection (x + ...). 
        # The gradients flow directly through the '+' without passing through the norm.
        h = self.input_norm(x)
        x = x + self.attn(h, freqs_cis)
        
        # 2. Mixture-of-Experts Sublayer
        h = self.post_attn_norm(x)
        x = x + self.moe(h)
        return x

class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.drop      = nn.Dropout(cfg.dropout)
        self.blocks    = nn.ModuleList([Block(cfg, i+1) for i in range(cfg.n_layer)])
        self.ln_f      = RMSNorm(cfg.d_model)
        self.head      = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        if cfg.pos_emb_type == PositionalEmbeddingType.ROPE:
            # RoPE constants
            self.head_dim = cfg.d_model // cfg.n_head
            self.register_buffer("freqs_cis", precompute_freqs_cis(self.head_dim, cfg.block_size), persistent=False)

        self.apply(self._init_weights)

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

        use_grad_checkpointing = self.training

        # Determine the argument to pass (RoPE only)
        freqs_cis = None
        if self.cfg.pos_emb_type == PositionalEmbeddingType.ROPE:
            freqs_cis = self.freqs_cis

        # Single unified loop
        for block in self.blocks:
            if use_grad_checkpointing:
                x = torch.utils.checkpoint.checkpoint(
                    block, x, freqs_cis, use_reentrant=False
                )
            else:
                x = block(x, freqs_cis)

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
    
    train_loader = ShuffledBlockDataLoader(train_ids, args.block_size, args.batch_size, device, args.seed)
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
        top_k = args.top_k,
        pos_emb_type = args.pos_emb_type

    )
    model = GPT(cfg).to(device)

    # Compile helps on newer PyTorch
    if hasattr(torch, 'compile'):
        try:
            model = torch.compile(model)
            logger.log("model_compiled", mode="default")
        except Exception as e:
            logger.log("compilation_failed", error=str(e))
            pass

    model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.log("model_info", parameters_count=model_params)
    
    opt = model.configure_optimizers(args.weight_decay, args.lr, args.betas, device)

    warmup_steps = int(args.warmup_frac * max_steps)
    scheduler = CosineWarmupScheduler(opt, warmup_steps, max_steps, args.min_lr)

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

    use_amp = (device == "cuda")
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    for epoch in range(1, args.epochs + 1):
        for xb, yb in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}"):
            step += 1

            with torch.amp.autocast('cuda'):
                _, loss = model(xb, yb)

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
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
