import json
import argparse
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import itertools

# --- CONFIGURATION FOR PUBLICATION QUALITY ---
PARAMS = {
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans', 'Bitstream Vera Sans'],
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'font.size': 12,
    'legend.fontsize': 11,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'figure.titlesize': 18,
    'axes.linewidth': 1.5,
    'lines.linewidth': 2.5,
    'grid.linewidth': 0.8,
    'grid.linestyle': '--',
    'grid.alpha': 0.6
}
plt.rcParams.update(PARAMS)

COLORS = [
    "#0072B2", "#D55E00", "#009E73", "#CC79A7", 
    "#F0E442", "#56B4E9", "#E69F00", "#000000"
]

def smooth_curve(points, window_size=50):
    """Smoothes data using a Centered Simple Moving Average."""
    if len(points) < window_size:
        return points
    window = np.ones(window_size) / window_size
    pad_size = window_size // 2
    padded_points = np.pad(points, (pad_size, pad_size), mode='edge')
    smoothed = np.convolve(padded_points, window, mode='valid')
    return smoothed[:len(points)]

def parse_logs(log_file):
    data = {
        "train_epochs": [], "train_loss": [], 
        "val_epochs": [], "val_loss": [],
        "grad_norm": [], "grad_norm_epochs": [], # Track epochs specifically for sparse metrics
        "throughput": [], "throughput_epochs": [],
        "batches_per_epoch": 1 
    }

    raw_train_steps = []
    raw_val_steps = []

    print(f"Parsing: {log_file}...")
    try:
        with open(log_file, 'r') as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    event = entry.get("event")

                    if event == "dataset_info":
                        data["batches_per_epoch"] = entry.get("batches_per_epoch", 1)

                    elif event == "training_step":
                        step = entry["step"]
                        raw_train_steps.append(step)
                        data["train_loss"].append(entry["loss"])
                        
                        # STRICT CHECK: Only add these if they exist in the log line
                        if "grad_norm" in entry:
                            data["grad_norm"].append(entry["grad_norm"])
                            data["grad_norm_epochs"].append(step) # Store step temporarily
                        
                        if "throughput" in entry:
                            data["throughput"].append(entry["throughput"])
                            data["throughput_epochs"].append(step)

                    elif event == "validation_step":
                        raw_val_steps.append(entry["step"])
                        data["val_loss"].append(entry["loss"])

                except json.JSONDecodeError:
                    continue
    except FileNotFoundError:
        print(f"Warning: File {log_file} not found. Skipping.")
        return None

    bpe = data["batches_per_epoch"]
    
    # Convert all steps to epochs
    data["train_epochs"] = [s / bpe for s in raw_train_steps]
    data["val_epochs"] = [s / bpe for s in raw_val_steps]
    data["grad_norm_epochs"] = [s / bpe for s in data["grad_norm_epochs"]]
    data["throughput_epochs"] = [s / bpe for s in data["throughput_epochs"]]

    return data

def style_axis(ax, title, ylabel):
    ax.set_title(title, pad=15, weight='bold')
    ax.set_xlabel("Epochs", labelpad=10)
    ax.set_ylabel(ylabel, labelpad=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, color='#e0e0e0')
    ax.tick_params(axis='both', which='major', length=5, width=1.2)

def save_graphs(results, output_dir):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # --- 1. Validation Loss (Top Left) ---
    # EXPECTATION: Plots BOTH lines (Main + Baseline)
    ax_val = axes[0, 0]
    style_axis(ax_val, "Validation Loss", "Cross Entropy Loss")
    
    color_cycle = itertools.cycle(COLORS) 
    for name, data in results.items():
        color = next(color_cycle)
        if data["val_epochs"] and data["val_loss"]:
            ax_val.plot(data["val_epochs"], data["val_loss"], 
                        label=name, color=color, alpha=0.9, marker='o', markersize=4)
    ax_val.legend(frameon=False, loc='upper right')

    # --- 2. Training Loss (Top Right) ---
    # EXPECTATION: Plots BOTH lines (Main + Baseline)
    ax_train = axes[0, 1]
    style_axis(ax_train, "Training Loss", "Cross Entropy Loss")
    
    color_cycle = itertools.cycle(COLORS) 
    for name, data in results.items():
        color = next(color_cycle)
        if data["train_epochs"] and data["train_loss"]:
            # Raw (Faint)
            ax_train.plot(data["train_epochs"], data["train_loss"], 
                          color=color, alpha=0.1, linewidth=1, zorder=1)
            # Smoothed (Solid)
            smoothed = smooth_curve(data["train_loss"], window_size=50)
            ax_train.plot(data["train_epochs"], smoothed, 
                          label=name, color=color, alpha=1.0, zorder=2)
    ax_train.legend(frameon=False, loc='upper right')

    # --- 3. Gradient Norm (Bottom Left) ---
    # EXPECTATION: Plots ONLY Main Run (Baseline skips automatically)
    ax_grad = axes[1, 0]
    style_axis(ax_grad, "Gradient Norm", "L2 Norm")
    
    color_cycle = itertools.cycle(COLORS) 
    for name, data in results.items():
        color = next(color_cycle)
        # Check if this specific log file actually had gradient norms
        if data["grad_norm_epochs"] and data["grad_norm"]:
            # Raw (Faint)
            ax_grad.plot(data["grad_norm_epochs"], data["grad_norm"], 
                         color=color, alpha=0.1, linewidth=1, zorder=1)
            # Smoothed (Solid)
            smoothed = smooth_curve(data["grad_norm"], window_size=50)
            ax_grad.plot(data["grad_norm_epochs"], smoothed, 
                         label=name, color=color, alpha=1.0, zorder=2)
            
    # Only show legend if something was actually plotted
    if ax_grad.has_data():
        ax_grad.legend(frameon=False, loc='upper right')

    # --- 4. Throughput (Bottom Right) ---
    # EXPECTATION: Plots ONLY Main Run (Baseline skips automatically)
    ax_thru = axes[1, 1]
    style_axis(ax_thru, "Training Throughput", "Tokens / Sec")
    
    color_cycle = itertools.cycle(COLORS) 
    for name, data in results.items():
        color = next(color_cycle)
        if data["throughput_epochs"] and data["throughput"]:
            # Raw only (Industry standard for throughput)
            ax_thru.plot(data["throughput_epochs"], data["throughput"], 
                         label=name, color=color, alpha=0.9, linewidth=1.5)
            
    if ax_thru.has_data():
        ax_thru.legend(frameon=False, loc='lower right')

    plt.tight_layout(pad=3.0)
    
    filename = f"{output_dir}/full_dashboard.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"High-quality dashboard saved to: {filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("log_files", type=str, nargs='+', help="List of log files to compare")
    parser.add_argument("--output_dir", type=str, default="./figures")
    args = parser.parse_args()

    all_results = {}
    for log_path in args.log_files:
        path = Path(log_path)
        data = parse_logs(str(path))
        if data:
            all_results[path.stem] = data

    if not all_results:
        print("No valid log files found.")
        exit(1)

    save_graphs(all_results, args.output_dir)
    