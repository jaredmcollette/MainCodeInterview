import json
import argparse
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def parse_logs(log_file):
    data = {
        "train_steps": [], "train_loss": [], "grad_norm": [], 
        "throughput": [], "vram": [],
        "val_steps": [], "val_loss": [],
        "parameters_count": 0,
    }

    with open(log_file, 'r') as f:
        for line in f:
            try:
                entry = json.loads(line)
                event = entry.get("event")

                if event == "model_info":
                    data["parameters_count"] = entry.get("parameters_count", 0)

                elif event == "training_step":
                    data["train_steps"].append(entry["step"])
                    data["train_loss"].append(entry["loss"])
                    data["grad_norm"].append(entry["grad_norm"])
                    data["throughput"].append(entry["throughput"])
                    data["vram"].append(entry.get("vram_gb", 0))

                elif event == "validation_step":
                    data["val_steps"].append(entry["step"])
                    data["val_loss"].append(entry["loss"])

                elif event == "final_model":
                    data["inference_latency_ms"] = entry.get("inference_latency_ms", 0)

            except json.JSONDecodeError:
                continue
    return data

def generate_table_on_axis(data, ax):
    """Generates the stats table directly onto a matplotlib axis."""
    avg_throughput = np.mean(data["throughput"]) if data["throughput"] else 0
    peak_vram = np.max(data["vram"]) if data["vram"] else 0
    final_grad_norm = data["grad_norm"][-1] if data["grad_norm"] else 0
    
    # Prepare Table Data
    columns = ["Metric", "Value"]
    rows = [
        ["Avg Throughput", f"{avg_throughput:.2f} tok/sec"],
        ["Peak VRAM Usage", f"{peak_vram:.2f} GB"],
        ["Parameter Count", f"{data['parameters_count']:,}"],
        ["Final Gradient Norm", f"{final_grad_norm:.4f}"],
        ["Inference Latency", f"{data.get('inference_latency_ms', 0):.2f} ms"],
    ]

    # Hide axes
    ax.axis('off')
    ax.axis('tight')

    # Create table
    table = ax.table(cellText=rows, colLabels=columns, loc='center', cellLoc='left', colLoc='left')
    
    # Styling
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 2) # Stretch rows vertically for readability
    
    # Header styling
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight='bold')
            cell.set_facecolor('#f0f0f0')

def save_dashboard(data, output_dir):
    # Ensure the directory exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Create a 3-row layout: 
    # Row 1: Val Loss | Grad Norm
    # Row 2: Train Loss | Throughput
    # Row 3: Table
    fig = plt.figure(figsize=(15, 14))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 0.5])

    # 1. Validation Loss (Top Left)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(data["val_steps"], data["val_loss"], marker='o', color='red', linestyle='-')
    ax1.set_title("Validation Loss")
    ax1.set_xlabel("Steps")
    ax1.set_ylabel("Cross Entropy Loss")
    ax1.grid(True, alpha=0.3)

    # 2. Gradient Norm (Top Right)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(data["train_steps"], data["grad_norm"], color='orange', alpha=0.7)
    ax2.set_title("Gradient Norm")
    ax2.set_xlabel("Steps")
    ax2.grid(True, alpha=0.3)

    # 3. Training Loss (Middle Left - Below Validation)
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(data["train_steps"], data["train_loss"], color='blue', alpha=0.4)
    ax3.set_title("Training Loss")
    ax3.set_xlabel("Steps")
    ax3.set_ylabel("Cross Entropy Loss")
    ax3.grid(True, alpha=0.3)

    # 4. Throughput (Middle Right)
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(data["train_steps"], data["throughput"], color='purple', alpha=0.6)
    ax4.set_title("Training Throughput")
    ax4.set_xlabel("Steps")
    ax4.set_ylabel("Tokens / Sec")
    ax4.grid(True, alpha=0.3)

    # 5. Table (Bottom - Spanning both columns)
    ax_table = fig.add_subplot(gs[2, :])
    generate_table_on_axis(data, ax_table)
    ax_table.set_title("Run Analytics", pad=20)

    plt.tight_layout()
    
    # Save file
    filename = f"{output_dir}/training_dashboard.png"
    plt.savefig(filename)
    plt.close()
    print(f"Dashboard saved to: {filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_file", type=str, default="./logs/mainrun.log")
    parser.add_argument("--output_dir", type=str, default="./figures")
    args = parser.parse_args()

    if not Path(args.log_file).exists():
        print(f"Error: Log file not found at {args.log_file}")
        exit(1)

    print(f"Parsing log file: {args.log_file}...")
    metrics = parse_logs(args.log_file)
    save_dashboard(metrics, args.output_dir)
