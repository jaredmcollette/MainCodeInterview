import json
import argparse
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from pathlib import Path

def parse_logs(log_file):
    data = {
        "train_steps": [], "train_loss": [], "grad_norm": [], 
        "throughput": [], "vram": [],
        "val_steps": [], "val_loss": [],
        "total_params": 0, "active_params": 0
    }

    with open(log_file, 'r') as f:
        for line in f:
            try:
                entry = json.loads(line)
                event = entry.get("event")

                if event == "model_info":
                    data["total_params"] = entry.get("total_parameters", 0)
                    data["active_params"] = entry.get("active_parameters", 0)

                elif event == "training_step":
                    data["train_steps"].append(entry["step"])
                    data["train_loss"].append(entry["loss"])
                    data["grad_norm"].append(entry["grad_norm"])
                    data["throughput"].append(entry["throughput"])
                    data["vram"].append(entry.get("vram_gb", 0))

                elif event == "validation_step":
                    data["val_steps"].append(entry["step"])
                    data["val_loss"].append(entry["loss"])

            except json.JSONDecodeError:
                continue
    return data

def generate_table(data):
    avg_throughput = np.mean(data["throughput"]) if data["throughput"] else 0
    peak_vram = np.max(data["vram"]) if data["vram"] else 0
    final_grad_norm = data["grad_norm"][-1] if data["grad_norm"] else 0
    
    print("\n" + "="*45)
    print("       TRAINING RUN ANALYTICS       ")
    print("="*45)
    print(f"{'Metric':<30} | {'Value':<15}")
    print("-" * 47)
    print(f"{'Avg Throughput (tok/sec)':<30} | {avg_throughput:.2f}")
    print(f"{'Peak VRAM Usage (GB)':<30} | {peak_vram:.2f}")
    print(f"{'Total Parameters':<30} | {data['total_params']:,}")
    print(f"{'Active Parameters':<30} | {data['active_params']:,}")
    print(f"{'Final Gradient Norm':<30} | {final_grad_norm:.4f}")
    print("="*45 + "\n")

def plot_all(data):
    fig = plt.figure(figsize=(15, 10))
    
    # 1. Loss Curves
    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(data["train_steps"], data["train_loss"], label="Train", alpha=0.5)
    ax1.plot(data["val_steps"], data["val_loss"], label="Validation", marker='o', color='red')
    ax1.set_title("Loss Curves")
    ax1.set_xlabel("Steps")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Gradient Norm
    ax2 = plt.subplot(2, 2, 2)
    ax2.plot(data["train_steps"], data["grad_norm"], color='orange', alpha=0.8)
    ax2.set_title("Gradient Norm")
    ax2.set_xlabel("Steps")
    ax2.grid(True, alpha=0.3)

    # 3. Throughput
    ax4 = plt.subplot(2, 2, 4)
    ax4.plot(data["train_steps"], data["throughput"], color='purple', alpha=0.6)
    ax4.set_title("Training Throughput")
    ax4.set_xlabel("Steps")
    ax4.set_ylabel("Tokens / Sec")
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_file", type=str, default="./logs/mainrun.log")
    args = parser.parse_args()

    if not Path(args.log_file).exists():
        print(f"Error: Log file not found at {args.log_file}")
        exit(1)

    print(f"Parsing log file: {args.log_file}...")
    metrics = parse_logs(args.log_file)
    generate_table(metrics)
    plot_all(metrics)