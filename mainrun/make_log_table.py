import json
import argparse
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# --- VISUALIZATION SETTINGS ---
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans'],
    'font.size': 12,
    'savefig.dpi': 300
})

def parse_single_log(log_file):
    """
    Parses a single log file and aggregates data into:
    1. config: Static settings (hyperparams, architecture)
    2. results: Dynamic results (losses, throughput, vram)
    """
    config = {}
    results = {
        "train_loss": [], 
        "val_loss": [], 
        "grad_norm": [], 
        "throughput": [], 
        "vram": [],
        "latency": None,
        "params": 0
    }

    print(f"Parsing: {log_file}...")
    try:
        with open(log_file, 'r') as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    event = entry.get("event")

                    if event == "hyperparameters_configured":
                        config.update(entry)
                        for k in ["event", "timestamp", "log_file", "seed", "evals_per_epoch"]:
                            config.pop(k, None)

                    elif event == "model_info":
                        results["params"] = entry.get("parameters_count", 0)

                    elif event == "training_step":
                        results["train_loss"].append(entry["loss"])
                        results["grad_norm"].append(entry["grad_norm"])
                        results["throughput"].append(entry["throughput"])
                        results["vram"].append(entry.get("vram_gb", 0))

                    elif event == "validation_step":
                        results["val_loss"].append(entry["loss"])
                    
                    elif event == "final_model":
                        results["latency"] = entry.get("inference_latency_ms", 0)

                except json.JSONDecodeError:
                    continue
    except FileNotFoundError:
        print(f"Error: File {log_file} not found.")
        exit(1)
        
    return config, results

def create_table_image(data_rows, title, filename, scale_factor=2.0):
    """
    Generates a high-quality table image.
    args:
        scale_factor: Controls the vertical spacing. 
                      Higher = more space (good for few rows).
                      Lower = compact (good for many rows).
    """
    # Estimate Base Height based on scale factor to ensure figure fits content
    # Heuristic: base_height * scale * num_rows
    estimated_height = (len(data_rows) + 2) * (scale_factor * 0.25)
    
    fig = plt.figure(figsize=(8, estimated_height))
    ax = fig.add_subplot(111)
    ax.axis('off')

    # Create the table
    table = ax.table(
        cellText=data_rows,
        colLabels=["Parameter", "Value"],
        loc='center',
        cellLoc='left',
        colLoc='left'
    )

    # Styling
    table.auto_set_font_size(False)
    table.set_fontsize(13)
    
    # Apply the specific scale factor requested
    table.scale(1, scale_factor) 

    # Loop through cells to apply specific styling
    for (row, col), cell in table.get_celld().items():
        cell.set_linewidth(0) # Remove default borders
        
        # Header Row
        if row == 0:
            cell.set_text_props(weight='bold', color='white', fontsize=14)
            cell.set_facecolor('#333333') # Dark Gray
            # Header needs to be slightly taller than data rows usually
            cell.set_height(cell.get_height() * 1.2) 
        else:
            cell.set_edgecolor('#dddddd') 
            cell.set_linewidth(0.5)
            
            # Bold the first column (Metric Name)
            if col == 0:
                cell.set_text_props(weight='bold', color='#333333')
            
            # Zebra striping
            if row % 2 == 0:
                cell.set_facecolor('#f9f9f9')
            else:
                cell.set_facecolor('white')

    plt.title(title, pad=15, fontsize=16, weight='bold', color='#333333')
    plt.tight_layout()
    
    plt.savefig(filename, bbox_inches='tight', pad_inches=0.2)
    plt.close()
    print(f"Saved: {filename} (Scale: {scale_factor})")

def process_and_save(log_file, output_dir):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    config, results = parse_single_log(log_file)
    
    # --- TABLE 1: Configuration & Architecture ---
    config_rows = [
        ["Model Dimension", config.get("d_model")],
        ["Layers", config.get("n_layer")],
        ["Heads", config.get("n_head")],
        ["Block Size (Seq Len)", config.get("block_size")],
        ["Vocab Size", f"{config.get('vocab_size'):,}"],
        ["MoE Experts", config.get("num_experts")],
        ["MoE Top-K", config.get("top_k")],
        ["Positional Embedding", config.get("pos_emb_type", "").upper()],
        ["Batch Size", config.get("batch_size")],
        ["Learning Rate", config.get("lr")],
        ["Dropout", config.get("dropout")],
        ["Epochs", config.get("epochs")],
    ]
    # Use tighter scale (2.0) for the long config table
    create_table_image(config_rows, "Hyperparameter Configuration", 
                       f"{output_dir}/table_config.png", scale_factor=2.0)

    # --- TABLE 2: Performance Results ---
    avg_throughput = np.mean(results["throughput"]) if results["throughput"] else 0
    peak_vram = np.max(results["vram"]) if results["vram"] else 0
    final_grad_norm = results["grad_norm"][-1] if results["grad_norm"] else 0
    final_train_loss = results["train_loss"][-1] if results["train_loss"] else 0
    final_val_loss = results["val_loss"][-1] if results["val_loss"] else 0
    latency = results["latency"] if results["latency"] else 0
    
    results_rows = [
        ["Total Parameters", f"{results['params']:,}"],
        ["Final Validation Loss", f"{final_val_loss:.4f}"],
        ["Final Training Loss", f"{final_train_loss:.4f}"],
        ["Final Gradient Norm", f"{final_grad_norm:.4f}"],
        ["Avg Throughput", f"{avg_throughput:.2f} tokens/sec"],
        ["Peak VRAM Usage", f"{peak_vram:.2f} GB"],
        ["Inference Latency", f"{latency:.2f} ms"],
    ]
    # Use looser scale (2.8) for the short results table
    create_table_image(results_rows, "Training Results", 
                       f"{output_dir}/table_results.png", scale_factor=2.8)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("log_file", type=str, help="Path to the log file")
    parser.add_argument("--output_dir", type=str, default="./figures")
    args = parser.parse_args()

    process_and_save(args.log_file, args.output_dir)
    