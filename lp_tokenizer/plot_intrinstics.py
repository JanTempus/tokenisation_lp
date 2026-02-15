import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_from_csv(lp_csv, bpe_csv, save_path_prefix="compare_tokenizers"):
    # Load CSVs
    lp_df = pd.read_csv(lp_csv)
    bpe_df = pd.read_csv(bpe_csv)

    # Sort by vocab size (just to be safe)
    lp_df = lp_df.sort_values("vocab_size")
    bpe_df = bpe_df.sort_values("vocab_size")

    x_vals = [1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]

    metrics = {
        "avg_length": "Average Length",
        "avg_fertility": "Average Fertility",
        "vocab_utilization": "Vocab Utilization"
    }

    for metric, label in metrics.items():
        plt.figure(figsize=(10,6))
        plt.plot(lp_df["vocab_size"], lp_df[metric], marker="o", label="LP Tokenizer")
        plt.plot(bpe_df["vocab_size"], bpe_df[metric], marker="s", label="BPE Tokenizer")
        
        plt.xscale("log", base=2)
        plt.xticks(x_vals, [str(v) for v in x_vals])  # powers of 2 on x-axis
        
        plt.xlabel("Vocab Size")
        plt.ylabel(label)
        plt.title(f"{label} vs Vocab Size")
        plt.grid(True, which="both", ls="--", alpha=0.6)
        plt.legend()
        
        out_path = f"{save_path_prefix}_{metric}.png"
        plt.tight_layout()
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"Saved {out_path}")


plot_from_csv("lp_stats.csv","bpe_stats.csv")