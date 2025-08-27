import numpy as np
import matplotlib.pyplot as plt

def compute_ratio(length_in_characters, length_in_tokens_list, optimal_length_in_tokens):
    length_in_tokens = np.array(length_in_tokens_list)
    return (length_in_characters - length_in_tokens) / (length_in_characters - optimal_length_in_tokens)

def plot_ratio(length_in_characters, optimal_length_in_tokens, length_in_tokens_list, 
               labels=None, save_path="ratio_plot.png"):
   
    ratios = compute_ratio(length_in_characters, length_in_tokens_list, optimal_length_in_tokens)
    
    plt.figure(figsize=(12,6))
    
    # If labels are provided, use them as x-axis
    if labels is not None:
        x_vals = np.arange(len(labels))
        plt.xticks(x_vals, labels, rotation=45)
    else:
        x_vals = length_in_tokens_list
    
    plt.plot(x_vals, ratios, marker='o', linestyle='-')
    plt.axhline(1.0, color='red', linestyle='--', label='Optimal Compression')
    plt.xlabel('Labels')
    plt.ylabel('Ratio')
    plt.title('Ratio vs Labels')
    plt.grid(True)
    plt.legend()

    # Save the plot as PNG
    plt.tight_layout(pad=3.0)
    plt.savefig(save_path, dpi=300)
    plt.close()
    
    print(f"Plot saved to {save_path}")
    return ratios

# Example usage:

labels = ['1024','2048','4096','8192','16384','32768','65536','131072']
length_in_characters = 321532328
optimal_length_in_tokens = 63540117
length_in_tokens_list = [131480702,106045791,91009417,80784151,73686126,69244651,66614105,65115513]

ratios = plot_ratio(length_in_characters, optimal_length_in_tokens, length_in_tokens_list, 
                    labels, save_path="token_ratio_labeled.png")

