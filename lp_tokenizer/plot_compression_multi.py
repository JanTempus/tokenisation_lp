import numpy as np
import matplotlib.pyplot as plt

def compute_ratio(length_in_characters, length_in_tokens_list, optimal_length_in_tokens):
    length_in_tokens = np.array(length_in_tokens_list)
    return (length_in_characters - length_in_tokens) / (length_in_characters - optimal_length_in_tokens)

def plot_ratio(length_in_characters, optimal_length_in_tokens, length_in_tokens_lists, 
               labels=None, list_labels=None, save_path="ratio_plot_multiple.png"):
    """
    length_in_tokens_lists: list of lists, e.g., [list1, list2]
    list_labels: optional labels for each list for the legend, e.g., ["Tokenizer A", "Tokenizer B"]
    """
    
    plt.figure(figsize=(12,6))
    
    markers = ['o', 's', 'D', '^', 'v', '*']  # extend as needed
    colors = ['blue', 'green', 'orange', 'purple', 'cyan', 'magenta']
    
    for i, length_in_tokens_list in enumerate(length_in_tokens_lists):
        ratios = compute_ratio(length_in_characters, length_in_tokens_list, optimal_length_in_tokens)
        # x-axis values
        if labels is not None:
            x_vals = np.arange(len(labels))
            plt.xticks(x_vals, labels, rotation=45)
        else:
            x_vals = length_in_tokens_list
        
        label = list_labels[i] if list_labels is not None else f"List {i+1}"
        plt.plot(x_vals, ratios, marker=markers[i % len(markers)], linestyle='-', color=colors[i % len(colors)], label=label)
    
    plt.axhline(1.0, color='red', linestyle='--', label='Optimal Compression')
    plt.xlabel('Vocab Size')
    plt.ylabel('Ratio')
    plt.title('Ratio vs Vocab Size')
    plt.grid(True)
    plt.legend()

    # Save the plot as PNG
    plt.tight_layout(pad=3.0)
    plt.savefig(save_path, dpi=300)
    plt.close()
    
    print(f"Plot saved to {save_path}")
    
    # Return ratios for each list
    return [compute_ratio(length_in_characters, lst, optimal_length_in_tokens) for lst in length_in_tokens_lists]

# Example usage:

labels = ['1024','2048','4096','8192','16384','32768','65536','131072']
length_in_characters = 300473343
optimal_length_in_tokens = 59137915

#length_list1 = [131480702,106045791,91009417,80784151,73686126,69244651,66614105,65115513]
#length_list2 = [127298703,108604473,94000859,83045180,75335536,70286495,67199170,65434969]


length_list1 = [121902480,98341232,84520045,75184926,68670458,64625338,62281055,60992497]
length_list2 = [118024471,100617185,87213091,77181972,70157403,65545202,62778151,61233441]




ratios = plot_ratio(
    length_in_characters,
    optimal_length_in_tokens,
    [length_list1, length_list2],
    labels=labels,
    list_labels=["LP Tokenizer", "BPE Tokenizer"],
    save_path="token_ratio_two_val.png"
)


