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


"""
lp_32768_finewebedu_data,987.1044006347656,0.21454550805076722,0.99822998046875,32768,65536
lp_2048_finewebedu_data,1501.568115234375,0.32565565742471014,0.99609375,2048,65536
lp_1024_finewebedu_data,1861.084228515625,0.4039568375465367,0.9921875,1024,65536
lp_131072_finewebedu_data,931.6716461181641,0.20311215373003555,0.9598007202148438,131072,65536
lp_65536_finewebedu_data,951.3334808349609,0.20712506172618111,0.99151611328125,65536,65536
lp_8192_finewebedu_data,1148.2309265136719,0.2488487020909445,0.9989013671875,8192,65536
lp_4096_finewebedu_data,1290.6735382080078,0.27967034907876737,0.998046875,4096,65536
lp_16384_finewebedu_data,1048.8280334472656,0.22757406081178227,0.99920654296875,16384,65536
"""

"""
tokenizer,avg_length,avg_fertility,vocab_utilization,vocab_size,dataset_size
bpe_1024_finewebedu,1809.9158172607422,0.3954412562413315,0.99609375,1024,65536
bpe_131072_finewebedu,939.3526458740234,0.20584332501608738,0.9717864990234375,131072,65536
bpe_4096_finewebedu,1336.7750701904297,0.2912116645732929,0.9990234375,4096,65536
bpe_16384_finewebedu,1075.5281219482422,0.23437663192416885,0.99945068359375,16384,65536
bpe_32768_finewebedu,1005.1466522216797,0.21945310850622599,0.998687744140625,32768,65536
bpe_8192_finewebedu,1183.7142791748047,0.2580275329469369,0.9993896484375,8192,65536
bpe_65536_finewebedu,962.9247131347656,0.210667368768464,0.994232177734375,65536,65536
bpe_2048_finewebedu,1541.3043975830078,0.3358135636091898,0.998046875,2048,65536
"""

ratios = plot_ratio(
    length_in_characters,
    optimal_length_in_tokens,
    [length_list1, length_list2],
    labels=labels,
    list_labels=["LP Tokenizer", "BPE Tokenizer"],
    save_path="token_ratio_two_val.png"
)


