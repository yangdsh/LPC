import matplotlib.pyplot as plt
import numpy as np
import os

# --- Data ---
# Organize data by context length
# Format: {context_length: {'prompts': [prompt_lengths], 'miss': [miss_ttfts], 'hit': [hit_ttfts]}}
data = {
    3000: {
        'prompts': [50, 500, 1000],
        'miss': [373, 360, 395],
        'hit': [95, 100, 165]
    },
    #1500: {
    #    'prompts': [50, 500, 1000],
    #    'miss': [226, 280, 361],
    #    'hit': [92, 96, 160]
    #},
    1000: {
        'prompts': [50, 500, 1000],
        'miss': [160, 220, 283],
        'hit': [93, 101, 160]
    }
}

# Context lengths to plot (determines the order of subplots)
context_lengths = [1000, 3000]

# --- Plotting Setup ---
# Create figure and subplots (1 row, 3 columns)
# sharey=True ensures all subplots use the same y-axis scale for easy comparison
fig, axes = plt.subplots(1, len(context_lengths), figsize=(7, 3.5), sharey=True)

# Bar width
bar_width = 0.35

# --- Generate Plots ---
for i, context_len in enumerate(context_lengths):
    ax = axes[i] # Get the current subplot axis
    context_data = data[context_len]
    prompts = context_data['prompts']
    miss_ttfts = context_data['miss']
    hit_ttfts = context_data['hit']
    print(context_len)
    for i in range(len(prompts)):
        print(hit_ttfts[i] / miss_ttfts[i])

    # X-axis positions for the groups
    x_positions = np.arange(len(prompts))

    # Plot the bars for 'Context Miss'
    rects1 = ax.bar(x_positions - bar_width/2, miss_ttfts, bar_width, label='Context Miss', color='tab:blue', alpha=0.8)

    # Plot the bars for 'Context Hit'
    rects2 = ax.bar(x_positions + bar_width/2, hit_ttfts, bar_width, label='Context Hit', color='tab:orange', alpha=0.8)

    # --- Subplot Styling ---
    # Set title for each subplot
    ax.set_title(f'Context: {context_len} tokens')

    # Set x-axis tick labels
    ax.set_xticks(x_positions)
    ax.set_xticklabels([f'{p} tokens' for p in prompts]) # Label with prompt lengths

    # Set x-axis label
    ax.set_xlabel('New Prompt Length')

    # Add grid lines (optional)
    ax.grid(True, linestyle='--', alpha=0.6, axis='y')
    ax.set_axisbelow(True) # Put grid behind bars

    # Add value labels on top of bars (optional, can make plot busy)
    # ax.bar_label(rects1, padding=3, fontsize=8)
    # ax.bar_label(rects2, padding=3, fontsize=8)

# --- Overall Figure Styling ---
# Set y-axis label only for the first (leftmost) subplot
axes[0].set_ylabel('TTFT (ms)')

# Add a single legend for the whole figure, placed outside the plots
# Adjust bbox_to_anchor and loc to position the legend appropriately
handles, labels = axes[0].get_legend_handles_labels() # Get handles/labels from one subplot
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.21, 0.9), ncol=1)

# Adjust layout to prevent labels/titles overlapping and make space for legend
plt.tight_layout(rect=[0, 0.05, 1, 1]) # rect=[left, bottom, right, top] leaves space at bottom

# --- Save Figure ---
output_filename = "fig/ttft_grouped_bar_chart.png"
output_dir = os.path.dirname(output_filename)
if output_dir and not os.path.exists(output_dir):
    try:
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")
    except OSError as e:
        print(f"Error creating directory {output_dir}: {e}")

try:
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_filename}")
except Exception as e:
    print(f"Error saving figure: {e}")
