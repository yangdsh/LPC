import matplotlib.pyplot as plt
import pandas as pd
import io
import os
import numpy as np

# Data measured in microbenchmark
csv_data = """hit_ratio,time
55,7.17
46,8.44
36,9.8
27,11.08
18,12.2
8,13.45
0,14.66
"""

def interpolate_y(x_target, x_data, y_data):
    """
    Linearly interpolates the y-value for a given x_target based on sorted x_data and y_data.
    Assumes x_data is sorted.
    """
    if not len(x_data) or not len(y_data): # Handle empty data
        return None
    if len(x_data) != len(y_data): # Data length mismatch
        return None

    if x_target <= x_data[0]:
        if x_target == x_data[0]: return y_data[0]
        if len(x_data) > 1 and x_data[1] != x_data[0]: # Check for distinct points for extrapolation
             return y_data[0] + (x_target - x_data[0]) * (y_data[1] - y_data[0]) / (x_data[1] - x_data[0])
        return y_data[0] # Return first y if only one point or cannot extrapolate

    if x_target >= x_data[-1]:
        if x_target == x_data[-1]: return y_data[-1]
        if len(x_data) > 1 and x_data[-1] != x_data[-2]: # Check for distinct points for extrapolation
            return y_data[-2] + (x_target - x_data[-2]) * (y_data[-1] - y_data[-2]) / (x_data[-1] - x_data[-2])
        return y_data[-1] # Return last y if only one point or cannot extrapolate

    for i in range(len(x_data) - 1):
        # Ensure x_data[i] and x_data[i+1] are in order for correct segment identification
        # This assumes x_data is sorted, which it is in this script (hit_ratio_sorted)
        if x_data[i] <= x_target <= x_data[i+1]:
            if x_data[i+1] == x_data[i]: # Avoid division by zero
                return y_data[i]
            y_target = y_data[i] + (x_target - x_data[i]) * \
                       (y_data[i+1] - y_data[i]) / (x_data[i+1] - x_data[i])
            return y_target
    return None # Should ideally not be reached if x_data is sorted and x_target is within bounds

def plot_throughput_vs_hit_ratio(data_string, output_filename="fig/throughput_vs_hit_ratio_annotated.png"):
    """
    Plots Throughput (100/time) vs. Hit Ratio from a CSV string,
    with vertical dashed lines and annotations.

    Args:
        data_string (str): A string containing the CSV data.
        output_filename (str): The path where the plot image will be saved.
    """
    try:
        df = pd.read_csv(io.StringIO(data_string))
    except Exception as e:
        print(f"Error reading data: {e}")
        return

    if 'time' not in df.columns or 'hit_ratio' not in df.columns:
        print("Error: CSV data must contain 'time' and 'hit_ratio' columns.")
        return

    hit_ratio_values = df['hit_ratio'].to_numpy()
    time_values = df['time'].to_numpy()

    epsilon = 1e-9
    throughput_values = 100 / (time_values + epsilon)

    sort_indices = np.argsort(hit_ratio_values)
    hit_ratio_sorted = hit_ratio_values[sort_indices]
    throughput_sorted = throughput_values[sort_indices]

    plt.figure(figsize=(3.5, 2.7))

    # Changed marker color to orange and marker style
    plt.plot(hit_ratio_sorted, throughput_sorted, marker='o', markerfacecolor='green', markeredgecolor='darkgreen', color='tab:green', linestyle='-', label='Prefilling', zorder=5)

    # --- Vertical Lines and Annotations ---
    annotation_x_values = [34.5, 41.5]
    # Define y-offsets for annotations to avoid overlap, can be adjusted
    annotation_y_offsets = [-1.2, -0.8] # Adjusted for better spacing
    annotation_x_offsets = [0, 1]       # Horizontal offset

    for i, x_val in enumerate(annotation_x_values):
        plt.axvline(x=x_val, color='gray', linestyle='--', linewidth=1, zorder=1)
        y_intersect = interpolate_y(x_val, hit_ratio_sorted, throughput_sorted)

        if y_intersect is not None:
            # Annotate only the y-value
            annotation_text = f'{y_intersect:.2f}'
            # Use predefined offsets
            text_x_offset = annotation_x_offsets[i]
            text_y_offset = annotation_y_offsets[i]

            plt.annotate(annotation_text,
                         xy=(x_val, y_intersect),
                         xytext=(x_val + text_x_offset, y_intersect + text_y_offset),
                         textcoords='data',
                         #arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2", color='black'),
                         fontsize=8,
                         # bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", lw=0.5, alpha=0.9),
                         zorder=10,
                         ha='left', va='bottom') # Horizontal and vertical alignment
            # Plot a small marker at the intersection point
            plt.plot(x_val, y_intersect, marker='o', color='black', markersize=3, zorder=6, markeredgewidth=1.5)

    plt.xlim(left=-5, right=max(hit_ratio_sorted) + 10 if len(hit_ratio_sorted) > 0 else 60)
    plt.ylim(bottom=0, top=max(throughput_sorted) * 1.15 if len(throughput_sorted) > 0 else 20) # Adjusted top ylim for annotation space

    plt.xlabel('Hit Ratio (%)')
    plt.ylabel('Throughput (req/s)')
    plt.xlim(-2, 57)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(loc='upper left')
    plt.tight_layout(pad=0.5) # Added padding

    output_dir = os.path.dirname(output_filename)
    if output_dir and not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
            print(f"Created directory: {output_dir}")
        except OSError as e:
            print(f"Error creating directory {output_dir}: {e}")
            return

    try:
        plt.savefig(output_filename, dpi=300, bbox_inches='tight', pad_inches=0.1) # Adjusted pad_inches
        print(f"Plot saved to: {output_filename}")
    except Exception as e:
        print(f"Error saving figure: {e}")

    plt.show()

if __name__ == "__main__":
    plot_throughput_vs_hit_ratio(csv_data)
