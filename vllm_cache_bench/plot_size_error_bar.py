import matplotlib.pyplot as plt
import json

name = {'ml': 'LPC', 'lru': 'LRU', 'ml-true': 'Oracle'}

import matplotlib.pyplot as plt
import json
from collections import defaultdict
import numpy as np

name = {'ml': 'LPC', 'lru': 'LRU', 'ml-true': 'Oracle'}

def plot_dataset(file_path, y_label, x_label):
    with open(file_path, 'r') as file:
        data = json.load(file)
    with open(file_path.replace('.json', '_0514.json'), 'r') as file:
        data_0514 = json.load(file)

    # Collect all values: {(algorithm, x): [y1, y2, ...]}
    grouped_data = defaultdict(list)
    dataset_name = data[0]['dataset_name']

    for config in data + data_0514:
        if config['request_rate'] != 0.01:
            continue
        if 'algorithm' in config:
            algorithm = config['algorithm']
            if 'true' in algorithm:
                continue
            x = config[x_label]
            if algorithm == 'ml':
                x += 250
            x = round(x * 16 / 1000, 2)  # round to avoid float precision mismatches
            if y_label == 'hit_ratios':
                y = float(config[y_label][-1])
            else:
                y = float(config[y_label])
            grouped_data[(algorithm, x)].append(y)
            # print(algorithm, x, y)

    # Aggregate values for plotting
    algo_series = defaultdict(lambda: {'x': [], 'y': [], 'yerr': []})
    for (algo, x), ys in grouped_data.items():
        y_mean = np.mean(ys)
        y_min = np.min(ys)
        y_max = np.max(ys)
        y_err = [[y_mean - y_min], [y_max - y_mean]]  # asymmetric error
        algo_series[algo]['x'].append(x)
        algo_series[algo]['y'].append(y_mean)
        print(algo, len(ys), x/16, y_max - y_min, (y_max - y_min) / y_mean)
        algo_series[algo]['yerr'].append((y_mean - y_min, y_max - y_mean))

    # Plotting
    plt.figure(figsize=(3.5, 2.7))
    for algo in ['lru', 'ml']:
        x_vals = algo_series[algo]['x']
        y_vals = algo_series[algo]['y']
        y_errs = np.array(algo_series[algo]['yerr']).T  # shape = (2, N) for asymmetric errorbars
        sorted_indices = np.argsort(x_vals)
        x_sorted = np.array(x_vals)[sorted_indices]
        y_sorted = np.array(y_vals)[sorted_indices]
        y_err_sorted = y_errs[:, sorted_indices]

        plt.errorbar(x_sorted, y_sorted, yerr=y_err_sorted, fmt='-o', capsize=3, markersize=0.6, label=name[algo])

    plt.ylim(bottom=0, top=0.5)
    plt.xlabel('Cache Size (×10³ tokens)')
    plt.ylabel('Hit Ratio')
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.legend()
    plt.tight_layout()

    output_path = f'fig/{y_label}_{dataset_name}_errorbar.png'
    print(output_path)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.01)
    plt.show()

dir = 'results/98a63908b41686889a6ade39c37616e54d49974d/'
file_paths = [
    f'{dir}/exp_lmsys.json',
    f'{dir}/exp_chatbot.json',
    f'{dir}/exp_sharegpt.json',
]

for file_path in file_paths:
    plot_dataset(file_path, 'hit_ratios', 'size')

# 200 does not have the conversaton end time adjustment by total output length
