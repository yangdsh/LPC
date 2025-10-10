import matplotlib.pyplot as plt
import json

name = {'ml': 'LPC', 'lru': 'LRU', 'ml-true': 'Oracle'}

def plot_dataset(file_path, y_label, x_label):
    with open(file_path, 'r') as file:
        data = json.load(file)

    # Organize data by eviction algorithm
    algorithm_data = {}
    for config in data:
        if config['request_rate'] != 0.01:
            continue
        if 'algorithm' in config:
            algorithm = config['algorithm']
        x = config[x_label]
        # add extra memory overhead for ml
        if 'ml' in algorithm:
            x += 250
        # number of tokens in the cache
        x = x * 16 / 1000
        if algorithm not in algorithm_data or x not in algorithm_data[algorithm]['xs']:
            if y_label == 'hit_ratios':
                y = float(config[y_label][-1])
            else:
                y = float(config[y_label])

            if algorithm not in algorithm_data:
                algorithm_data[algorithm] = {'xs': [], y_label: [], 'miss_ratios': []}

            algorithm_data[algorithm]['xs'].append(x)
            algorithm_data[algorithm][y_label].append(y)
            algorithm_data[algorithm]['miss_ratios'].append(1-y)
    dataset_name = data[0]['dataset_name']

    # Plotting
    plt.figure(figsize=(3.5, 2.7))
    for algorithm in ['lru', 'ml', 'ml-true']:
        values = algorithm_data[algorithm]
        sorted_pairs = sorted(zip(values['xs'], values[y_label]))
        xs_sorted, ys_sorted = zip(*sorted_pairs)
        print(algorithm, ys_sorted)
        plt.plot(xs_sorted[0:], ys_sorted[0:], marker='o', label=name[algorithm])

    plt.ylim(bottom=0)
    # plt.ylim(top=0.5)
    plt.xlabel('Cache Size (×10³ tokens)')
    if y_label == 'p99_ttft':
        plt.ylabel('P99 TTFT (ms)')
    else:    
        plt.ylabel('Hit Ratio')
    # plt.title(dataset_name)
    # plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # Save figure to file
    print(f'fig/{y_label}_{dataset_name}_true.png')
    plt.savefig(f'fig/{y_label}_{dataset_name}_true.png', dpi=300, bbox_inches='tight', pad_inches=0.01)
    plt.show()

dir = 'results/98a63908b41686889a6ade39c37616e54d49974d/'
# dir = 'results/a29cae3df5d16cc895083497dad6ba9530c7d84c'
# dir = 'results/98a63908b41686889a6ade39c37616e54d49974d/result_Nconv=300_1'
file_paths = [f'{dir}/exp_chatbot.json',
f'{dir}/exp_sharegpt.json',
f'{dir}/exp_lmsys.json',
]

for file_path in file_paths:
    plot_dataset(file_path, 'hit_ratios', 'size')