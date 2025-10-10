import matplotlib.pyplot as plt
import json

name = {'ml': 'LPC', 'lru': 'LRU', 'ml-true': 'With Oracle'}

def plot_dataset(file_path, y_label):
    with open(file_path, 'r') as file:
        data = json.load(file)

    # Organize data by eviction algorithm
    algorithm_data = {}
    for config in data:
    # for config in reversed(data):
        if 'algorithm' in config:
            algorithm = config['algorithm']
            if 'true' in algorithm:
                continue
        x = config['request_rate']
        if x > 0.02:
            continue
        if x < 0.0025:
            continue
        # print(config['size'])
        x = 1 / x
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
    for algorithm in ['lru', 'ml']:
        values = algorithm_data[algorithm]
        sorted_pairs = sorted(zip(values['xs'], values[y_label]))
        xs_sorted, ys_sorted = zip(*sorted_pairs)
        # print(algorithm, ys_sorted)
        if algorithm == 'lru':
            lru_hits = ys_sorted
        else:
            for i in range(len(ys_sorted)):
                print(ys_sorted[i] - lru_hits[i], (ys_sorted[i] - lru_hits[i]) / lru_hits[i])
        plt.plot(xs_sorted, ys_sorted, marker='o', label=name[algorithm])

    plt.ylim(bottom=0.2)
    plt.ylim(top=0.5)
    plt.xlabel('Chat Interval (s)')
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
    print(f'fig/{y_label}_interval_{dataset_name}.png')
    plt.savefig(f'fig/{y_label}__interval__{dataset_name}.png', dpi=300, bbox_inches='tight', pad_inches=0.01)
    plt.show()

dir = 'results/98a63908b41686889a6ade39c37616e54d49974d/'
# dir = 'results/a29cae3df5d16cc895083497dad6ba9530c7d84c'
file_paths = [
f'{dir}/exp_sharegpt_reqrate.json',
f'{dir}/exp_lmsys_reqrate.json',
f'{dir}/exp_chatbot_reqrate.json',
]
for file_path in file_paths:
    plot_dataset(file_path, 'hit_ratios')
    # plot_dataset(file_path, 'p99_ttft')

# 200 does not have the conversaton end time adjustment by total output length
