import json
import numpy as np
import os


def calculate_average_accuracy(base_path, seeds=[1, 2, 3]):
    accuracies = []

    for seed in seeds:
        file_path = f"{base_path}/seed{seed}/results.json"

        try:
            with open(file_path, 'r') as f:
                data = json.load(f)

            # 根据实际JSON结构调整这个字段名
            if 'accuracy' in data:
                accuracies.append(data['accuracy'])
        except:
            continue

    if accuracies:
        return {
            'avg': np.mean(accuracies),
            'std': np.std(accuracies),
            'values': [round(acc, 4) for acc in accuracies]
        }
    return None

# 主程序
base_dir = "/data1/mazc/cz/dpmoe/output/DPMDA/officehome_ctx20/real_world/"
seeds = [1, 2, 3]
weight_range = [f"CT_W{i / 10:.1f}" for i in range(0, 41)]  # CT_W0.1 到 CT_W1.0

results = {}

for weight in weight_range:
    current_path = os.path.join(base_dir, weight)
    if not os.path.exists(current_path):
        continue

    stats = calculate_average_accuracy(current_path, seeds)
    if stats:
        results[weight] = stats

# 打印结果
print("{:<10} {:<10} {:<15} {:<20}".format("Weight", "Avg Acc", "Std Dev", "Seed Values"))
print("-" * 60)
for weight, stat in results.items():
    print("{:<10} {:<10.4f} {:<15.4f} {}".format(
        weight,
        stat['avg'],
        stat['std'],
        stat['values']
    ))