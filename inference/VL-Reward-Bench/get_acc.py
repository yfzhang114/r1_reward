import json
from collections import defaultdict

# 定义分类规则
def classify_id(data_id):
    if "mmmu" in data_id or "mathverse" in data_id:
        return "reasoning"
    elif "hallucination" in data_id or "rlhf" in data_id or "rlaif" in data_id:
        return "hallucination"
    else:
        return "general"

# 计算准确率
def calculate_accuracy_and_stats(file_path):
    # 初始化计数器
    stats = {
        "reasoning": {"count": 0, "correct": 0},
        "hallucination": {"count": 0, "correct": 0},
        "general": {"count": 0, "correct": 0}
    }

    # 读取 JSONL 文件
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            # 解析 JSON 行
            data = json.loads(line.strip())
            data_id = data["id"]
            rewards = data["rewards"][:1]

            # 分类
            category = classify_id(data_id)

            # 更新计数器
            stats[category]["count"] += 1
            if rewards[0] > rewards[1]:  # 判断准确性
                stats[category]["correct"] += 1

    # 计算每个类别的准确率
    results = {}
    for category, values in stats.items():
        count = values["count"]
        correct = values["correct"]
        accuracy = correct / count if count > 0 else 0
        results[category] = {
            "samples": count,
            "accuracy": accuracy
        }

    # 计算平均准确率
    total_samples = sum(values["count"] for values in stats.values())
    total_correct = sum(values["correct"] for values in stats.values())
    average_accuracy = total_correct / total_samples if total_samples > 0 else 0

    return results, average_accuracy

# 主函数
if __name__ == "__main__":
    file_path = ""  # JSONL 文件路径
    results, average_accuracy = calculate_accuracy_and_stats(file_path)

    # 输出结果
    overall = 0
    for category, stats in results.items():
        print(f"Category: {category}")
        print(f"  Samples: {stats['samples']}")
        print(f"  Accuracy: {stats['accuracy']:.4f}")
        overall += stats['accuracy']
    print(f"Average Accuracy: {average_accuracy:.4f}")
    macro_acc = overall / 3
    print(f"Macro Accuracy: {macro_acc:.4f}")
