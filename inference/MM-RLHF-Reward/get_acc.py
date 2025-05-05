import json
from collections import defaultdict

# 输入文件路径
input_file = ""

# 类别关键词
category_keywords = ["mcq", "long", "short", "safety", "video"]

# 初始化统计
category_stats = {keyword: {"accuracy": 0, "acc_plus": 0, "total": 0} for keyword in category_keywords}
overall_stats = {"accuracy": 0, "acc_plus": 0, "total": 0}

# 用于存储每个id的items
id_to_items = defaultdict(list)

# 读取数据并分类
with open(input_file, "r") as infile:
    for line in infile:
        item = json.loads(line.strip())
        image_path = item.get("image", "") or item.get("video", "")
        item_id = item.get("id", "")
        id_to_items[item_id].append(item)

        # 分类到相应类别
        for keyword in category_keywords:
            if keyword in image_path:
                category_stats[keyword]["total"] += 1
                break

        # 更新总计
        overall_stats["total"] += 1

# 计算accuracy和acc_plus
for item_id, items in id_to_items.items():
    # 统计单个id是否满足acc+
    all_correct = True
    for item in items:
        reward_0 = item["rewards"][0]
        reward_1 = item["rewards"][1]
        correct = reward_0 > reward_1

        # 分类统计accuracy
        for keyword in category_keywords:
            if keyword in item.get("image", "") or keyword in item.get("video", ""):
                if correct:
                    category_stats[keyword]["accuracy"] += 1
                else:
                    all_correct = False
                break

        # 总体统计accuracy
        if correct:
            overall_stats["accuracy"] += 1
        else:
            all_correct = False

    # 更新acc+统计
    if all_correct:
        for keyword in category_keywords:
            if any(keyword in item.get("image", "") or keyword in item.get("video", "") for item in items):
                category_stats[keyword]["acc_plus"] += 1
                break
        overall_stats["acc_plus"] += 1

# 计算每个类别的accuracy和acc+
for keyword, stats in category_stats.items():
    if stats["total"] > 0:
        stats["accuracy"] = stats["accuracy"] / stats["total"]
        stats["acc_plus"] = stats["acc_plus"] / len(
            [item_id for item_id in id_to_items if any(keyword in (item.get("image", "") + item.get("video", "")) for item in id_to_items[item_id])]
        )

# 计算总体accuracy和acc+
if overall_stats["total"] > 0:
    overall_stats["accuracy"] = overall_stats["accuracy"] / overall_stats["total"]
    overall_stats["acc_plus"] = overall_stats["acc_plus"] / len(id_to_items)

# 输出结果
def print_metrics():
    print("\nCategory-wise Metrics:")
    for keyword, stats in category_stats.items():
        print(f"Category: {keyword}")
        print(f"  Accuracy: {stats['accuracy']:.2f}")
        print(f"  ACC+: {stats['acc_plus']:.2f}")
        print(f"  Total: {stats['total']}")

    print("\nOverall Metrics:")
    print(f"Overall Accuracy: {overall_stats['accuracy']:.2f}")
    print(f"Overall ACC+: {overall_stats['acc_plus']:.2f}")
    print(f"Total Items: {overall_stats['total']}")

# 输出
print_metrics()
