# Copyright: Meta Platforms, Inc. and affiliates

import json
import argparse
import os
import time
from tqdm import tqdm

import random
from collections import defaultdict


def compute_acc(args):
    accs = defaultdict(list)
    random.seed(123)
    for line in open(args.answers_file):
        ex = json.loads(line)
        # pred = extract_judgment(ex['output'])
        acc = int(ex['rewards'][0] > ex['rewards'][1])
        accs['all'].append(acc)
        category = ex['Category']
        if category == 'safety':
            if ex['ID'].lower().startswith('pairs'):
                category = 'safety/bias'
            else:
                category = 'safety/toxicity'
        elif category == 'reasoning':
            if ex['ID'].lower().startswith('math'):
                category = 'reasoning/math'
            else:
                category = 'reasoning/coding'
        accs[category].append(acc)
    for task in accs:
        print(f"acc {task}: {sum(accs[task])} / {len(accs[task])} = {sum(accs[task])/len(accs[task])}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--answers-file", type=str, default="/mllm_hdd/mllm_hdd/yfzhang/MMPreference/multimodal_rewardbench-main/outputs/llava_ov_reward_7b_50k_each_class_mmrlhf_each8k_exist_20k_ov-reward-wotie.jsonl")
    args = parser.parse_args()

    compute_acc(args)

"""
cd <repo>

python scripts/2_get_accuracy.py
"""


