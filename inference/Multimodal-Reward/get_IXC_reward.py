import argparse
import torch
import os
import json
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

from PIL import Image
import math
import numpy as np
# import debugpy
import re


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

from transformers import AutoModel, AutoTokenizer
def eval_model(args):
    # Model
    model_path = args.model_path
    print(model_path)
    # model_name = "internlm/internlm-xcomposer2d5-7b-reward"
    model = AutoModel.from_pretrained(
        model_path, 
        torch_dtype=torch.float16, 
        trust_remote_code=True,
    ).cuda().eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model.tokenizer = tokenizer

    questions = []
    with open(args.question_file, 'r') as file:
        questions.extend(json.load(file))
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file_with_chunks = f"{args.answers_file.rsplit('.', 1)[0]}_chunk{args.chunk_idx}_of_{args.num_chunks}.jsonl"
    existing_ids = set()
    try:
        with open(answers_file_with_chunks, 'r') as file:
            for line in file:
                answer = json.loads(line)
                existing_ids.add(answer.get("id"))
    except FileNotFoundError:
        print(f"No existing file found: {answers_file_with_chunks}, proceeding with empty set of existing IDs.")

    questions = [q for q in questions if q.get("id") not in existing_ids]
    print(f'saving all the answers to {answers_file_with_chunks}')
    answers_file = os.path.expanduser(answers_file_with_chunks)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "a+")

    index, cnt_images = 0, []
    import random
    for line in tqdm(questions, total=len(questions)):
        video_file = line.get("video", None)
        image_file = line.get("Image", None)

        rewards = []
        critic_texts = []
        rand_num = random.random()
        if line['Better'] == "Output2":
            chosen, rejected = line['Output2'], line['Output1']
        else:
            chosen, rejected = line['Output1'], line['Output2']
        
        
        for input_qs in [chosen, rejected]:
            chat = [
                {"role": "user", "content": line['Text']},
                {"role": "assistant", "content": input_qs}
            ]
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                score = model.get_score(chat, [os.path.join(args.image_folder, image_file)], hd_num=9)
            rewards.append(score)

        index += 1
        if rewards[0] >= rewards[1]:
            line['rewards'] = [1, 0]
        else:
            line['rewards'] = [0, 1]
        ans_file.write(json.dumps(line) + "\n")
        ans_file.flush()
    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # sampling strategy follows simpo
    # nohup python examples/scripts/reward_model/vl_reward_bench.py --num-chunks 4 --chunk-idx 3 >> examples/scripts/reward_model/logs/qwenvl25_7b_ciritc_lr1e_5_bsz128_wo_critic_newhead_0_4.log 2>&1 &
    # parser.add_argument("--rm-model-path", type=str, default="../model/alignment/baseline_promptv3")
    parser.add_argument("--model-path", type=str, default="internlm/internlm-xcomposer2d5-7b-reward")
    parser.add_argument("--image-folder", type=str, default="/mllm_hdd/mllm_hdd/yfzhang/MMPreference/multimodal_rewardbench-main/data/")
    parser.add_argument("--question-file", type=str, default="/mllm_hdd/mllm_hdd/yfzhang/MMPreference/multimodal_rewardbench-main/data/all_data.json")
    parser.add_argument("--answers-file", type=str, default="/mllm_hdd/mllm_hdd/yfzhang/MMPreference/multimodal_rewardbench-main/outputs/ixc-rweard.jsonl")
    parser.add_argument("--conv-mode", type=str, default="qwen_1_5")
    parser.add_argument("--num-chunks", type=int, default=10)
    parser.add_argument("--chunk-idx", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=0.)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--num_critic", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=4096)

    parser.add_argument("--use_gt_critic", type=bool, default=False)
    parser.add_argument("--wo_critic", type=bool, default=False)

    parser.add_argument(
        "--test-prompt",
        type=str,
        default="",
    )
    parser.add_argument("--hd_num", type=int, default=9)
    args = parser.parse_args()
    print(args)

    eval_model(args)