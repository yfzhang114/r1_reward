import json
import os
import random
import re
from argparse import ArgumentParser
from multiprocessing import Process, Queue

import Levenshtein
from flask import Flask, jsonify, request
from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify
from openrlhf.models.remote_rm.strict_math_verify import evaluate_math, verify_choice_answer

app = Flask(__name__)

problem_to_answer = {}


def get_response_from_query(q: str):
    ends_of_sentence = ["<|im_end|>", "<｜end▁of▁sentence｜>", "<|endoftext|>"]
    pos = re.search(response_prefix, q)
    if pos is None:
        return None
    response = q[pos.end() :]
    for e in ends_of_sentence:
        response = response.replace(e, "")
    return response.strip()


def verify_format(content):
    """
    Verify if the string meets the format requirements:
    - Must start with <think> and end with </answer>
    - Must contain exactly one pair of <think>...</think> and <answer>...</answer> tags
    - No extra characters allowed between </think> and <answer> tags
    """
    think_count = content.count("<think>")
    answer_count = content.count("<answer>")
    return bool(re.match(format_pattern, content, re.DOTALL)) and think_count == 1 and answer_count == 1


def find_similar_problem(problem):
    max_sim = -1
    target_problem = None
    for p in problem_to_answer.keys():
        sim = Levenshtein.ratio(problem, p)
        if sim > max_sim:
            max_sim = sim
            target_problem = p
    return target_problem


def extract_characters_regex(s):
    s = s.strip()
    answer_prefixes = [
        "The best answer is",
        "The correct answer is",
        "The answer is",
        "The answer: ",
        "Answer: ",
        "The best option is",
        "The correct option is",
        "Best answer:",
        "Best option:",
    ]
    for answer_prefix in answer_prefixes:
        s = s.replace(answer_prefix, "")
    return s

def repetition_penalty(completion: str, repetition_n_grams: int = 3, repetition_max_penalty: float = -1.0) -> float:
    """
    Reward function that penalizes repetitions in a text completion.

    Args:
        completion: A single model completion (string).
        repetition_n_grams: Size of n-grams to consider for repetition penalty.
        repetition_max_penalty: Maximum penalty for repeated n-grams.

    Returns:
        Penalty score for the completion.
    """
    def zipngram(text: str, ngram_size: int):
        words = text.lower().split()
        return zip(*[words[i:] for i in range(ngram_size)])

    # Handle empty completion
    if completion == '':
        return 0.0
    
    # Handle short completions
    if len(completion.split()) < repetition_n_grams:
        return 0.0

    ngrams = set()
    total = 0
    for ng in zipngram(completion, repetition_n_grams):
        ngrams.add(ng)
        total += 1

    scaling = 1 - len(ngrams) / total
    reward = scaling * repetition_max_penalty
    return reward


def verify_math(input_queue, output_queue):
    while True:
        content, sols, problem = input_queue.get()
        
        # Try symbolic verification first
        try:
            for sol in sols:
                sol_match = re.search(r"<answer>(.*?)</answer>", sol, re.DOTALL)
                ground_truth = sol_match.group(1).strip() if sol_match else sol.strip()

                # Extract answer from content if it has think/answer tags
                content_match = re.search(r"<answer>(.*?)</answer>", content, re.DOTALL)
                student_answer = content_match.group(1).strip() if content_match else content.strip()
                student_answer = extract_characters_regex(student_answer)
                
                answer = parse(student_answer)
                if evaluate_math(student_answer, ground_truth):
                    reward = 1.0
                    break
                elif float(verify(answer, parse(ground_truth))) > 0:
                    reward = 1.0
                    break
                else:
                    reward = 0.0
        except Exception:
            reward = 1.0
            pass  # Continue to next verification method if this fails
        output_queue.put(reward)

def get_gpt_reward(response):
    content_match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
    if content_match:
        student_answer = content_match.group(1).strip()
        try:
            answer = int(student_answer)
            if answer == 1:
                student_answer = f'response 1 is better'
            elif answer == 2:
                student_answer = f'response 2 is better'
            else:
                student_answer = f'two response are the same'
        except:
            student_answer = student_answer
    else:
        student_answer = response.strip()
    
    cleaned_response = response.split("<answer>")[0].strip()
    cleaned_response = cleaned_response.split('\n')[-3:]
    cleaned_response = ';'.join(cleaned_response)
    prompt = f"""
You are a strict judge with one task: Determine if the Model's Response  exactly matches the model' Answer.

# Rules:
1. Output 1 when model's response and answer have the same meaning.
2. Output 0 when the model answer and response mismatch:
   - Ambiguous conclusion without clear choice
   - Choice direction opposite to answer
   - Approving both answers equally


######### Model's Response #########\n {cleaned_response} \n\n
######### Model's Answer#########\n {student_answer}\n\n
Output: 
""" 
    
#3. Strictly output just one int value: 1 (match) or 0 (mismatch)

    return prompt


import openai
def verify_conflict(prompt, image_paths=[], max_cycle=3):
    prompt = get_gpt_reward(prompt)
    while max_cycle:
        # prompt = 'who are you?'
        openai_api_key = "EMPTY"
        openai_api_base = "http://10.82.120.96:8000/v1"

        content = [{"type": "text", "text": prompt}]
        messages = [{"role": "user", "content": content}]

        client = openai.OpenAI(
            # defaults to os.environ.get("OPENAI_API_KEY")
            api_key=openai_api_key,
            base_url=openai_api_base,
        )
        models = client.models.list()
        model = models.data[0].id
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=4096,
            temperature=0.1,
            stream=False      # 确保完整返回
        )

        answer = response.choices[0].message.content
        answer = answer.replace('output:', '').strip()
        if len(answer) == 1 and (answer == '1' or answer == '0'):
            score = int(answer)
            return 1 if score == 1 else -1
        elif '0' in answer:
            return -1
        elif '1' in answer:
            return 1
        max_cycle -= 1
    return 0



@app.route("/get_reward_mllm", methods=["POST"])
def get_reward():
    # 获取请求中的 JSON 数据
    data = request.get_json()
    # 检查是否有 'query' 字段
    if "query" not in data:
        return jsonify({"error": "queries field is required"}), 400
    rewards = []
    rewards_format = []
    for q,problem in zip(data["query"],data["prompts"]):
        if problem is None:
            return jsonify({"error": f"problem not found from {q}"}), 400
        if problem not in problem_to_answer:
            # This should not happen
            print(f"problem not exists: {problem}")
            problem = find_similar_problem(problem)
        answer = problem_to_answer[problem]
        response = get_response_from_query(q) or q
        if response is None:
            return jsonify({"error": f"response not found from {q}"}), 400
        format_reward = float(verify_format(response))
        conflict_reward = float(verify_conflict(response))
        input_queue.put((response, answer, problem))
        acc_reward = float(output_queue.get())
        do_print = random.randint(1, 10) == 1
        if do_print:
            info=f"Query: {q}\n\nProblem: {problem}\n\n Answer: {answer}\n\n Response: {response}\n\n Format Reward: {format_reward}\n\n Acc Reward: {acc_reward}\n\nconflict_reward: {conflict_reward}"
            info = re.sub(r"<\|.*?\|>","",info)
            print(info)
        
        rewards.append(0.5 * format_reward + acc_reward * (1 + 0.5 * conflict_reward))
    # 返回包含 rewards 的响应
    return jsonify({"rewards": rewards})    

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--dataset", type=str, default="examples/data/llava_10k_sfft_margin3_math.json", help="Dataset to use"
    )
    parser.add_argument(
        "--prompt-template", type=str, default="chatml", help="Prompt template"
    )
    parser.add_argument(
        "--input_key", type=str, default="message", help="The key name of prompt."
    )
    args = parser.parse_args()
    if args.dataset.endswith("json"):
        with open(args.dataset, "r") as f:
            dataset = json.load(f)
    elif args.dataset.endswith("jsonl"):
        with open(args.dataset, "r") as f:
            dataset = [json.loads(l) for l in f.readlines()]

    format_pattern = r"^<think>.*?</think><answer>.*?</answer>$"

    if args.prompt_template=="chatml":
        problem_pattern = r"<\|im_start\|>user\n(.*?)<\|im_end\|>"
        response_prefix = r"<\|im_start\|>assistant\n"
    elif args.prompt_template=="qwen1":
        problem_pattern = r"｜User｜>(.*?)<｜Assistant｜>"
        response_prefix = r"<｜Assistant｜>"
    elif args.prompt_template=="base":
        problem_pattern = r"User: (.*?)\n\nAssistant:"
        response_prefix = r"Assistant: "
    else:
        raise ValueError(f"Unknown chat format: {args.dataset}")
    print("load dataset success")
    for item in dataset:
        problem = item[args.input_key]
        answer = item["answer"]
        assert type(item["answer"]) is list
        problem_to_answer[problem] = answer

    input_queue = Queue()
    output_queue = Queue()
    # math_verify can only run in main thread
    p = Process(target=verify_math, args=(input_queue, output_queue))
    p.start()

    app.run(host="0.0.0.0", port=20240, debug=False, use_reloader=False)
    p.kill()