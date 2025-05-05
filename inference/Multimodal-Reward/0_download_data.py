# Copyright: Meta Platforms, Inc. and affiliates

import os, json, shutil, string, ast
from collections import Counter
from tqdm import tqdm
from datasets import load_dataset

ROOT='./data'

os.system(f'mkdir -p {ROOT}/images')
os.system(f'mkdir -p {ROOT}/tmp')
os.system(f'mkdir -p {ROOT}/original_data')


def download_pairs():
    dname = "pairs"
    print(f"Downloading {dname}")
    os.system(f"mkdir -p {ROOT}/images/{dname}")
    os.system(f"git clone https://github.com/katiefraser/PAIRS {ROOT}/tmp/PAIRS")
    os.system(f"mv {ROOT}/tmp/PAIRS/data/* {ROOT}/images/{dname}/")


def download_nocaps():
    dname = "nocaps"
    print(f"Downloading {dname}")
    os.system(f"mkdir -p {ROOT}/images/{dname}")
    exs = load_dataset("HuggingFaceM4/NoCaps", split="test")
    final_exs = []
    for j in tqdm(range(len(exs))):
        try:
            ex = exs[j]
            i = ex['image_id']
            img = ex.pop('image')
            img.save(f"{ROOT}/images/{dname}/{i}.jpg", format="JPEG")
            ex["image"] = f"images/{dname}/{i}.jpg"
            final_exs.append(ex)
        except Exception as e:
            print(j, e)
    json.dump(final_exs, open(f"{ROOT}/original_data/{dname}.json", "w"), indent=2)


def setup_nocaps(data):
    dname = "nocaps"
    print(f"Settng up {dname}")
    for i, ex in enumerate(data):
        if ex['ID'].startswith(dname):
            data[i]['Text'] = "Please generate a detailed caption of this image. Please be as descriptive as possible."


def download_visitbench():
    dname = "visitbench"
    print(f"Downloading {dname}")
    os.system(f"mkdir -p {ROOT}/images/{dname}")
    exs = load_dataset("mlfoundations/VisIT-Bench", split="test", verification_mode='no_checks')
    final_exs = []
    for i, ex in enumerate(tqdm(exs)):
        img = ex.pop('image')
        if img.mode != "RGB":
            img = img.convert("RGB")
        img.save(f"{ROOT}/images/{dname}/{i}.jpg", format="JPEG")
        ex["image"] = f"images/{dname}/{i}.jpg"
        final_exs.append(ex)
    json.dump(final_exs, open(f"{ROOT}/original_data/{dname}.json", "w"), indent=2)


def setup_visitbench(data):
    dname = "visitbench"
    print(f"Settng up {dname}")
    src = json.load(open(f"{ROOT}/original_data/{dname}.json"))
    for i, ex in enumerate(data):
        if ex['ID'].startswith(dname):
            idnum = int(ex['Image'].split('/')[-1].split('.')[0])
            data[i]['Text'] = src[idnum]["instruction"]


def download_mmmu_pro():
    dname = "mmmu-s"
    print(f"Downloading {dname}")
    os.system(f"mkdir -p {ROOT}/images/{dname}")
    exs = load_dataset("MMMU/MMMU_Pro", "standard (10 options)", split="test")
    final_exs = []
    for i, ex in enumerate(tqdm(exs)):
        if ex['image_2'] is not None:
            final_exs.append(None)
            continue
        img = ex.pop('image_1')
        if img.mode != "RGB":
            img = img.convert("RGB")
        img.save(f"{ROOT}/images/{dname}/{i}.jpg", format="JPEG")
        ex["image"] = f"images/{dname}/{i}.jpg"
        final_exs.append(ex)
    json.dump(final_exs, open(f"{ROOT}/original_data/{dname}.json", "w"), indent=2)
    #
    dname = "mmmu-v"
    print(f"Downloading {dname}")
    os.system(f"mkdir -p {ROOT}/images/{dname}")
    exs = load_dataset("MMMU/MMMU_Pro", "vision", split="test")
    final_exs = []
    for i, ex in enumerate(tqdm(exs)):
        img = ex.pop('image')
        if img.mode != "RGB":
            img = img.convert("RGB")
        img.save(f"{ROOT}/images/{dname}/{i}.jpg", format="JPEG")
        ex["image"] = f"images/{dname}/{i}.jpg"
        final_exs.append(ex)
    json.dump(final_exs, open(f"{ROOT}/original_data/{dname}.json", "w"), indent=2)


def setup_mmmu_pro(data):
    dname = "mmmu-s"
    print(f"Settng up {dname}")
    src = json.load(open(f"{ROOT}/original_data/{dname}.json"))
    for i, ex in enumerate(data):
        if ex['ID'].startswith(dname):
            idnum = int(ex['Image'].split('/')[-1].split('.')[0])
            ex = src[idnum]
            question = ex['question'].replace('\r', '')
            choices = ast.literal_eval(ex["options"])
            labels = string.ascii_uppercase[:len(choices)]
            option = " ".join([f"({label}) {choice}" for label, choice in zip(labels, choices)])
            instruction = "Answer the preceding multiple choice question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of options. Think step by step before answering." 
            prompt = instruction + "\n" + question + " " + option
            data[i]['Text'] = prompt
    #
    dname = "mmmu-v"
    print(f"Settng up {dname}")
    src = json.load(open(f"{ROOT}/original_data/{dname}.json"))
    for i, ex in enumerate(data):
        if ex['ID'].startswith(dname):
            data[i]['Text'] = "Write out the multiple-choice question in the image and then solve it. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of options. Think step by step before answering." 
    

def download_mathvista():
    dname = "mathvista"
    print(f"Downloading {dname}")
    os.system(f"mkdir -p {ROOT}/images/{dname}")
    exs = load_dataset("AI4Math/MathVista", split="testmini")
    final_exs = []
    for _i, ex in enumerate(tqdm(exs)):
        if ex["metadata"]["language"]!= "english":
            final_exs.append(None)
            continue
        i = ex['pid']
        assert _i == int(i) - 1
        img = ex.pop('decoded_image')
        if img.mode != "RGB":
            img = img.convert("RGB")
        img.save(f"{ROOT}/images/{dname}/{i}.jpg", format="JPEG")
        ex["image"] = f"images/{dname}/{i}.jpg"
        final_exs.append(ex)
    json.dump(final_exs, open(f"{ROOT}/original_data/{dname}.json", "w"), indent=2)


def setup_mathvista(data):
    dname = "mathvista"
    print(f"Settng up {dname}")
    src = json.load(open(f"{ROOT}/original_data/{dname}.json"))
    for i, ex in enumerate(data):
        if ex['ID'].startswith(dname):
            idnum = int(ex['Image'].split('/')[-1].split('.')[0]) - 1
            ex = src[idnum]
            question = ex['question'].replace('\r', '')
            if ex["unit"]:
                question += " (Unit: "+ex["unit"]+")"
            if ex["question_type"] == "multi_choice":
                choices = ex["choices"]
                labels = string.ascii_uppercase[:len(choices)] 
                option = " ".join([f"({label}) {choice}" for label, choice in zip(labels, choices)])
                question = question + " "+ option
                instruction = "Think step by step about the following question and provide the correct option letter in **bold** format (for example, **A** or **B** or **C**). If you cannot determine the correct answer, take your best guess."
            elif ex["question_type"] == "free_form":
                if ex["answer_type"] == "integer":
                    instruction = "Think step by step about the following question, and then put your final answer in **bold** as a single integer (for example, **0** or **1** or **2**). If you don't know, guess."
                elif ex["answer_type"] == "float":
                    if ex["precision"] == 1:
                        instruction = "Think step by step about the following question, and then put your final answer in **bold** as a floating-point number with one decimal place (for example, **1.2** or **1.3** or **1.4**). If you don't know, guess."
                    elif ex["precision"] == 2:
                        instruction = "Think step by step about the following question, and then put your final answer in **bold** as a floating-point number with two decimal places (for example, **1.23** or **1.34** or **1.70**). If you don't know, guess."
            prompt = instruction + "\nQuestion:" + question
            data[i]['Text'] = prompt


def download_image2struct():
    dname = "image2struct"
    print(f"Downloading {dname}")
    os.system(f"mkdir -p {ROOT}/images/{dname}")
    exss = [
        load_dataset('stanford-crfm/image2struct-latex-v1', 'algorithm', split='validation'),
        load_dataset('stanford-crfm/image2struct-latex-v1', 'equation', split='validation'),
        load_dataset('stanford-crfm/image2struct-latex-v1', 'table', split='validation')
    ]
    final_exs = []
    for exs in exss:
        for i, ex in enumerate(tqdm(exs)):
            id = str(ex["uuid"]).replace('"', "")
            ex["uuid"] = id
            img = ex["image"]
            img.save(f"{ROOT}/images/{dname}/{id}.{str(img.format).lower()}")
            ex["image"] = f"images/{dname}/{id}.{str(img.format).lower()}"
            final_exs.append(ex)
    json.dump(final_exs, open(f"{ROOT}/original_data/{dname}.json", "w"), indent=2)


def setup_image2struct(data):
    dname = "image2struct"
    print(f"Settng up {dname}")
    for i, ex in enumerate(data):
        if ex['ID'].startswith(dname):
            data[i]['Text'] = "Please provide the LaTeX code used to generate this image. Only generate the code relevant to what you see. Your code will be surrounded by all the imports necessary as well as the begin and end document delimiters."


def download_realworldqa():
    dname = "RealworldQA"
    print(f"Downloading {dname}")
    os.system(f"mkdir -p {ROOT}/images/{dname}")
    exs = load_dataset("xai-org/RealworldQA", split="test")
    final_exs = []
    for i, ex in enumerate(tqdm(exs)):
        img = ex["image"]
        img.convert("RGB").save(f"{ROOT}/images/{dname}/{i}.jpg", "JPEG")
        ex["image"] = f"images/{dname}/{i}.jpg"
        final_exs.append(ex)
    json.dump(final_exs, open(f"{ROOT}/original_data/{dname}.json", "w"), indent=2)


def setup_realworldqa(data):
    dname = "RealworldQA"
    print(f"Settng up {dname}")
    src = json.load(open(f"{ROOT}/original_data/{dname}.json"))
    for i, ex in enumerate(data):
        if ex['ID'].startswith(dname):
            idnum = int(ex['Image'].split('/')[-1].split('.')[0])
            data[i]['Text'] = src[idnum]['question']

        
def download_mmbench():
    dname = "MMBench"
    print(f"Downloading {dname}")
    os.system(f"mkdir -p {ROOT}/images/{dname}")
    final_exs = []
    exs = load_dataset("lmms-lab/MMBench", "en", split="dev")
    for i, ex in enumerate(tqdm(exs)):
        img = ex["image"]
        img.save(f"{ROOT}/images/{dname}/{i}.{str(img.format).lower()}")
        ex["image"] = f"images/{dname}/{i}.{str(img.format).lower()}"
        final_exs.append(ex)
    json.dump(final_exs, open(f"{ROOT}/original_data/{dname}.json", "w"), indent=2)


def setup_mmbench(data):
    dname = "MMBench"
    print(f"Settng up {dname}")
    src = json.load(open(f"{ROOT}/original_data/{dname}.json"))
    for i, ex in enumerate(data):
        if ex['ID'].startswith(dname):
            idnum = int(ex['Image'].split('/')[-1].split('.')[0])
            ex = src[idnum]
            choices = [c for c in "ABCD" if ex[c] != "nan"]
            prompt = ex['question']
            for c in choices:
                prompt += f"\n({c}) {ex[c]}"
            prompt += f"\n\nAnswer with the option's letter from the given choices directly."
            data[i]['Text'] = prompt


def download_seed_bench():
    dname = "SEED-Bench"
    print(f"Downloading {dname}")
    os.system(f"mkdir -p {ROOT}/images/{dname}")
    exs = load_dataset("lmms-lab/SEED-Bench", split="test")
    final_exs = []
    for i, ex in enumerate(tqdm(exs)):
        if i >= 14200:
            break
        if ex["data_type"] == "image":
            img = ex["image"]
            if type(img) != list or len(img) != 1:
                final_exs.append(None)
                continue 
            img = img[0]
            ext = str(img.format).lower()
            if ext != 'none':
                img.save(f"{ROOT}/images/{dname}/{i}.{ext}")
                ex["image"] = f"images/{dname}/{i}.{ext}"
                final_exs.append(ex)
            else:
                final_exs.append(None)
    print(len(final_exs))
    json.dump(final_exs, open(f"{ROOT}/original_data/{dname}.json", "w"), indent=2)


def setup_seed_bench(data):
    dname = "SEED-Bench"
    print(f"Settng up {dname}")
    src = json.load(open(f"{ROOT}/original_data/{dname}.json"))
    for i, ex in enumerate(data):
        if ex['ID'].startswith(dname):
            idnum = int(ex['Image'].split('/')[-1].split('.')[0])
            ex = src[idnum]
            choices = list("ABCD")
            prompt = ex['question']
            for c in choices:
                prompt += f"\n({c}) " + ex[f"choice_{c.lower()}"]
            prompt += f"\n\nAnswer with the option's letter from the given choices directly."
            data[i]['Text'] = prompt


def download_emma_coding():
    dname = "EMMA-Coding"
    print(f"Downloading {dname}")
    os.system(f"mkdir -p {ROOT}/images/{dname}")
    exs = load_dataset("luckychao/EMMA", "Coding", split="test")
    final_exs = []
    for i, ex in enumerate(tqdm(exs)):
        img_count = 0
        for j in range(5):
            img = ex[f"image_{j+1}"]
            try:
                img.save(f"{ROOT}/images/{dname}/{i}_{j}.{str(img.format).lower()}")
                img_count += 1
            except:
                pass
        if img_count == 1:
            j = 0
            img = ex[f"image_{j+1}"]
            ex["image"] = f"images/{dname}/{i}_{j}.{str(img.format).lower()}"
            assert os.path.exists(f"{ROOT}/images/{dname}/{i}_{j}.{str(img.format).lower()}")
            for j in range(5):
                _ = ex.pop(f"image_{j+1}")
            final_exs.append(ex)
        else:
            final_exs.append(None)
    print(len(final_exs))
    json.dump(final_exs, open(f"{ROOT}/original_data/{dname}.json", "w"), indent=2)


def setup_emma_coding(data):
    dname = "EMMA-Coding"
    print(f"Settng up {dname}")
    src = json.load(open(f"{ROOT}/original_data/{dname}.json"))
    for i, ex in enumerate(data):
        if ex['ID'].startswith(dname):
            idnum = int(ex['Image'].split('/')[-1].split('.')[0].split('_')[0])
            ex = src[idnum]
            prompt = ex['question'][len("<image_1>\n\n"):]
            prompt = prompt.replace("Which code snippet below", "What code snippet")
            data[i]['Text'] = prompt


def check_all_images_exist():
    data = json.load(open(f"{ROOT}/all_data_release.json"))
    for ex in tqdm(data):
        assert os.path.exists(f"{ROOT}/{ex['Image']}")


if __name__ == "__main__":
    download_pairs()
    download_nocaps()
    download_visitbench()
    download_mmmu_pro()
    download_mathvista()
    download_image2struct()
    download_realworldqa()
    download_mmbench()
    download_seed_bench()
    download_emma_coding()

    check_all_images_exist()


    data = json.load(open(f"{ROOT}/all_data_release.json"))
    setup_nocaps(data)
    setup_visitbench(data)
    setup_mmmu_pro(data)
    setup_mathvista(data)
    setup_image2struct(data)
    setup_realworldqa(data)
    setup_mmbench(data)
    setup_seed_bench(data)
    setup_emma_coding(data)

    json.dump(data, open(f"{ROOT}/all_data.json", 'w'), indent=2)
    print('Done.')
