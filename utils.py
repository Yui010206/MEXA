import pickle
import json
from pathlib import Path
import argparse
import pandas as pd
from pprint import pprint


def load_pkl(fn):
    with open(fn, 'rb') as f:
        data = pickle.load(f)
    return data

def save_pkl(data, fn):
    with open(fn, 'wb') as f:
        pickle.dump(data, f)

def load_json(fn):
    with open(fn, 'r') as f:
        data = json.load(f)
    return data

def save_json(data, fn, indent=4):
    with open(fn, 'w') as f:
        json.dump(data, f, indent=indent)

def makedir(path):
    Path(path).mkdir(parents=True, exist_ok=True)

def parse_args():
    parser = argparse.ArgumentParser("")

    # data
    parser.add_argument("--dataset", default='videommmu', type=str)  # 'egoschema', 'nextqa', 'nextgqa', 'intentqa'
    parser.add_argument("--data_path", default='/nas-ssd2/shoubin/datasets/multimodal_symbolic_moe/video-mmmu.json', type=str) 
    parser.add_argument("--anno_path", default='/nas-ssd2/shoubin/datasets/multimodal_symbolic_moe/video-mmmu.json', type=str)  
    parser.add_argument("--duration_path", default='data/egoschema/duration.json', type=str) 
    parser.add_argument("--fps", default=1.0, type=float) 
    parser.add_argument("--caption_every", default=1, type=int) 
    parser.add_argument("--num_examples_to_run", default=-1, type=int)
    ## backup pred
    parser.add_argument("--backup_pred_path", default="", type=str)
    ## fewshot
    parser.add_argument("--fewshot_example_path", default="", type=str) 
    ## nextgqa
    parser.add_argument("--nextgqa_gt_ground_path", default="", type=str)
    parser.add_argument("--nextgqa_pred_qa_path", default="", type=str)

    # output
    parser.add_argument("--output_base_path", required=True, type=str)  
    parser.add_argument("--output_filename", required=True, type=str)  

    # prompting
    parser.add_argument("--model", default="gpt-3.5-turbo-0613", type=str)
    parser.add_argument("--api_key", default="", type=str)
    parser.add_argument("--temperature", default=0.0, type=float)
    parser.add_argument("--prompt_type", default="qa_standard", type=str)
    parser.add_argument("--task", default="qa", type=str)  # sum, qa, gqa
    ## sum
    parser.add_argument("--num_words_in_sum", default=500, type=int)  
    ## for llama acceleration
    parser.add_argument("--max_new_tokens", default=100, type=int)  

    # other
    parser.add_argument("--disable_eval", action='store_true')
    parser.add_argument("--videomme_type", default="long", type=str)  # for videomme eval
    parser.add_argument("--start_from_scratch", action='store_true')
    parser.add_argument("--save_info", action='store_true')
    parser.add_argument("--save_every", default=10, type=int)
    parser.add_argument("--hf_token", default="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", type=str)

    return parser.parse_args()


def build_fewshot_examples(qa_path, data_path):
    if len(qa_path) == 0 or len(data_path) == 0:
        return None
    qa = load_json(qa_path)
    data = load_json(data_path)  # uid --> str or list 
    examplars = []
    int_to_letter = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E'}
    for i, (uid, examplar) in enumerate(qa.items()):
        description = data[uid]
        if isinstance(description, list):
            description = '. '.join(description)
        examplars.append(f"Examplar {i}.\n Descriptions: {description}.\n Question: {examplar['question']}\n A: {examplar['0']}\n B: {examplar['1']}\n C: {examplar['2']}\n D: {examplar['3']}\n E: {examplar['4']}\n Answer: {int_to_letter[examplar['truth']]}.")
    examplars = '\n\n'.join(examplars)
    return examplars
    

def restore_videomme(data, anno_path):
    anno = load_json(anno_path)
    res = []
    for video_data in anno:
        if video_data['questions'][0]['question_id'] not in data:
            continue
        questions = []
        for el in video_data['questions']:
            questions.append(data[el['question_id']])
        res.append({
            "video_id": video_data['video_id'],
            "duration": video_data['duration_category'],
            "domain": video_data['video_category'],
            "sub_category": video_data['video_subcategory'],
            "questions": questions
        })
    return res