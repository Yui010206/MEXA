

import os
import base64
os.environ['DECORD_EOF_RETRY_MAX']='20480'

import openai
from decord import VideoReader, cpu
from PIL import Image
import cv2

import tempfile
from openai import AzureOpenAI

import json
from tqdm import tqdm 
import numpy as np

def map_numbers_to_letters(n):
    return chr(65 + n)

# set API Key 
AZURE_API_ENDPOINT = ""
AZURE_API_KEY = ""

client = AzureOpenAI(
    azure_endpoint = AZURE_API_ENDPOINT,
    api_key = AZURE_API_KEY, 
  api_version="2024-02-01"
)

def extract_frames_decord(video_path, num_frames=4):
    vr = VideoReader(video_path, ctx=cpu(0))
    total_frames = len(vr)
    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

    image_paths = []
    for idx in indices:
        frame = vr[idx].asnumpy()
        img = Image.fromarray(frame)
        temp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        img.save(temp_file.name)
        image_paths.append(temp_file.name)

    return image_paths

def extract_frames_from_dir(frame_dir, num_frames=4, extensions={'.png', '.jpg', '.jpeg'}):
    
    all_files = [f for f in os.listdir(frame_dir) if os.path.splitext(f)[1].lower() in extensions]
    def extract_frame_id(filename):
        digits = ''.join(filter(str.isdigit, filename))
        return int(digits) if digits else -1

    sorted_files = sorted(all_files, key=extract_frame_id)

    total = len(sorted_files)
    if total == 0:
        return []

    indices = np.linspace(0, total - 1, num=num_frames, dtype=int)
    selected_files = [os.path.join(frame_dir, sorted_files[i]) for i in indices]

    return selected_files

def extract_frames_opencv(video_path, num_frames=4):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(0, frame_count - 1, num=num_frames, dtype=int)

    saved_paths = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            print(f"Warning: failed to read frame {idx}")
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        img.save(tmp.name)
        saved_paths.append(tmp.name)

    cap.release()
    return saved_paths

def gpt4o_video_text_reasoning(image_paths, user_question):

    messages = [
        {"role": "system", "content": "You are a helpful assistant for multimodal understanding."},
        {"role": "user", "content": [{"type": "text", "text": user_question}] + [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encode_image_base64(p)}"}} for p in image_paths
        ]}
    ]

    response = client.chat.completions.create(
                model="gpt-4o", # "deployment_name".
                messages=messages,
                max_tokens=512,
                temperature=0.7,
            )
    
    return response.choices[0].message.content


def gpt4otext_reasoning(user_question):

    messages = [
        {"role": "system", "content": "You are a helpful assistant for multimodal understanding."},
        {"role": "user", "content": [{"type": "text", "text": user_question}]}
    ]

    response = client.chat.completions.create(
                model="gpt-4o", # "deployment_name".
                messages=messages,
                max_tokens=512,
                temperature=0.7,
            )
    
    return response.choices[0].message.content


def encode_image_base64(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode()


prompt_template = '''You are an expert multimodal reasoning assistant. Given a multimodal question (e.g., related to video, audio, 3D scenes, medical images, etc.), your task is to select all relevant skills and modalities required to accurately answer the question.
Task Type:\n{}
Question:\n{}
Options:\n{}

Available Skills and Modalities:

General Visual Perception: 
A1. Video Description; A2. Detailed Image Description; A3. Short Image Description;

Audio Perception: 
B1. Video Subtitle Extraction; B2. Audio Description; B3. Music Description; 

3D Visual Understanding: 
C1. 3D Scene Description; C2. 3D Situated Context Description;

Medical Visual Understanding :
D1. CT Scan Interpretation; D2. Medical Image Description;

OCR/Text Extraction 
E1. General OCR; E2. Poster/Slides Caption; E3. PDF Text Extraction;

Structured Visual Data Interpretation 
F1. Chart/Plot Description; F2. Table Description;

Mathematical and Geometric Understanding 
G1. Equation Caption; G2. Mathematics & Geometry Caption;

Instructions:
1. Only select skill/modality IDs necessary to answer the provided question.
2. Respond strictly with the selected skill IDs, separated by commas.

Selected IDs:
'''

# for open ended
prompt_template_open_ended = '''You are an expert multimodal reasoning assistant. Given a multimodal question (e.g., related to video, audio, 3D scenes, medical images, etc.), your task is to select all relevant skills and modalities required to accurately answer the question.
Task Type:\n{}
Question:\n{}

Available Skills and Modalities:

General Visual Perception: 
A1. Video Description; A2. Detailed Image Description; A3. Short Image Description;

Audio Perception: 
B1. Video Subtitle Extraction; B2. Audio Description; B3. Music Description;

3D Visual Understanding: 
C1. 3D Scene Description; C2. 3D Situated Context Description;

Medical Visual Understanding:
D1. CT Scan Interpretation; D2. Medical Image Description;

OCR/Text Extraction 
E1. General OCR; E2. Poster/Slides Caption; E3. PDF Text Extraction;

Structured Visual Data Interpretation 
F1. Chart/Plot Description; F2. Table Description;

Mathematical and Geometric Understanding 
G1. Equation Caption; G2. Mathematics & Geometry Caption;

Instructions:
1. Only select skill/modality IDs necessary to answer the provided question.
2. Respond strictly with the selected skill IDs, separated by commas.

Selected IDs:
'''


import argparse
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser("")
    
    parser.add_argument("--dataset", default='videommmu', type=str) # sqa3d, mmau, m3d
    parser.add_argument("--data_path", default='video-mmmu.json', type=str) 
    parser.add_argument("--video_base", default='VideoMMMU', type=str)
    parser.add_argument("--save_base", default='./skills/', type=str)
    parser.add_argument("--model", default='gpt4o', type=str)

    args = parser.parse_args()
    
    dataset = args.dataset
    data = json.load(open(args.data_path))
    video_base = args.video_base
    result = {}
    
    for d in tqdm(data):
        
        if dataset == 'videommmu':
            
            task_prompt = 'Video Reasoning, and you are provided with the video, and the video subtitle is generally helpful.'
            
            question = d['question'].strip('<image 1>') 
            if d['question_type'] != 'open':
                options = ""
                for i, o in enumerate(d['options']):
                    options += f"{map_numbers_to_letters(i)}: {o} \n"
                prompt = prompt_template.format(task_prompt, question, options)
            else:
                prompt = prompt_template_open_ended.format(task_prompt, question)
                
            video_url = video_base + d['vpath']
            
            # some videos are unsafe?
            try:
                frames = extract_frames_opencv(video_url, num_frames=10)
                response = gpt4o_video_text_reasoning(frames, prompt)
                # check response here
                print(response)
                result[d['uid']] = response  
            except:
                try:
                    task_prompt = 'Video Reasoning, and the video subtitle is generally helpful.'
                    prompt = prompt_template.format(task_prompt, question)
                    response = gpt4otext_reasoning(prompt)
                    print(response)
                    result[d['uid']] = response 
                except:
                    print('Error')
                    result[d['uid']] = 'Error'
                
        elif dataset == 'sqa3d':
            
            task_prompt = '3D Situated Question Answering, and you are provided with multiview images of the scene.' 
            question = d['question']
            situation = d['situation']
            question = situation + ' ' + question
            prompt = prompt_template_open_ended.format(task_prompt, question)
            video_url = video_base + d['scene_id']
            try:
                frames = extract_frames_from_dir(video_url, num_frames=4)
                response = gpt4o_video_text_reasoning(frames, prompt)
                print(response)
                result[d['question_id']] = response  
            except:
                
                try:
                    task_prompt = '3D Situated Question Answering'
                    prompt = prompt_template_open_ended.format(task_prompt, question)
                    response = gpt4otext_reasoning(prompt)
                    print(response)
                    result[d['question_id']] = response 
                except:
                    print('Error')
                    result[d['question_id']] = 'Error'
                
            # break
        elif dataset == 'mmau':
            
            task_prompt = '{} Reasoning, {}, {}.'.format(d['task'], d['category'], d['sub-category'])
            question = d['question']
            
            options = ""
            for i, o in enumerate(d['choices']):
                options += f"{map_numbers_to_letters(i)}: {o} \n"
            prompt = prompt_template.format(task_prompt, question, options)
            try:
                response = gpt4otext_reasoning(prompt)
                print(response)
                result[d['id']] = response  
            except:
                print('Error')
                result[d['id']] = 'Error'
        
        elif dataset == 'm3d':
            task_prompt = 'Medical Images Reasoning, and you are provided with a CT Scan.'
            question = d['Question']
            options = ""
            options += 'Option A: {}\n'.format(d['Choice A'])
            options += 'Option B: {}\n'.format(d['Choice B']) 
            options += 'Option C: {}\n'.format(d['Choice C'])
            options += 'Option D: {}\n'.format(d['Choice D'])
            
            video_url = video_base + '/'.join(d['Image Path'].split('/')[1:]).split('.')[0]
            prompt = prompt_template.format(task_prompt, question, options)
            try:
                frames = extract_frames_from_dir(video_url, num_frames=4)
                response = gpt4otext_reasoning(prompt)
                print(response)
                result[d['qid']] = response  
            except:
                try:
                    task_prompt = 'Medical Images Reasoning'
                    prompt = prompt_template.format(task_prompt, question, options)
                    response = gpt4otext_reasoning(prompt)
                    print(response)
                    result[d['id']] = response 
                except:
                    print('Error')
                    result[d['qid']] = 'Error'
                

    with open('./{}/{}_skill_id_{}.json'.format(args.save_base, args.dataset, args.model), 'w') as f:
        json.dump(result, f, indent=2)
        
