
import json
import os
from utils import save_json, load_json, save_pkl, load_pkl, makedir, parse_args
from torch.utils.data import Dataset
import pandas as pd
import pdb
import re
from pprint import pprint

from datetime import timedelta


def extract_subtitles_srt(subtitle_dir, video_id, include_time=True):
    subtitle_file = os.path.join(subtitle_dir, f'{video_id}.srt')
    
    if not os.path.exists(subtitle_file):
        print(f"Subtitle file not found for videoID: {video_id} and path {subtitle_file}")
        return ""
    
    with open(subtitle_file, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    
    # Remove timestamps and numbers; extract only subtitle text
    subtitles = []
    if include_time:
        for line in lines:
            line = line.strip()
            if '<' in line and '>' in line:
                line = re.sub(r'<.*?>', '', line)  # Remove HTML tags
            subtitles.append(line)
    else:
        for line in lines:
            line = line.strip()
            if not re.match(r'^[0-9]+$', line) and '-->' not in line:  # Skip numbers and timestamps
                line = re.sub(r'<.*?>', '', line)  # Remove HTML tags
                if line:
                    subtitles.append(line)
    
    return '\n'.join(subtitles)



def format_seconds(seconds):
    return str(timedelta(seconds=seconds)).rjust(8, '0')



def extract_captions(caption_dir, video_id, stride, clip_length, include_time=True):
    caption_file_path = os.path.join(caption_dir, f'{video_id}.txt')

    if not os.path.exists(caption_file_path):
        print(f"Caption file not found for video_id: {video_id} and path {caption_file_path}")
        return ""
    
    with open(caption_file_path, "r", encoding="utf-8") as file:
        vid_captions_list = file.readlines()  # Read the entire file as a single string
    vid_captions_list = vid_captions_list[::stride]

    if not include_time:
        return ''.join(vid_captions_list)
    
    timestamped_captions = ""
    for i in range(len(vid_captions_list)):
        start_time, end_time = format_seconds(i*stride*clip_length), format_seconds(i*stride*clip_length+clip_length)
        timestamped_captions += f"{start_time} --> {end_time}\n{vid_captions_list[i]}"
        
    return timestamped_captions

class BaseDataset(Dataset):
    def __init__(self, args, quids_to_exclude=None, num_examples_to_run=-1):
        '''
        num_examples_to_run < 0: run all
        '''
        self.args = args
        self.narrations = self.get_descriptions()  # uid --> list of str  or  uid --> str
        self.anno = self.get_anno()
        self.durations = load_json(args.duration_path)  # uid --> float
        data = self.build()
        data = self.filter(data, quids_to_exclude, num_examples_to_run)
        self.data = data

    def set_ukey(self, name):
        self.ukey = name

    def filter(self, data, quids_to_exclude, num_examples_to_run):
        if quids_to_exclude is not None:
            # print(data[0])
            data = [el for el in data if el[self.ukey] not in quids_to_exclude]
            
        if num_examples_to_run >= 0:
            data = data[:num_examples_to_run]
            
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


skill_mapping = {
    'A1': ['Video Caption'],
    'A2' : ['Detailed_Natural'], 
    'A3' : ['Short_Natural'],    
    'B1' : ['Subtitle'],
    'B2' : ['Audio'],
    'B3' : ['Music'],
    'C1' : ['3D Scene'],
    'C2' : ['3D Situated Reasoning'],
    'D1' : ['CT Caption'],
    'D2' : ['Medical Image Caption'],
    'E1' : ['general_ocr'],
    'E2' : ['poster_caption'],
    'E3': ['pdf_ocr'],
    'F1': ['chart_caption'],
    'F2' : ['table_caption'],
    'G1' : ['equation_caption'],  
    'G2' : ['mathgeo_caption'],  
}

skill_name_mapping = {
    'general_ocr' : 'OCR Result',
    'poster_caption' : 'Poster Caption',
    'poster_ocr' : 'Poster OCR',
    'pdf_ocr' : 'PDF OCR',
    'chart_caption': 'Chart Caption',
    'table_caption' : 'Table Caption',
    'equation_caption': 'Equation Caption',
    'mathgeo_caption' : 'Math & Geometry Caption',
    'mathgeo_tikz' : 'Math & Geometry LaTex', 
    'gui' : 'GUI caption'
    
}

def map_numbers_to_letters(n):
    return chr(65 + n)

from tqdm import tqdm
# video mmmu
class VideoMMMUDataset(BaseDataset):
    def __init__(self, args, quids_to_exclude=None, num_examples_to_run=-1):        
        self.set_ukey('q_uid')
        self.args = args
        self.clip_length = args.clip_length
        self.narration_list = self.get_description_list()  # uid --> list of str  or  uid --> str
        self.anno = self.get_anno()
        # self.durations = load_json(args.duration_path)  # uid --> float
        
        if os.path.exists(self.args.skill_path):
            self.skills = load_json(self.args.skill_path)
        else:
            self.skills = {}
            
        self.quids_to_exclude = quids_to_exclude

        data = self.build()
         
        data = self.filter(data, quids_to_exclude, num_examples_to_run)
        self.data = data

    def get_description_list(self):
        narration_list = os.listdir(self.args.caption_path) 
        return narration_list
    
    def get_descriptions(self, vid):
        with open(os.path.join(self.args.caption_path, vid + '.txt' ), 'r', encoding='utf-8') as file:
            lines = [line.strip().strip("'\"") for line in file if line.strip()]
        return lines

    
    def load_skill_caption(self, 
        skills, 
        vpath,
        vid,
        caption_time,
        subtitle_time):
        

        all_captions = {} 
        frame_caption_base = os.path.join(self.args.skill_caption_path, vpath)
        selected_frames = sorted(os.listdir(frame_caption_base))[::self.args.frame_cap_stride]
        
        # for debug
        # skills = ['A1', 'A2', 'A3', 'B1', 'B2', 'B3', 
        #           'C1', 'C2', 'D1', 'D2', 'E1', 'E2', 'E3',
        #           'F1', 'F2', 'G1', 'G2']
        
        for skill in skills:
            
            if skill in ['A1']:
                # print('here', skill) 
                cap = extract_captions(self.args.caption_path, vid.split('/')[-1], 
                                           self.args.stride, 
                                           self.args.clip_length,
                                           include_time=caption_time)
                
                all_captions[skill] = cap
            
            elif skill in ['A2', 'A3', 'E1', 'E2', 'E3',
                           'F1', 'F2', 'G1', 'G2']:
                
                all_captions[skill] = [] 
                for frame in selected_frames:
                    all_cap = load_json(os.path.join(self.args.skill_caption_path, vpath, frame))
                    all_cap = all_cap[vpath + '/' + frame[:-5]]
                    all_captions[skill].append(all_cap[skill_mapping[skill][0]]) 
            
            elif skill in ['D1', 'D2']:
                pass
            
            elif skill in ['B1']:
                
                subtitle = extract_subtitles_srt(self.args.subtitle_path, vid, include_time=subtitle_time)
                
                all_captions[skill] = subtitle
                
            
            elif skill in ['B2', 'B3']:
                pass 
            
            elif skill in ['C1', 'C2']:
                pass
        
        return all_captions

    def get_skills(self):
        result = load_json(self.args.rounter_result_path)
        return result 

    def format_narration(self, narr):
        if isinstance(narr, list):
            narr = '.\n'.join([f'{int(i*self.args.caption_every)}: {cap}' for i, cap in enumerate(narr)])
        return narr

    def get_anno(self):
        anno = load_json(self.args.anno_path) 
        return anno

    def build(self):
        data = []
        for item in tqdm(self.anno):
            
            if item['uid'] in self.quids_to_exclude:
                continue
            
            if item['question_type'] == 'open':
                continue
             
            if self.args.filter_by_skill:
                if self.skills[item['uid']] != self.args.targeted_skill:
                    continue
    
            if item['id']+'.txt' not in self.narration_list:
                continue
            
            
            
            video_id = item['vpath'].split('.')[0]
            
            # load skills / expert
            
            if ', ' in self.skills[item['uid']]:
                skills = self.skills[item['uid']].split(', ')
            elif ',' in self.skills[item['uid']]:
                skills = self.skills[item['uid']].split(',')
            else:
                skills = [self.skills[item['uid']]]
                
            # if skills != ['A1', 'G2']:
            #     continue
            
            # if skills != ['A1']:
            #     continue
            
            # skills = ['A1', 'B1', 'G2']
            
            subtitle_time = not self.args.subtitle_no_time
            caption_time = not self.args.caption_no_time
            
            # print(skills)
            
            skill_captions = self.load_skill_caption(skills,
                                                     item['vpath'], 
                                                     video_id, 
                                                     # subtitle
                                                     caption_time, 
                                                     subtitle_time
                                                     )
            
            # print(skill_captions)
        
            question = item['question']
            gt = item['answer']
            uid = item['uid']
            qid = item['id']
            
            options = ''
            for i, o in enumerate(item['options']):
                options += f"{map_numbers_to_letters(i)}: {o} \n"
            
            # question caption
            question_caption_path = os.path.join(self.args.image_caption_path, f"{item['id']}.txt")
            if not os.path.exists(question_caption_path):
                question_caption = ""
            else:
                with open(question_caption_path, 'r') as f:
                    question_caption = f.read()
            # print('question_caption', question_caption) 
            
            data.append({
                'id': qid,
                'q_uid': uid,
                
                'caption': skill_captions,
                'question_caption': question_caption,
                
                'question': question,
                'options': options,
                'truth': gt,
                
                'skill' : skills
            })
            
            # break
                        
        return data  

    def __getitem__(self, idx):
        
        data = self.data[idx]
        
        templates = {
            'A1': '''The video's captions are listed below. Each caption describes a {} seconds clip.\n{}\n''',
            'A2': '''The video keyframe's detailed captions are listed below. Each caption describes a keyframe.\n{}\n''',
            'A3': '''The video keyframe's short captions are listed below. Each caption describes a keyframe.\n{}\n''',
            'B1': '''The video's subtitles are listed below.\n{}\n''',
            'B2': '''The video's audio caption is listed below.\n{}\n''',
            'B3': '''The video's music caption is listed below.\n{}\n''',
            'C1': '''The video's 3D scene caption is listed below.\n{}\n''',
            'C2': '''The video's 3D situated reasoning caption is listed below.\n{}\n''',
            'D1': '''The video's CT caption is listed below.\n{}\n''',
            'D2': '''The video's medical image caption is listed below.\n{}\n''',
            'E1': '''The video's OCR result is listed below. Each caption describes a keyframe.\n{}\n''',
            'E2': '''The video's poster caption is listed below. Each caption describes a keyframe.\n{}\n''',
            'E3': '''The video's PDF OCR is listed below. Each caption describes a keyframe.\n{}\n''',
            'F1': '''The video's chart caption is listed below. Each caption describes a keyframe.\n{}\n''',
            'F2': '''The video's table caption is listed below. Each caption describes a keyframe.\n{}\n''',
            'G1': '''The video's equation caption is listed below. Each caption describes a keyframe.\n{}\n''',
            'G2': '''The video's math & geometry caption is listed below. Each caption describes a keyframe.\n{}\n''',
        }
        
        prompt = ""
        
        for k in data['skill']:
            
            if k in ['B2', 'B3', 'C1', 'C2', 'D1', 'D2']:
                continue
            
            v = data['caption'][k]
            
            if k in ['A1']:
                prompt += templates[k].format(self.clip_length, v)
            elif k in ['A2', 'A3', 'E1', 'E2', 'E3','F1', 'F2', 'G1', 'G2']:
                final_cap = ''
                for i, cap in enumerate(v):
                    start_time, end_time = format_seconds(i*self.args.frame_cap_stride*4), format_seconds((i+1)*self.args.frame_cap_stride*4)
                    # timestamped_captions += f"{start_time} --> {end_time}\n{vid_captions_list[i]}"
                    final_cap += f"{start_time}: \n{cap}\n"
                    # f"Timestamp {i*self.args.frame_cap_stride*4}s: {cap}\n"
                prompt += templates[k].format(final_cap)
                # prompt += templates[k].format(final_cap)
            elif k in ['B1']:
                prompt += templates[k].format(v)
            elif k in ['B2', 'B3', 'C1', 'C2', 'D1', 'D2']:
                continue
                      
        # print(prompt)
        
        if '<image 1>' in data['question']:
            qa_prompt = f"Answer the question based on all the provided information. If the options are given, please select the most accurate answer. In this case, please respond with only the letter (A, B, C, D, E, etc.) of the correct option. However, if the options are not given, please directly answer the question. In your final response, please only include only the answer without explanation.\n\nQuestion.\n{data['question']}\nBelow is the caption of <image 1>.\n{data['question_caption']}\n\nOptions.\n{data['options']}\n\nThe answer is:"
        else:
            qa_prompt = f"Select the best answer to the following multiple-choice question based on all the provided information. Respond with only the letter (A, B, C, D, E, etc.) of the correct option. In your final response, please only include only the answer without explanation.\n\nQuestion.\n{data['question']}\n\nOptions.\n{data['options']}\n\nThe answer is:"
                 
        prompt += qa_prompt
                 
        data['prompt'] = [prompt] 
         
        return data
        
# SQA3D
class MMAU(BaseDataset):
    def __init__(self, args, quids_to_exclude=None, num_examples_to_run=-1):        
        self.set_ukey('q_uid')
        self.args = args
        self.clip_length = args.clip_length
        self.narration_list = self.get_description_list()  # uid --> list of str  or  uid --> str
        self.anno = self.get_anno()
        
        # router selected skills 
        speech_caption = json.load(open('/nas-ssd2/jhyoon/mmau-caption-extraction/GAMA/speech-stage5_answers_qformer_all_mmau_1k_jy.json'))
        audio_caption = json.load(open('/nas-ssd2/jhyoon/mmau-caption-extraction/GAMA/audio-stage5_answers_qformer_all_mmau_1k_jy.json'))
        music_caption = json.load(open('/nas-ssd2/jhyoon/mmau-caption-extraction/GAMA/music-stage5_answers_qformer_all_mmau_1k_jy.json'))
        self.audio_caption = {}
        self.music_caption = {}
        self.speech_caption = {}
        
        for d in speech_caption:
            self.speech_caption[d['id']] = d
            
        for d in music_caption:
            self.music_caption[d['id']] = d
        
        for d in audio_caption:
            self.audio_caption[d['id']] = d
        
        
        if os.path.exists(self.args.skill_path):
            self.skills = load_json(self.args.skill_path)
        else:
            self.skills = {}

        data = self.build()
         
        data = self.filter(data, quids_to_exclude, num_examples_to_run)
        self.data = data

    def get_description_list(self):
        narration_list = os.listdir(self.args.caption_path) 
        return narration_list
        
    def load_skill_caption(self, skills, qid):
        
        skills = skills.split(', ')
        
        captions = {}
        
        for skill in skills:
            
            if skill == 'B2':
                caption = self.audio_caption[qid]['prediction']
                captions['Audio Caption'] = caption
            elif skill == 'B3':
                caption = self.music_caption[qid]['prediction']
                captions['Music Caption'] = caption
            elif skill == 'B4':
                caption = self.speech_caption[qid]['prediction']
                captions['Speech Caption'] = caption
            else:
                pass # implement later
        
        return captions 
                    

    def get_skills(self):
        result = load_json(self.args.rounter_result_path)
        return result 

    def get_anno(self):
        anno = load_json(self.args.anno_path) 
        return anno

    def build(self):
        data = []
        for item in tqdm(self.anno):
            
             
            if self.args.filter_by_skill:
                if self.skills[item['id']] != self.args.targeted_skill:
                    continue
            
            # if 'Adaptation' not in item['uid']:
            #     continue
    
            qid = str(item['id'])
            
            if len(self.skills) !=0:
            # skill based caption
                skill = self.skills[qid]
                skill_caption = self.load_skill_caption(skill, qid) 
            else:
                skill_caption = None
                
            
            if len(skill_caption) == 0:
                continue
             
            
            gt = item['choices'].index(item['answer'])
            # qtype = self.annotation[qid]['answer_type']
                
            
            question =  item['question']
            
            options = ''
            for i, o in enumerate(item['choices']):
                options += f"{map_numbers_to_letters(i)}: {o} \n"
                
                
            # gt = item['answer']
            
            # question caption
            
            data.append({
                'id': qid,
                'q_uid': qid,
                'caption': skill_caption,
                'question': question,
                'truth': gt,
                'skill' : skill,
                'options': options
            })
            
            # break
            
        return data  

    def __getitem__(self, idx):
        
        data = self.data[idx]
        
        prompt = '''You are a answerer for Audio Question Answering. Select the best answer to the following multiple-choice question based on all the provided information. Respond with only the letter (A, B, C, D, E, etc.) of the correct option. In your final response, please only include only the answer without explanation. The Information related to solve QA is listed below.\n\n{}\n\nAnswer the .\n\nQuestion.\n{}\n\n Options:\n{}\nThe answer is:
        '''
        
        caption = ''
        for k, v in data['caption'].items():
            caption += f"{k}: {v}\n"
            
        prompt = prompt.format(caption, data['question'], data['options'])        
        data['prompt'] = [prompt] 
         
        return data
        


class SQA3D(BaseDataset):
    def __init__(self, args, quids_to_exclude=None, num_examples_to_run=-1):        
        self.set_ukey('q_uid')
        self.args = args
        self.clip_length = args.clip_length
        self.narration_list = self.get_description_list()  # uid --> list of str  or  uid --> str
        self.anno = self.get_anno()
        
        # router selected skills 
        # YUI: TODO
        self.scene_caption = json.load(open('/nas-ssd2/shoubin/datasets/multimodal_symbolic_moe/sqa3d_scene_caption.json'))
        self.situated_caption = json.load(open('/nas-ssd2/shoubin/datasets/multimodal_symbolic_moe/sqa3d_situated_caption.json'))
        self.table_caption = None
        annotation = json.load(open('/nas-ssd2/yuezhang/Spartun3D/leo_data/annotations/instruction/sqa3d/v1_balanced_sqa_annotations_test_scannetv2.json'))
        anno_dict = {}
        for d in annotation['annotations']:
            anno_dict[str(d['question_id'])] = d
        
        self.annotation = anno_dict
        # self.durations = load_json(args.duration_path)  # uid --> float
        
        if os.path.exists(self.args.skill_path):
            self.skills = load_json(self.args.skill_path)
        else:
            self.skills = {}

        data = self.build()
         
        data = self.filter(data, quids_to_exclude, num_examples_to_run)
        self.data = data

    def get_description_list(self):
        narration_list = os.listdir(self.args.caption_path) 
        return narration_list
        
    def load_skill_caption(self, skills, qid):
        
        skills = skills.split(', ')
        
        captions = {}
        
        for skill in skills:
            
            if skill == 'C1':
                caption = self.scene_caption[qid]['response']
                captions['3D Scene Caption'] = caption
            elif skill == 'C2':
                caption = self.situated_caption[qid]['response']
                captions['Situation Caption'] = caption
            else:
                pass # implement later
        
        return captions 
                    

    def get_skills(self):
        result = load_json(self.args.rounter_result_path)
        return result 

    def get_anno(self):
        anno = load_json(self.args.anno_path) 
        return anno

    def build(self):
        data = []
        for item in tqdm(self.anno):
            
             
            if self.args.filter_by_skill:
                if self.skills[item['uid']] != self.args.targeted_skill:
                    continue
            
            # if 'Adaptation' not in item['uid']:
            #     continue
    
            qid = str(item['question_id'])
            
            if len(self.skills) !=0:
            # skill based caption
                skill = self.skills[qid]
                skill_caption = self.load_skill_caption(skill, qid) 
            else:
                skill_caption = None
                
            if skill not in ['C1, C2', 'C1', 'C2']:
                continue
            
            
            if len(skill_caption) == 0:
                continue
             
            # if self.args.no_skill_cap:
            #     skill = self.args.targeted_skill
            #     skill_caption = self.load_skill_caption(skill, qid , self.args.skill_cap_stride) 
                
            
            gt = self.annotation[qid]['answers']
            # qtype = self.annotation[qid]['answer_type']
                
            
            question = item['situation'] + ' ' + item['question']
            # gt = item['answer']
            
            # question caption
            
            data.append({
                'id': qid,
                'q_uid': qid,
                'caption': skill_caption,
                'question': question,
                'truth': gt,
                'skill' : skill
            })
            
            # break
            
        return data  

    def __getitem__(self, idx):
        
        data = self.data[idx]
        
        prompt = '''You are a answerer for 3D situated Question Answering. The Information related to solve QA is listed below.\n\n{}\n\nAnswer the question using a single word or phase.\n\nQuestion.\n{}\n\nThe answer is:
        '''
        
        caption = ''
        for k, v in data['caption'].items():
            caption += f"{k}: {v}\n"
            
        prompt = prompt.format(caption, data['question'])        
        data['prompt'] = [prompt] 
         
        return data

# YUI: TODO: copy-paste M3D dataset from Yue


def get_dataset(args, quids_to_exclude=None, num_examples_to_run=-1):

    if args.dataset == 'videommmu':
        return VideoMMMUDataset(args, quids_to_exclude=quids_to_exclude, num_examples_to_run=num_examples_to_run)
    
    elif args.dataset == 'sqa3d':
        return SQA3D(args, quids_to_exclude=quids_to_exclude, num_examples_to_run=num_examples_to_run)
    elif args.dataset == 'mmau':
        return MMAU(args, quids_to_exclude=quids_to_exclude, num_examples_to_run=num_examples_to_run) 
    elif args.dataset == 'm3d':
        return M3D(args, quids_to_exclude=quids_to_exclude, num_examples_to_run=num_examples_to_run) 
    else:
        raise NotImplementedError()


if __name__ == '__main__':
    
    import argparse
    
    parser = argparse.ArgumentParser("")
    parser.add_argument("--dataset", default='videommmu', type=str)
    parser.add_argument("--anno_path", default='/nas-ssd2/shoubin/datasets/multimodal_symbolic_moe/video-mmmu.json', type=str)  
    parser.add_argument("--caption_path", default='/nas-ssd2/video_datasets/captions/VILA_cap/8_yb_mmmu_captions/', type=str) 
    
    # parser.add_argument("--dataset", default='mmau', type=str)
    # parser.add_argument("--anno_path", default='/nas-ssd2/jhyoon/mmau-caption-extraction/GAMA/speech-stage5_answers_qformer_all_mmau_1k_jy.json', type=str)  
    # parser.add_argument("--caption_path", default='/nas-ssd2/video_datasets/captions/VILA_cap/8_yb_mmmu_captions/', type=str)
     
    # parser.add_argument("--dataset", default='sqa3d', type=str) 
    # parser.add_argument("--anno_path", default='/nas-ssd2/shoubin/datasets/multimodal_symbolic_moe/sqa3d-sub.json', type=str) 
    # parser.add_argument("--caption_path", default='/nas-ssd2/video_datasets/captions/VILA_cap/8_yb_mmmu_captions/', type=str)   
 
    parser.add_argument("--caption_every", default=8, type=int)
    parser.add_argument("--num_examples_to_run", default=-1, type=int)
    parser.add_argument("--clip_length", default=8, type=int) 
    parser.add_argument("--stride", default=1, type=int) 
    parser.add_argument("--subtitle_path", default="/nas-ssd2/video_datasets/captions/video_mmmu_subtitles/", type=str)
    parser.add_argument("--skill_path", default="/nas-ssd2/shoubin/code/multimodal-smoe/MSMOE/skills/videommmu_skill_id_gpt4o.json", type=str) 
    
     
    parser.add_argument("--skill_caption_path", default="/nas-ssd2/shoubin/caption/videommmu/", type=str)
    parser.add_argument("--image_caption_path", default="/nas-ssd2/shoubin/code/multimodal-smoe/LLoVi/data/videommmu/question_captions", type=str)
    parser.add_argument("--frame_cap_stride", default=8, type=int)
     
    parser.add_argument("--caption_no_time", action='store_true')
    parser.add_argument("--subtitle_no_time", action='store_true')
    parser.add_argument("--filter_by_skill", action='store_true')
    parser.add_argument("--targeted_skill", default="", type=str)
    
    
    args = parser.parse_args()
    
    dataset = get_dataset(args, num_examples_to_run=args.num_examples_to_run)
    
    print(len(dataset))
    for data in dataset:
        print(data['prompt'][0])
        break
