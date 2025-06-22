import os
from pathlib import Path
from utils import save_json, load_json, save_pkl, load_pkl, makedir
from dataset import *
from model import get_model
from tqdm import tqdm
from pprint import pprint
from concurrent.futures import ThreadPoolExecutor, as_completed
import concurrent.futures
import time
import shutil
import argparse


def parse_args():
    parser = argparse.ArgumentParser("")

    # data
    parser.add_argument("--dataset", default='videommmu', type=str)  # videomme, sqa3d, mmau, m3d
    parser.add_argument("--caption_path", default="", type=str) 
    parser.add_argument("--subtitle_path", default="", type=str) 
    parser.add_argument("--audio_caption_path", default="", type=str) 
    parser.add_argument("--anno_path", default='', type=str)
    parser.add_argument("--caption_every", default=8, type=int)
    parser.add_argument("--fps", default=1, type=int)
    
    parser.add_argument("--skill_path", default="", type=str) 
    parser.add_argument("--skill_caption_path", default="/nas-ssd2/shoubin/caption/videommmu/", type=str)
    parser.add_argument("--frame_cap_stride", default=2, type=int)
    
    
    parser.add_argument("--filter_by_skill", action='store_true')
    parser.add_argument("--targeted_skill", default="", type=str) 
    
    
    parser.add_argument("--clip_length", default=64, type=int) 
    parser.add_argument("--stride", default=1, type=int) 
    parser.add_argument("--num_examples_to_run", default=-1, type=int)

    # prompt
    parser.add_argument("--prompt_type", default="videommmu", type=str)
    parser.add_argument("--caption_no_time", action='store_true')
    parser.add_argument("--subtitle_no_time", action='store_true')
    parser.add_argument("--no_skill_cap", action='store_true')

    # output
    parser.add_argument("--output_base_path", default='./debug', type=str)  

    # model
    parser.add_argument("--model", default="deepseek-reasoner", type=str)
    parser.add_argument("--endpoint", default="", type=str)
    parser.add_argument("--api_key", default="", type=str)
    parser.add_argument("--api_url", default="", type=str)
    parser.add_argument("--openai_api_key", default="", type=str)

    
    # videommmu
    parser.add_argument("--image_caption_path", default="", type=str) 



    # eval
    ## backup pred
    parser.add_argument("--backup_path", default="", type=str)

    # other
    parser.add_argument("--hf_token", default="", type=str)
    parser.add_argument("--single_process", action='store_true')
    parser.add_argument("--num_workers", default=32, type=int) 
    parser.add_argument("--time_sleep", default=0, type=float) 
    parser.add_argument("--disable_infer", action='store_true')
    parser.add_argument("--disable_eval", action='store_true')
    parser.add_argument("--from_scratch", action='store_true')

    return parser.parse_args()





def process_one(output_path, model, prompt_type, item, fps, caption_every, time_sleep=1):
    # if len(item['caption']) == 0:
    #     return

    prompt = item['prompt'] #get_prompt(prompt_type, item, fps, caption_every)
    # print(prompt[0])
     
    # pred = model.forward("", prompt)
    
    try:
        pred = model.forward("", prompt)
    except Exception as e:
        output_error_path = Path(output_path).parent / 'error'
        makedir(str(output_error_path))
        item['prompt'] = prompt
        save_json(item, os.path.join(output_error_path, f"{item['q_uid']}.json"))
        print(f"Error in processing {item['q_uid']}: {e}")
        return
    
    output = {}
    output['prompt'] = prompt
    output.update(item)
    output.update(pred)
    if 'message' in output:
        del output['message']
    if 'subtitle' in output:
        del output['subtitle']
    if 'caption' in output:
        del output['caption']
    save_json(output, os.path.join(output_path, f"{item['q_uid']}.json"))
    
    # Sleep briefly between requests
    time.sleep(time_sleep)


def launch():
    args = parse_args()
    pprint(args)
    # os.environ["HF_TOKEN"] = args.hf_token
    # os.environ["OPENAI_API_KEY"] = args.openai_api_key

    # output
    makedir(args.output_base_path)
    output_path = os.path.join(args.output_base_path, 'logs')
    makedir(output_path)

    # save args
    save_json(vars(args), os.path.join(args.output_base_path, 'config.json'))

    # check processed questions
    processed_indices = []
    if not args.from_scratch:
        for filename in os.listdir(output_path):
            if filename.endswith('.json'):
                try:
                    vid_idx = filename.split('.')[0]
                except ValueError:
                    continue
                
                processed_indices.append(vid_idx)
                
    # get input
    print(processed_indices, len(processed_indices))
    dataset = get_dataset(args, quids_to_exclude=processed_indices, num_examples_to_run=args.num_examples_to_run)    
    
    if not args.disable_infer:
        # get model
        model = get_model(args)
        
        # answer
        if args.single_process:
            for item in tqdm(dataset):
                
                process_one(output_path, model, args.prompt_type, item, args.fps, args.caption_every, time_sleep=args.time_sleep)
                # import pdb; pdb.set_trace()
        
        else:
            max_inflight = args.num_workers # + 30  # Adjust this as needed
            futures = []
            pbar = tqdm(total=len(dataset))
            with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
                for i, item in enumerate(dataset):
                    pbar.update(1)
                    # print(f"{i} / {len(dataset)}")
                    # If too many tasks are in flight, wait for some to finish
                    while len(futures) >= max_inflight:
                        done, not_done = concurrent.futures.wait(futures, return_when=concurrent.futures.FIRST_COMPLETED)
                        futures = list(not_done)

                    future = executor.submit(process_one, output_path, model, args.prompt_type, item, args.fps, args.caption_every, time_sleep=args.time_sleep)
                    futures.append(future)
                # Wait for any remaining futures
                print('waiting for remaining features to complete')
                for future in concurrent.futures.as_completed(futures):
                    try:
                        future.result()
                    except Exception as e:
                        print(f"[Error] {e}")
            pbar.close()


if __name__ == '__main__':
    launch()