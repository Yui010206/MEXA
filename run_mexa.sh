# get skills
python get_skills.py \
--dataset videommmu \
--data_path ./video-mmmu.json \
--video_base ./VideoMMMU/ \
--save_base ./skills/ 

# get response with deepseek-reasoner
python main.py --dataset videommmu \
--output_base_path ./mexa_videommmu/ \
--caption_path ./ \
--subtitle_path ./ \
--image_caption_path ./ \
--skill_path ./videommmu_skill_id_gpt4o.json \
--skill_caption_path ./videommmu/ \
--frame_cap_stride 4 \
--num_workers 4 \
--clip_length 8 \
--caption_every 8 \
--stride 1 \
--fps 1 \
--model deepseek-reasoner \
--api_key KEY \
--api_url URL \
--endpoint ENDPOINT \
--prompt_type videommmu \
--anno_path ./video-mmmu.json
