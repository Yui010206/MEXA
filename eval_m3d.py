import json
import os


with open("m3d_balanced_subset.json") as f_in1:
    m3d_test = json.load(f_in1)

    
task_dict = {str(item["pid"]): item["Question Type"] for item in m3d_test}
gt_answer_dict = {}
for item in m3d_test:
    gt_answer_dict[str(item["pid"])] = item["Answer Choice"]

save_path = ''

files = os.listdir(save_path)
type_eval = {"1":0, "2":0, "3":0, "4":0, "5":0}
type_eval_pred = {"1":0, "2":0, "3":0, "4":0, "5":0}
count = 0
for file in files:
    if 'json' not in file:
        continue
    count += 1
    result = json.load(open(os.path.join(save_path, file)))
    q_id = file[:-5]
    predict_answer = result["response"]
    gt_answer = gt_answer_dict[q_id]
    q_type = task_dict[q_id]
    if predict_answer == gt_answer:
        type_eval_pred[str(q_type)] += 1
    type_eval[str(q_type)] += 1
result = {}
for each_type in ['1', '2', '3', '4', '5']:
    result[each_type] = type_eval_pred[each_type]/type_eval[each_type]
avg = sum(list(result.values()))/5
print("")


 